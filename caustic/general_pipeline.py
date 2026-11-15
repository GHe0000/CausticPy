"""One automatic pipeline for a convex lens mask and square raster target."""
from pathlib import Path
from dataclasses import replace
import contextlib,sys,json,time,hashlib,shutil,traceback,logging
import numpy as np
from . import backend
from .settings import RunSettings
from .target import load,center_pattern,save_gray
from .domain import MaskDomain
from .transport import solve
from .optimize import optimize
from .adaptive import refine_mesh
from .verify import trace
from .export import export
from .report import plots
from .visualize import figure
ROOT=Path(__file__).resolve().parents[1]
def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,
                    default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
@contextlib.contextmanager
def log_stage(name):
    started=time.monotonic();logging.info('%s | START',name)
    try:
        yield
    except BaseException:
        logging.exception('%s | FAILED',name)
        raise
    else:logging.info('%s | DONE (%.1fs)',name,time.monotonic()-started)

def quality(m):
    basic=m['dark_leak_captured']<.02 and m['rmse_relative']<.35 and m['tir']==0 and m['captured_flux']>.98
    fine=m['thin_flux_ratio'] is None or .8<=m['thin_flux_ratio']<=1.2
    return dict(basic_imaging_pass=basic,thin_detail_pass=fine,optical_quality_pass=basic and fine)
def run(settings:RunSettings):
    settings.validate();out=Path(settings.out).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must be within JAX3')
    out.mkdir(parents=True,exist_ok=False)
    settings.image=str(Path(settings.image).resolve());settings.mask=str(Path(settings.mask).resolve());settings.out=str(out)
    started=time.monotonic()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s PID %(process)d | %(message)s',
                        datefmt='%H:%M:%S',handlers=[logging.StreamHandler(sys.stdout),
                        logging.FileHandler(out/'run.log',encoding='utf-8')],force=True)
    try:
        write(out/'settings.json',settings.to_dict());write(out/'environment.json',backend.environment())
        logging.info(f'Python {sys.executable}; CPU threads={settings.threads or __import__("os").cpu_count()}; JAX float64={backend.jax.config.jax_enable_x64}')
        logging.info(f'Physics: flat entrance z=0; screen z={settings.distance_mm:g} mm; n={settings.refractive_index:g}/{settings.air_index:g}; mask L={settings.L_mm:g} mm')
        with log_stage('[1] Square input validation and convex mask'):
            raw=load(settings.image)
            if raw.shape[0]!=raw.shape[1] or raw.shape[0]<2:raise ValueError(f'Target must be a square PNG of at least 2x2, got {raw.shape}; no implicit resize/padding')
            gray,shift=center_pattern(raw) if settings.center_target else (raw,(0,0))
            domain=MaskDomain(settings.mask,settings.L_mm,settings.mask_threshold)
            config=settings.optical_config(gray.shape[1]);config.save(out/'config.json')
            write(out/'workflow.json',settings.refinement_plan())
            save_gray(out/'processed_target.png',gray);save_gray(out/'mask_binary.png',domain.mask.astype(float))
            np.savez_compressed(out/'domain.npz',vertices=domain.vertices,directions=domain.directions,offsets=domain.offsets)
            write(out/'domain.json',domain.metadata())
            write(out/'input.json',dict(image=settings.image,mask=settings.mask,shape=list(gray.shape),pixel_shift=shift,
                 image_sha256=hashlib.sha256(Path(settings.image).read_bytes()).hexdigest(),
                 mask_sha256=hashlib.sha256(Path(settings.mask).read_bytes()).hexdigest(),
                 illumination='uniform parallel +z light restricted to the convex lens aperture'))
            logging.info(f'Input {gray.shape}; center shift {shift}; mask area {domain.area:.6f} mm^2; grid spacing L/n={settings.L_mm/settings.mesh_n:.6f} mm')
        with log_stage('[2] SDOT source capacities and target power diagram'):
            mapping,ot=solve(gray,config,out,source_domain=domain)
            write(out/'transport.json',ot)
            logging.info(f"OT mass L1 residual={ot['capacity_L1']:.3e}; source mass={ot['source_mass']:.15g}; target mass={ot['target_mass']:.15g}")
        base=out/'base';base.mkdir();config.save(base/'config.json')
        def factory(c,n):
            with log_stage(f'Mesh n={n}, spacing={settings.L_mm/n:.6f} mm'):
                mesh=domain.mesh(c,n)
                logging.info(f'Mesh: {len(mesh[0]):,} vertices / {len(mesh[1]):,} triangles')
                return mesh
        with log_stage('[3] Multi-resolution JAX/Optax surface design'):
            xy,f,b,z,history=optimize(mapping,gray,config,base,mesh_factory=factory)
        candidates=[]
        def evaluate(folder,c,state,hist,label):
            xx,ff,bb,zz=state;c.save(folder/'config.json')
            with log_stage(f'{label}: CNC solid export'):
                mesh=export(xx,ff,bb,zz,c,folder);mesh['body_count']=1
                write(folder/'mesh.json',mesh)
            with log_stage(f'{label}: independent forward raytrace ({2**c.trace_power:,} rays)'):
                metrics,I,T=trace(xx,ff,zz,gray,c,folder,source_domain=domain)
                write(folder/'metrics.json',metrics)
            with log_stage(f'{label}: save diagnostic figures'):
                plots(xx,ff,zz,mapping,hist,I,T,folder,c)
            q=quality(metrics);write(folder/'status.json',dict(pipeline_complete=True,**q,cnc_release=False))
            logging.info(f"{label}: leak={metrics['dark_leak_captured']:.3%}; RMSE={metrics['rmse_relative']:.5f}; thin flux={metrics['thin_flux_ratio']}; optical gate={q['optical_quality_pass']}")
            row=dict(directory=str(folder),label=label,metrics=metrics,mesh=mesh,**q);candidates.append(row)
            return row
        evaluate(base,config,(xy,f,b,z),history,'base')
        for index,stage in enumerate(settings.refinement_plan(),1):
            name=stage.get('name',f'refine_{index}')
            if Path(name).name!=name or name in ['.','..','base']:raise ValueError('Invalid stage name')
            folder=out/name;folder.mkdir()
            c=replace(config,mesh_levels=[settings.mesh_n],fit_iterations=0,
                      image_iterations=int(stage.get('iterations',200)),image_resolution=max(16,int(stage.get('resolution',0)) or gray.shape[1]),
                      thin_weight=float(stage.get('thin_weight',60)),mapping_anchor=float(stage.get('mapping_anchor',.003)))
            if 'fine_samples' in stage:c.fine_quadrature=int(stage['fine_samples'])
            c.validate()
            with log_stage(f'[4.{index}] Local mesh refinement and physical image optimization'):
                rounds=int(stage.get('adaptive',0))
                if rounds:
                    xy,f,b,z,_,record=refine_mesh(xy,f,z,b,gray,c,rounds)
                    write(folder/'adaptive.json',record)
                    logging.info(f'After local refinement: {len(xy):,} vertices / {len(f):,} triangles')
                from scipy.spatial import Delaunay
                xy,f,b,z,history=optimize(mapping,gray,c,folder,initial=(xy,z,f),mesh_override=(xy,f,b,Delaunay(xy)))
            evaluate(folder,c,(xy,f,b,z),history,name)
        eligible=[r for r in candidates if r['optical_quality_pass']]
        selected=min(eligible,key=lambda r:r['metrics']['rmse_relative']) if eligible else min(candidates,key=lambda r:r['metrics']['target_l1'])
        # Selection is based only on independent renders. Retain every stage.
        write(out/'candidates.json',candidates)
        selected_folder=Path(selected['directory'])
        # Root-level files are the chosen model and images, ready for the user.
        for name in ['lens.stl','lens_precision.stl','lens.obj','target.png','simulated.png','error.png',
                     'comparison.png','height.png','mapping.png','loss.png','loss.csv','metrics.json','mesh.json',
                     'raytrace.npz','config.json']:
            source=selected_folder/name
            if source.exists():shutil.copy2(source,out/name)
        result=dict(pipeline_complete=False,final_directory=selected['directory'],selection='lowest RMSE among passing candidates, otherwise lowest L1',
                    **quality(selected['metrics']),metrics=selected['metrics'],mesh=selected['mesh'],cnc_release=False,
                    elapsed_seconds=time.monotonic()-started,model=str(out/'lens.stl'),precision_model=str(out/'lens_precision.stl') if (out/'lens_precision.stl').exists() else None)
        write(out/'result.json',result);write(out/'status.json',result)
        from .visualize import load_result
        with log_stage('[5] Height-colored mesh and imaging overview'):
            import matplotlib.pyplot as plt
            data=load_result(out);fig=figure(*data[:-1],data[-1])
            fig.savefig(out/'overview.png',dpi=160)
            from matplotlib.transforms import Bbox
            renderer=fig.canvas.get_renderer()
            box=Bbox.union([ax.get_tightbbox(renderer) for ax in fig.axes[:2]]).transformed(fig.dpi_scale_trans.inverted())
            box=Bbox.from_extents(box.x0-.12,box.y0-.12,box.x1+.35,box.y1+.12)
            fig.savefig(out/'mesh.png',dpi=160,bbox_inches=box,pad_inches=.06)
            plt.close(fig)
        result['pipeline_complete']=True
        result['elapsed_seconds']=time.monotonic()-started
        write(out/'result.json',result);write(out/'status.json',result)
        logging.info(f'[6] FINISHED. Selected model: {selected["directory"]}')
        logging.info(f'Numerical imaging gate={result["optical_quality_pass"]}; exact tool-radius/finishing validation is not included')
        return result
    except BaseException:
        write(out/'status.json',dict(pipeline_complete=False,error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        raise
