"""Terminal-only lens design: one complete JSON + unchanged numerical modules."""
from pathlib import Path
import argparse,contextlib,hashlib,json,logging,os,shutil,sys,time,traceback

ROOT=Path(__file__).resolve().parent

def load_run_config(path=None):
    path=Path(path).resolve() if path else ROOT/'configs/general_high.json'
    settings=json.loads(path.read_text(encoding='utf-8'))
    keys={'image','mask','out','refractive_index','air_index','L_mm','distance_mm','target_size_mm',
          'mesh_n','threads','center_target','mask_threshold','coarse_mesh_levels','optics','stages'}
    missing=keys-set(settings);unknown=set(settings)-keys
    if missing:raise ValueError('Missing configuration keys: '+', '.join(sorted(missing)))
    if unknown:raise ValueError('Unknown configuration keys: '+', '.join(sorted(unknown)))
    for key in ['image','mask','out']:
        if settings[key] and not Path(settings[key]).is_absolute():
            settings[key]=str((path.parent/settings[key]).resolve())
    return settings

def validate_settings(s):
    for name in ['L_mm','distance_mm','target_size_mm']:
        if s[name]<=0:raise ValueError(name+' must be positive')
    if not s['refractive_index']>s['air_index']>0:raise ValueError('Need refractive_index > air_index > 0')
    if not isinstance(s['mesh_n'],int) or s['mesh_n']<8:raise ValueError('mesh_n must be an integer >=8')
    if s['threads']<0:raise ValueError('threads must be >=0; 0 means all available cores')
    if not 0<=s['mask_threshold']<255:raise ValueError('mask_threshold must be in [0,254]')
    if not isinstance(s['stages'],list):raise ValueError('stages must be a list; [] disables refinement')
    if not isinstance(s['optics'],dict):raise ValueError('optics must be an object')
    if not isinstance(s['coarse_mesh_levels'],list) or not all(isinstance(n,int) and n>=8 for n in s['coarse_mesh_levels']):
        raise ValueError('coarse_mesh_levels must be a list of integers >=8; [] uses only mesh_n')

def optical_config(s,pixels):
    from caustic.config import Config
    values=dict(s['optics'])
    controlled={'diameter_mm','n_lens','n_air','screen_z_mm','target_width_mm','center_target','mesh_levels'}
    if controlled.intersection(values):raise ValueError('Set physical parameters at the top level, not in optics')
    for key in ['ot_resolution','image_resolution']:values[key]=max(16,values[key] or pixels)
    levels=s['coarse_mesh_levels']+[s['mesh_n']]
    values.update(diameter_mm=s['L_mm'],n_lens=s['refractive_index'],n_air=s['air_index'],
                  screen_z_mm=s['distance_mm'],target_width_mm=s['target_size_mm'],center_target=s['center_target'],
                  mesh_levels=sorted(set(min(n,s['mesh_n']) for n in levels)))
    c=Config(**values)
    if not 0<c.min_thickness_mm<c.base_mm<c.max_thickness_mm<s['distance_mm']:
        raise ValueError('Need 0 < min_thickness_mm < base_mm < max_thickness_mm < distance_mm; set thickness values in optics')
    c.validate()
    return c

def write(path,data):
    import numpy as np
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
def run(settings):
    import numpy as np
    from dataclasses import replace
    from caustic import backend
    from caustic.target import load,center_pattern,save_gray
    from caustic.domain import MaskDomain
    from caustic.transport import solve
    from caustic.optimize import optimize
    from caustic.adaptive import refine_mesh
    from caustic.verify import trace
    from caustic.export import export
    from caustic.report import plots
    from caustic.visualize import figure
    out=Path(settings['out']).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must be within the program folder')
    out.mkdir(parents=True,exist_ok=False)
    settings['image']=str(Path(settings['image']).resolve());settings['mask']=str(Path(settings['mask']).resolve());settings['out']=str(out)
    started=time.monotonic()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s PID %(process)d | %(message)s',
                        datefmt='%H:%M:%S',handlers=[logging.StreamHandler(sys.stdout),
                        logging.FileHandler(out/'run.log',encoding='utf-8')],force=True)
    try:
        write(out/'settings.json',settings);write(out/'environment.json',backend.environment())
        logging.info(f'Python {sys.executable}; CPU threads={settings['threads'] or __import__("os").cpu_count()}; JAX float64={backend.jax.config.jax_enable_x64}')
        logging.info(f'Physics: flat entrance z=0; screen z={settings['distance_mm']:g} mm; n={settings['refractive_index']:g}/{settings['air_index']:g}; mask L={settings['L_mm']:g} mm')
        with log_stage('[1] Square input validation and convex mask'):
            raw=load(settings['image'])
            if raw.shape[0]!=raw.shape[1] or raw.shape[0]<2:raise ValueError(f'Target must be a square PNG of at least 2x2, got {raw.shape}; no implicit resize/padding')
            gray,shift=center_pattern(raw) if settings['center_target'] else (raw,(0,0))
            domain=MaskDomain(settings['mask'],settings['L_mm'],settings['mask_threshold'])
            config=optical_config(settings,gray.shape[1]);config.save(out/'config.json')
            write(out/'workflow.json',settings['stages'])
            save_gray(out/'processed_target.png',gray);save_gray(out/'mask_binary.png',domain.mask.astype(float))
            np.savez_compressed(out/'domain.npz',vertices=domain.vertices,directions=domain.directions,offsets=domain.offsets)
            write(out/'domain.json',domain.metadata())
            write(out/'input.json',dict(image=settings['image'],mask=settings['mask'],shape=list(gray.shape),pixel_shift=shift,
                 image_sha256=hashlib.sha256(Path(settings['image']).read_bytes()).hexdigest(),
                 mask_sha256=hashlib.sha256(Path(settings['mask']).read_bytes()).hexdigest(),
                 illumination='uniform parallel +z light restricted to the convex lens aperture'))
            logging.info(f'Input {gray.shape}; center shift {shift}; mask area {domain.area:.6f} mm^2; grid spacing L/n={settings['L_mm']/settings['mesh_n']:.6f} mm')
        with log_stage('[2] SDOT source capacities and target power diagram'):
            mapping,ot=solve(gray,config,out,source_domain=domain)
            write(out/'transport.json',ot)
            logging.info(f"OT mass L1 residual={ot['capacity_L1']:.3e}; source mass={ot['source_mass']:.15g}; target mass={ot['target_mass']:.15g}")
        base=out/'base';base.mkdir();config.save(base/'config.json')
        def factory(c,n):
            with log_stage(f'Mesh n={n}, spacing={settings['L_mm']/n:.6f} mm'):
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
        for index,stage in enumerate(settings['stages'],1):
            name=stage.get('name',f'refine_{index}')
            if Path(name).name!=name or name in ['.','..','base']:raise ValueError('Invalid stage name')
            folder=out/name;folder.mkdir()
            c=replace(config,mesh_levels=[settings['mesh_n']],fit_iterations=0,
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
        from caustic.visualize import load_result
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

def main():
    p=argparse.ArgumentParser(description='Convex mask + square PNG -> lens STL and result images',
                              epilog='One complete run JSON. CLI values override its parameters.')
    p.add_argument('--config',help='Complete run JSON (default: configs/general_high.json beside the script)')
    p.add_argument('--image','--target',dest='image')
    p.add_argument('--mask')
    p.add_argument('--out')
    p.add_argument('--n','--refractive-index',dest='refractive_index',type=float)
    p.add_argument('--n-air',dest='air_index',type=float)
    p.add_argument('--L','--length',dest='L_mm',type=float)
    p.add_argument('--distance','--focal-length',dest='distance_mm',type=float)
    p.add_argument('--target-size',dest='target_size_mm',type=float)
    p.add_argument('--grid','--mesh-n',dest='mesh_n',type=int)
    p.add_argument('--threads',type=int)
    p.add_argument('--no-center',action='store_true')
    args=p.parse_args()
    settings=load_run_config(args.config)
    for key in ['image','mask','out','refractive_index','air_index','L_mm','distance_mm','target_size_mm','mesh_n','threads']:
        if getattr(args,key) is not None:settings[key]=getattr(args,key)
    if args.no_center:settings['center_target']=False
    if not all(settings[key] for key in ['image','mask','out']):p.error('--image, --mask and --out are required, directly or in JSON')
    validate_settings(settings)

    # Configure CPU libraries before importing NumPy/JAX.
    count=settings['threads'] or (os.cpu_count() or 1)
    for key in ['CAUSTIC_THREADS','LOOM_NB_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key]=str(count)
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    os.environ['JAX_ENABLE_X64']='true'
    os.environ['XLA_FLAGS']=f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={count}'
    compiler=Path('D:/Software/msys2/ucrt64/bin/g++.exe')
    if os.name=='nt' and compiler.exists():
        os.environ.setdefault('CXX',str(compiler))
        os.environ['PATH']=str(Path(sys.executable).parent)+';'+str(compiler.parent)+';'+os.environ.get('PATH','')
    run(settings)

if __name__=='__main__':
    main()
