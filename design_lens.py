"""Terminal-only lens design: one complete JSON + unchanged numerical modules."""
from pathlib import Path
import argparse,contextlib,hashlib,json,logging,os,shutil,sys,time,traceback

from runtime import resolve_runtime,validate_runtime,configure_runtime

ROOT=Path(__file__).resolve().parent

def load_run_config(path=None):
    path=Path(path).resolve() if path else ROOT/'configs/general_high.json'
    settings=json.loads(path.read_text(encoding='utf-8'))
    keys={'image','mask','out','refractive_index','air_index','L_mm','distance_mm','target_size_mm',
          'mesh_n','center_target','mask_threshold','coarse_mesh_levels','optics','stages','solver','runtime'}
    missing=(keys-{'runtime'})-set(settings);unknown=set(settings)-keys
    if missing:raise ValueError('Missing configuration keys: '+', '.join(sorted(missing)))
    if unknown:raise ValueError('Unknown configuration keys: '+', '.join(sorted(unknown)))
    for key in ['image','mask','out']:
        if settings[key]:
            value=Path(os.path.expandvars(settings[key])).expanduser()
            settings[key]=str((path.parent/value).resolve() if not value.is_absolute() else value.resolve())
    settings['runtime']=resolve_runtime(settings.get('runtime',{}),path.parent)
    return settings

def validate_settings(s):
    for name in ['L_mm','distance_mm','target_size_mm']:
        if s[name]<=0:raise ValueError(name+' must be positive')
    if not s['refractive_index']>s['air_index']>0:raise ValueError('Need refractive_index > air_index > 0')
    if not isinstance(s['mesh_n'],int) or s['mesh_n']<8:raise ValueError('mesh_n must be an integer >=8')
    validate_runtime(s.get('runtime',{}))
    if not 0<=s['mask_threshold']<255:raise ValueError('mask_threshold must be in [0,254]')
    if not isinstance(s['stages'],list):raise ValueError('stages must be a list; [] disables refinement')
    if not isinstance(s['optics'],dict):raise ValueError('optics must be an object')
    if not isinstance(s['solver'],dict):raise ValueError('solver must be an object')
    if not isinstance(s['coarse_mesh_levels'],list) or not all(isinstance(n,int) and n>=8 for n in s['coarse_mesh_levels']):
        raise ValueError('coarse_mesh_levels must be a list of integers >=8; [] uses only mesh_n')

def optical_config(s,pixels):
    from caustic.config import Config
    values=dict(s['optics'])
    values['solver']=s['solver']
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

def no_total_internal_reflection(metrics):
    # Check both traced rays and every facet's Snell discriminant.
    return metrics['tir']==0 and metrics['min_snell_discriminant']>0

def select_best(candidates):
    import math
    eligible=[row for row in candidates if no_total_internal_reflection(row['metrics'])
              and math.isfinite(row['metrics']['rmse_relative'])]
    return min(eligible,key=lambda row:row['metrics']['rmse_relative'],default=None)

def run(settings):
    from dataclasses import asdict,replace
    from caustic import backend
    from caustic.target import load,center_pattern
    from caustic.domain import MaskDomain
    from caustic.transport import solve
    from caustic.optimize import optimize
    from caustic.adaptive import refine_mesh
    from caustic.verify import trace
    from caustic.export import export
    from caustic.report import plots

    out=Path(settings['out']).resolve()
    out.mkdir(parents=True,exist_ok=False)
    for key in ['image','mask','out']:settings[key]=str(Path(settings[key]).resolve())
    started=time.monotonic()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(message)s',datefmt='%H:%M:%S',stream=sys.stdout,force=True)
    candidates=[]
    try:
        logging.info('Python %s; devices=%s; threads=%s; JAX float64=%s',
                     sys.executable,list(map(str,backend.jax.devices())),
                     os.environ.get('CAUSTIC_THREADS','system default'),backend.jax.config.jax_enable_x64)
        logging.info(f'Physics: flat entrance z=0; screen z={settings["distance_mm"]:g} mm; n={settings["refractive_index"]:g}/{settings["air_index"]:g}; mask L={settings["L_mm"]:g} mm')
        with log_stage('[1] Square input validation and convex mask'):
            raw=load(settings['image'])
            if raw.shape[0]!=raw.shape[1] or raw.shape[0]<2:raise ValueError(f'Target must be a square PNG of at least 2x2, got {raw.shape}; no implicit resize/padding')
            gray,shift=center_pattern(raw) if settings['center_target'] else (raw,(0,0))
            domain=MaskDomain(settings['mask'],settings['L_mm'],settings['mask_threshold'])
            config=optical_config(settings,gray.shape[1])
            input_info=dict(image=settings['image'],mask=settings['mask'],shape=list(gray.shape),pixel_shift=shift,
                            image_sha256=hashlib.sha256(Path(settings['image']).read_bytes()).hexdigest(),
                            mask_sha256=hashlib.sha256(Path(settings['mask']).read_bytes()).hexdigest(),
                            illumination='uniform parallel +z light restricted to the convex lens aperture')
            logging.info(f'Input {gray.shape}; center shift {shift}; mask area {domain.area:.6f} mm^2; grid spacing L/n={settings["L_mm"]/settings["mesh_n"]:.6f} mm')
        with log_stage('[2] SDOT source capacities and target power diagram'):
            mapping,ot=solve(gray,config,out,source_domain=domain)
            logging.info(f"OT mass L1 residual={ot['capacity_L1']:.3e}; source mass={ot['source_mass']:.15g}; target mass={ot['target_mass']:.15g}")
        base=out/'base';base.mkdir()
        def factory(c,n):
            with log_stage(f'Mesh n={n}, spacing={settings["L_mm"]/n:.6f} mm'):
                mesh=domain.mesh(c,n)
                logging.info(f'Mesh: {len(mesh[0]):,} vertices / {len(mesh[1]):,} triangles')
                return mesh
        with log_stage('[3] Multi-resolution JAX/Optax surface design'):
            xy,f,b,z,history=optimize(mapping,gray,config,base,mesh_factory=factory)
        def evaluate(folder,c,state,hist,label,refinement=None):
            xx,ff,bb,zz=state
            with log_stage(f'{label}: CNC solid export'):
                mesh=export(xx,ff,bb,zz,c,folder);mesh['body_count']=1
            with log_stage(f'{label}: independent forward raytrace ({2**c.trace_power:,} rays)'):
                metrics,I,T=trace(xx,ff,zz,gray,c,folder,source_domain=domain)
            with log_stage(f'{label}: save four diagnostic figures'):
                plots(xx,ff,zz,mapping,hist,I,T,folder,c)
            row=dict(stage=label,pipeline_complete=True,config=asdict(c),metrics=metrics,mesh=mesh,
                     no_total_internal_reflection=no_total_internal_reflection(metrics),cnc_release=False,model='lens.stl')
            if refinement is not None:row['refinement']=refinement
            write(folder/'result.json',row)
            candidates.append(row)
            logging.info(f"{label}: leak={metrics['dark_leak_captured']:.3%}; RMSE={metrics['rmse_relative']:.5f}; thin flux={metrics['thin_flux_ratio']}; TIR-free={row['no_total_internal_reflection']}")
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
            record=None
            with log_stage(f'[4.{index}] Local mesh refinement and physical image optimization'):
                rounds=int(stage.get('adaptive',0))
                if rounds:
                    xy,f,b,z,_,record=refine_mesh(xy,f,z,b,gray,c,rounds)
                    logging.info(f'After local refinement: {len(xy):,} vertices / {len(f):,} triangles')
                from scipy.spatial import Delaunay
                xy,f,b,z,history=optimize(mapping,gray,c,folder,initial=(xy,z,f),mesh_override=(xy,f,b,Delaunay(xy)))
            evaluate(folder,c,(xy,f,b,z),history,name,record)
        selected=select_best(candidates)
        if selected is not None:
            for name in ['lens.stl','height.png','comparison.png','loss.png','mapping.png']:
                shutil.copyfile(out/selected['stage']/name,out/name)
        result=dict(pipeline_complete=True,selected_stage=selected['stage'] if selected else None,
                    selection='lowest independent raytrace relative RMSE among TIR-free stages',
                    selection_status='selected' if selected else 'no_tir_free_candidate',
                    no_total_internal_reflection=selected is not None,
                    metrics=selected['metrics'] if selected else None,mesh=selected['mesh'] if selected else None,
                    config=selected['config'] if selected else None,settings=settings,input=input_info,domain=domain.metadata(),
                    transport=ot,environment=backend.environment(),candidates=candidates,
                    cnc_release=False,elapsed_seconds=time.monotonic()-started,model='lens.stl' if selected else None)
        write(out/'result.json',result)
        if selected:
            logging.info(f'[5] FINISHED. Selected stage: {selected["stage"]}; RMSE={selected["metrics"]["rmse_relative"]:.8g}; model: {out/"lens.stl"}')
        else:
            logging.warning('[5] FINISHED. No TIR-free candidate; no root model selected. Inspect stage result.json files.')
        return result
    except BaseException:
        write(out/'result.json',dict(pipeline_complete=False,error=traceback.format_exc(),settings=settings,
                                    candidates=candidates,elapsed_seconds=time.monotonic()-started))
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
    for key in ['image','mask','out','refractive_index','air_index','L_mm','distance_mm','target_size_mm','mesh_n']:
        if getattr(args,key) is not None:settings[key]=getattr(args,key)
    if args.threads is not None:settings['runtime']['threads']=args.threads
    if args.no_center:settings['center_target']=False
    if not all(settings[key] for key in ['image','mask','out']):p.error('--image, --mask and --out are required, directly or in JSON')
    validate_settings(settings)

    configure_runtime(settings.get('runtime',{}))
    run(settings)

if __name__=='__main__':
    main()
