"""Convert a saved design to an exactly pixel-centered target and physically refit it."""
from caustic import backend
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from scipy.spatial import Delaunay
from caustic.config import Config
from caustic.target import load,center_pattern,dimensions
from caustic.transport import solve
from caustic.optimize import optimize
from caustic.export import export
from caustic.verify import trace
from caustic.report import plots
from run import dump
def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',required=True);p.add_argument('--polynomial',action='store_true');a=p.parse_args()
    source=Path(a.source);root=Path(__file__).resolve().parent;out=Path(a.out).resolve()
    if not out.is_relative_to(root):raise ValueError('Output must be inside JAX3')
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    try:
        c=Config.load(source/'config.json')
        if c.center_target:raise ValueError('Source already uses the centered target')
        info=json.loads((source/'input.json').read_text(encoding='utf-8'));path=Path(info['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Input changed')
        raw=load(path);gray,shift=center_pattern(raw)
        w,h=dimensions(gray,c);translation=np.array([shift[0]*w/gray.shape[1],-shift[1]*h/gray.shape[0]])
        c.center_target=True;c.mesh_levels=[600];c.fit_iterations=180;c.image_iterations=360;c.image_resolution=gray.shape[1]
        c.ot_resolution=gray.shape[1];c.thin_weight=1000.;c.mapping_anchor=.001
        c.save(out/'config.json');dump(out/'environment.json',backend.environment())
        info.update(center_target=True,pixel_shift=shift,target_measure_translation_mm=translation)
        dump(out/'input.json',info);dump(out/'parent.json',dict(source=str(source.resolve()),surface_sha256=hashlib.sha256((source/'surface_600.npz').read_bytes()).hexdigest()))
        old=np.load(source/'transport.npz')
        warm=old['weights']-2*(old['seeds']@translation);warm-=warm[0]
        try:mapping,stats=solve(gray,c,out,warm=warm)
        except Exception as error:
            # Same original zero-density objective; rebuild an admissible start if raster changes invalidate the analytic warm start.
            dump(out/'warm_failure.json',dict(error=repr(error)))
            mapping,stats=solve(gray,c,out)
        dump(out/'transport.json',stats)
        data=np.load(source/'surface_600.npz');xy=data['xy'];z=data['z']
        slope=translation/((c.screen_z_mm-c.base_mm)*(c.n_lens/c.n_air-1))
        initial=z+xy@slope
        if a.polynomial:
            from caustic.translation import translated_surface
            initial,diagnostic=translated_surface(xy,data['faces'],z,translation,c,out)
            dump(out/'translation.json',diagnostic)
            c.fit_iterations=0;c.image_iterations=400;c.save(out/'config.json')
        dump(out/'centering.json',dict(pixel_shift=shift,translation_mm=translation,initial_tilt=slope,
             clipped_vertices=int(np.sum((initial<c.min_thickness_mm)|(initial>c.max_thickness_mm)))))
        initial=np.clip(initial,c.min_thickness_mm,c.max_thickness_mm)
        xy,f,b,z,history=optimize(mapping,gray,c,out,initial=(xy,initial),mesh_override=(xy,data['faces'],data['boundary'],Delaunay(xy)))
        dump(out/'mesh.json',export(xy,f,b,z,c,out))
        metrics,I,T=trace(xy,f,z,gray,c,out);dump(out/'metrics.json',metrics);plots(xy,f,z,mapping,history,I,T,out,c)
        basic=metrics['dark_leak_captured']<.02 and metrics['rmse_relative']<.35 and metrics['tir']==0 and metrics['captured_flux']>.98
        fine=metrics['thin_flux_ratio'] is None or .8<=metrics['thin_flux_ratio']<=1.2
        dump(out/'status.json',dict(pipeline_complete=True,center_target=True,basic_imaging_pass=basic,thin_detail_pass=fine,optical_quality_pass=basic and fine,cnc_release=False,seconds=time.time()-start))
        print(metrics,flush=True)
    except Exception:
        import traceback
        dump(out/'status.json',dict(pipeline_complete=False,error=traceback.format_exc()));raise
if __name__=='__main__':main()
