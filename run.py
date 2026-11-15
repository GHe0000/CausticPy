from caustic import backend
from pathlib import Path
import argparse,json,traceback,time,hashlib
import numpy as np
from caustic.config import Config
from caustic.target import load,center_pattern
from caustic.transport import solve
from caustic.optimize import optimize
from caustic.export import export
from caustic.verify import trace
from caustic.report import plots
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',default='target_saturn.png');ap.add_argument('--config');ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out).resolve()
    root=Path(__file__).resolve().parent
    if not out.is_relative_to(root): raise ValueError('Output must be inside JAX3')
    out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:
        c=Config.load(args.config);c.save(out/'config.json')
        dump(out/'environment.json',backend.environment())
        raw=load(args.target)
        gray,shift=center_pattern(raw) if c.center_target else (raw,(0,0))
        dump(out/'input.json',dict(path=str(Path(args.target).resolve()),sha256=hashlib.sha256(Path(args.target).read_bytes()).hexdigest(),shape=gray.shape,center_target=c.center_target,pixel_shift=shift,
             reference='screen z=100 mm from flat entrance z=0; propagation +z',image_model='relative linear grayscale irradiance, zero black density'))
        mapping,ot=solve(gray,c,out);dump(out/'transport.json',ot)
        xy,f,boundary,z,history=optimize(mapping,gray,c,out)
        mesh=export(xy,f,boundary,z,c,out);dump(out/'mesh.json',mesh)
        metrics,I,T=trace(xy,f,z,gray,c,out);dump(out/'metrics.json',metrics)
        plots(xy,f,z,mapping,history,I,T,out,c)
        # Honest quality gate: a running pipeline or a watertight mesh does not establish optical quality.
        basic=metrics['dark_leak_captured']<.02 and metrics['rmse_relative']<.35 and metrics['tir']==0 and metrics['captured_flux']>.98
        fine=metrics['thin_flux_ratio'] is None or .8<=metrics['thin_flux_ratio']<=1.2
        passed=basic and fine
        dump(out/'status.json',dict(pipeline_complete=True,optical_quality_pass=passed,basic_imaging_pass=basic,thin_detail_pass=fine,cnc_release=False,seconds=time.time()-started,
             reason='Tool radius, scallop, polishing and measured material dispersion require machining-specific validation.'))
        print(json.dumps(metrics,indent=2),flush=True)
    except Exception:
        dump(out/'status.json',dict(pipeline_complete=False,error=traceback.format_exc(),seconds=time.time()-started));raise
if __name__=='__main__':main()
