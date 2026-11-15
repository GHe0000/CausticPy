"""Resume physical image optimization on an existing 600 grid; never alter reference inputs."""
from caustic import backend
from pathlib import Path
import argparse,time,shutil
import numpy as np
from caustic.config import Config
from caustic.target import load
from caustic.transport import Correspondence
from caustic.optimize import optimize
from caustic.verify import trace
from caustic.export import export
from caustic.report import plots
from run import dump
def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);p.add_argument('--resolution',type=int,default=1280);p.add_argument('--iterations',type=int,default=200);p.add_argument('--thin-weight',type=float,default=1.);p.add_argument('--adaptive',type=int,default=0);p.add_argument('--mapping-anchor',type=float);p.add_argument('--fine-samples',type=int)
    a=p.parse_args();source=Path(a.run);out=Path(a.out).resolve();root=Path(__file__).resolve().parent
    if not out.is_relative_to(root):raise ValueError('Output must be inside JAX3')
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    try:
        import hashlib
        c=Config.load(source/'config.json');last=c.mesh_levels[-1];
        dump(out/'parent.json',dict(source=str(source.resolve()),surface_sha256=hashlib.sha256((source/f'surface_{last}.npz').read_bytes()).hexdigest(),arguments=vars(a)));c.mesh_levels=[last];c.fit_iterations=0;c.image_iterations=a.iterations;c.image_resolution=a.resolution;c.thin_weight=a.thin_weight
        if a.mapping_anchor is not None:c.mapping_anchor=a.mapping_anchor
        if a.fine_samples is not None:c.fine_quadrature=a.fine_samples
        c.validate()
        c.save(out/'config.json');dump(out/'environment.json',backend.environment())
        data=np.load(source/f'surface_{last}.npz');ot=np.load(source/'transport.npz');mapping=Correspondence(ot['source'],ot['target'])
        for file in ['transport.npz','transport.json','input.json']:shutil.copy2(source/file,out/file)
        import json,hashlib
        provenance=json.loads((source/'input.json').read_text(encoding='utf-8'))
        target_path=Path(provenance['path'])
        if hashlib.sha256(target_path.read_bytes()).hexdigest()!=provenance['sha256']:raise ValueError('Target has changed since original run')
        gray=load(target_path,center=c.center_target)
        from scipy.spatial import Delaunay
        initial=(data['xy'],data['z']);override=(data['xy'],data['faces'],data['boundary'],Delaunay(data['xy']))
        if a.adaptive:
            from caustic.adaptive import refine_mesh
            xx,ff,bb,zz,tt,record=refine_mesh(data['xy'],data['faces'],data['z'],data['boundary'],gray,c,a.adaptive)
            initial=(xx,zz);override=(xx,ff,bb,tt);dump(out/'adaptive.json',record)
        xy,f,b,z,history=optimize(mapping,gray,c,out,initial=initial,mesh_override=override)
        dump(out/'mesh.json',export(xy,f,b,z,c,out))
        metrics,I,T=trace(xy,f,z,gray,c,out);dump(out/'metrics.json',metrics)
        plots(xy,f,z,mapping,history,I,T,out,c)
        basic=metrics['dark_leak_captured']<.02 and metrics['rmse_relative']<.35 and metrics['tir']==0 and metrics['captured_flux']>.98
        fine=metrics['thin_flux_ratio'] is None or .8<=metrics['thin_flux_ratio']<=1.2
        passed=basic and fine
        dump(out/'status.json',dict(pipeline_complete=True,optical_quality_pass=passed,basic_imaging_pass=basic,thin_detail_pass=fine,cnc_release=False,seconds=time.time()-start))
        print(metrics,flush=True)
    except Exception:
        import traceback
        dump(out/'status.json',dict(pipeline_complete=False,error=traceback.format_exc()));raise
if __name__=='__main__':main()
