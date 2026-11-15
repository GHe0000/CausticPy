"""Integration regression: CLI continuation must retain an irregular saved mesh."""
from pathlib import Path
import json,shutil,subprocess,sys
import numpy as np
from scipy.spatial import Delaunay
root=Path(__file__).resolve().parents[1]
import tempfile
(root/'verification').mkdir(exist_ok=True)
workspace=Path(tempfile.mkdtemp(prefix='resume_',dir=root/'verification'))
source=workspace/'source';out=workspace/'run'
source.mkdir(parents=True,exist_ok=False)
base=root/'results/smoke_v7'
data=np.load(base/'surface_48.npz')
xy=data['xy'];f=data['faces'];z=data['z']
chosen=np.linspace(0,len(f)-1,30).astype(int)
extra=xy[f[chosen]].mean(axis=1);ez=z[f[chosen]].mean(axis=1)
xx=np.vstack([xy,extra]);zz=np.r_[z,ez];ff=Delaunay(xx).simplices
np.savez_compressed(source/'surface_48.npz',xy=xx,faces=ff,boundary=data['boundary'],z=zz)
config=json.loads((base/'config.json').read_text());config.update(mesh_levels=[48],trace_power=12)
(source/'config.json').write_text(json.dumps(config),encoding='utf-8')
for name in ['transport.npz','transport.json','input.json']:shutil.copy2(base/name,source/name)
subprocess.run([sys.executable,str(root/'refine.py'),str(source),'--out',str(out),'--resolution','64','--iterations','1'],cwd=root,check=True)
saved=np.load(out/'surface_48.npz')
np.testing.assert_array_equal(saved['xy'],xx);np.testing.assert_array_equal(saved['faces'],ff)
assert len(saved['xy'])>len(xy)
report=dict(pass_resume=True,base_vertices=len(xy),saved_irregular_vertices=len(xx),resumed_vertices=len(saved['xy']),identical_xy=True,identical_faces=True)
(root/'verification/resume.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
