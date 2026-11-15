from pathlib import Path
import json,numpy as np,trimesh
root=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser();parser.add_argument('folder',nargs='?');args=parser.parse_args()
folder=Path(args.folder) if args.folder else Path(json.loads((root/'LATEST_RESULT.json').read_text(encoding='utf-8'))['result'])
data=np.load(folder/'surface_600.npz')
mesh=trimesh.load_mesh(folder/'lens_precision.stl')
expected=np.vstack([np.c_[data['xy'],data['z']],np.c_[data['xy'],np.zeros(len(data['xy']))]])
sort=lambda a:a[np.lexsort((a[:,2],a[:,1],a[:,0]))]
assert len(mesh.vertices)==len(expected)
delta=float(np.max(abs(sort(mesh.vertices)-sort(expected))))
assert delta==0 and mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
report=dict(file=str(folder/'lens_precision.stl'),vertices=len(mesh.vertices),faces=len(mesh.faces),watertight=bool(mesh.is_watertight),
            winding_consistent=bool(mesh.is_winding_consistent),max_float64_vertex_difference_mm=delta,volume_mm3=float(mesh.volume))
(root/'verification/precision_roundtrip.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
