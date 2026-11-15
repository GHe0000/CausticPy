"""Re-evaluate a saved actual triangle surface without SDOT."""
from caustic import backend
from pathlib import Path
import argparse,json
import numpy as np
from caustic.config import Config
from caustic.target import load
from caustic.verify import trace
from run import dump
def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--power',type=int,default=24);p.add_argument('--level',type=int);p.add_argument('--out',required=True);p.add_argument('--target');a=p.parse_args()
    root=Path(__file__).resolve().parent
    output=Path(a.out).resolve()
    if not output.is_relative_to(root):raise ValueError('Output must be in JAX3')
    output.mkdir(parents=True,exist_ok=False)
    saved=Path(a.run);c=Config.load(saved/'config.json');c.trace_power=a.power
    data=np.load(saved/f'surface_{a.level or c.mesh_levels[-1]}.npz')
    target=a.target or json.loads((saved/'input.json').read_text(encoding='utf-8'))['path']
    stats,_,_=trace(data['xy'],data['faces'],data['z'],load(target,center=c.center_target),c,output)
    dump(output/'metrics.json',stats);print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
