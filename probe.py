import os,sys,inspect,json
from pathlib import Path
import numpy as np,sdot
print(sys.executable,flush=True)
rng=np.random.default_rng(0)
p=rng.uniform(-.6,.6,(16,2))
a=np.arange(64)*2*np.pi/64
domain=sdot.Polytope(np.c_[np.cos(a),np.sin(a)],np.ones(64))
print('solving polygon',flush=True)
sol=sdot.ot_solve(sdot.SumOfDiracs(p),domain,tol=1e-9,max_iter=100)
print('polygon',sol.stats, np.asarray(sol.barycenters).shape,flush=True)
im=sdot.Image(np.ones((8,8)),origin=np.array([-1.,-1.]),frame=np.eye(2)*2)
print('image solve',flush=True)
sol=sdot.ot_solve(sdot.SumOfDiracs(p),im,tol=1e-9,max_iter=100)
print('image',sol.stats,np.asarray(sol.barycenters),flush=True)
print('PD methods', [x for x in dir(sol.power_diagram()) if any(y in x for y in ['mass','integr','measure','cell'])],flush=True)
