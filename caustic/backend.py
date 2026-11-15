import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for key,sub in [('LOOM_CACHE_DIR','loom'),('LOOM_BUILD_DIR','build'),('MPLCONFIGDIR','matplotlib')]:
    os.environ.setdefault(key,str(ROOT/'.cache'/sub))
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ['JAX_ENABLE_X64']='true'
# Python 3.8+ deliberately ignores PATH for extension DLL dependencies.
_dll_handles=[]
if os.name=='nt':
    for folder in [Path(r'D:\Software\msys2\ucrt64\bin'),ROOT/'.cache'/'build'/'runtime',ROOT/'.cache'/'build'/'units']:
        folder.mkdir(parents=True,exist_ok=True) if folder.is_relative_to(ROOT) else None
        if folder.exists(): _dll_handles.append(os.add_dll_directory(str(folder)))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
def environment():
    import sys,importlib.metadata as md
    return dict(python=sys.executable,version=sys.version,devices=list(map(str,jax.devices())),
                float64=bool(jax.config.jax_enable_x64),logical_cpus=os.cpu_count(),
                threads={k:os.environ.get(k) for k in ['LOOM_NB_THREADS','SDOT_NB_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS']},
                packages={p:md.version(p) for p in ['numpy','scipy','jax','jaxlib','optax','trimesh','Pillow','scikit-image','matplotlib','sdot']})
