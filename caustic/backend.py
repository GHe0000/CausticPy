"""JAX access and environment reporting; no implicit runtime overrides."""
import os
import jax
import jax.numpy as jnp

def environment():
    import sys,importlib.metadata as md
    return dict(python=sys.executable,version=sys.version,devices=list(map(str,jax.devices())),
                float64=bool(jax.config.jax_enable_x64),logical_cpus=os.cpu_count(),
                compiler=os.environ.get('CXX'),
                cache={k:os.environ.get(k) for k in ['LOOM_CACHE_DIR','LOOM_BUILD_DIR','MPLCONFIGDIR','JAX_COMPILATION_CACHE_DIR']},
                threads={k:os.environ.get(k) for k in ['CAUSTIC_THREADS','LOOM_NB_THREADS','SDOT_NB_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','XLA_FLAGS']},
                packages={p:md.version(p) for p in ['numpy','scipy','jax','jaxlib','optax','trimesh','Pillow','scikit-image','matplotlib','sdot']})
