"""Runtime environment configuration."""
from pathlib import Path
import os,shutil

_DLL_HANDLES=[]
_KEYS={'jax_platforms','jax_enable_x64','threads','compiler','path_dirs','dll_dirs','xla_flags','cpu_devices'}

def validate_runtime(options):
    if not isinstance(options,dict):raise ValueError('runtime must be an object')
    unknown=set(options)-_KEYS
    if unknown:raise ValueError('Unknown runtime keys: '+', '.join(sorted(unknown)))
    if 'threads' in options and (type(options['threads']) is not int or options['threads']<0):
        raise ValueError('runtime.threads must be an integer >=0; omit for library defaults')
    if 'jax_enable_x64' in options and type(options['jax_enable_x64']) is not bool:
        raise ValueError('runtime.jax_enable_x64 must be true or false')
    if 'cpu_devices' in options and (type(options['cpu_devices']) is not int or options['cpu_devices'] < 1):
        raise ValueError('runtime.cpu_devices must be a positive integer')
    for key,value in options.items():
        if key in ['threads','jax_enable_x64','cpu_devices']:continue
        if key in ['path_dirs','dll_dirs']:
            if not isinstance(value,list) or any(not isinstance(v,str) or not v for v in value):
                raise ValueError('runtime.'+key+' must be a list of nonempty paths')
        elif not isinstance(value,str) or not value:
            raise ValueError('runtime.'+key+' must be a nonempty string')
    if '{threads}' in options.get('xla_flags','') and 'threads' not in options:
        raise ValueError('runtime.xla_flags uses {threads}, but runtime.threads is omitted')

def resolve_runtime(options,base):
    if not isinstance(options,dict):raise ValueError('runtime must be an object')
    values=dict(options)
    values.setdefault('jax_enable_x64',True)
    validate_runtime(values)
    def path(value):
        p=Path(os.path.expandvars(value)).expanduser()
        return str((base/p).resolve() if not p.is_absolute() else p.resolve())
    for key in ['path_dirs','dll_dirs']:
        if key in values:values[key]=[path(value) for value in values[key]]
    if 'compiler' in values:
        value=os.path.expandvars(values['compiler'])
        values['compiler']=path(value) if '/' in value or '\\' in value or value.startswith('~') else value
    return values

def configure_runtime(options):
    validate_runtime(options)
    add_dll_directory=getattr(os,'add_dll_directory',None)
    if options.get('dll_dirs') and not callable(add_dll_directory):
        raise ValueError('runtime.dll_dirs is configured, but this Python has no DLL-directory registration API; omit dll_dirs and configure the native library loader externally')
    if 'path_dirs' in options:
        for folder in options['path_dirs']:
            if not Path(folder).is_dir():raise ValueError('runtime.path_dirs directory not found: '+folder)
        if options['path_dirs']:
            os.environ['PATH']=os.pathsep.join(options['path_dirs']+[os.environ.get('PATH','')])
    if 'compiler' in options:
        compiler=shutil.which(options['compiler'])
        if compiler is None:raise ValueError('runtime.compiler executable not found: '+options['compiler'])
        os.environ['CXX']=compiler
    if 'jax_platforms' in options:os.environ['JAX_PLATFORMS']=options['jax_platforms']
    if 'cpu_devices' in options:os.environ['JAX_NUM_CPU_DEVICES']=str(options['cpu_devices'])
    os.environ['JAX_ENABLE_X64']=str(options.get('jax_enable_x64',True)).lower()
    count=None
    if 'threads' in options:
        count=options['threads'] or (os.process_cpu_count() or 1)
        for key in ['CAUSTIC_THREADS','LOOM_NB_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
            os.environ[key]=str(count)
    if 'xla_flags' in options:
        os.environ['XLA_FLAGS']=options['xla_flags'].replace('{threads}',str(count))
    if options.get('dll_dirs'):
        for folder in options.get('dll_dirs',[]):
            path=Path(folder)
            if not path.is_dir():raise ValueError('runtime.dll_dirs directory not found: '+folder)
            _DLL_HANDLES.append(add_dll_directory(str(path)))


def collect_environment():
    """Return runtime environment metadata."""
    import importlib.metadata as md, sys
    import jax
    return dict(python=sys.executable, version=sys.version, devices=list(map(str, jax.devices())),
                float64=bool(jax.config.jax_enable_x64), logical_cpus=os.cpu_count(),
                compiler=os.environ.get('CXX'),
                cache={k: os.environ.get(k) for k in ['LOOM_CACHE_DIR', 'LOOM_BUILD_DIR', 'MPLCONFIGDIR', 'JAX_COMPILATION_CACHE_DIR']},
                threads={k: os.environ.get(k) for k in ['CAUSTIC_THREADS', 'LOOM_NB_THREADS', 'SDOT_NB_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS']},
                packages={p: md.version(p) for p in ['numpy', 'scipy', 'jax', 'jaxlib', 'optax', 'trimesh', 'Pillow', 'scikit-image', 'matplotlib', 'sdot']})
