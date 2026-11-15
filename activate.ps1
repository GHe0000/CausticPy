$env:CONDA_PKGS_DIRS = "$PSScriptRoot\.cache\mamba"
$env:MAMBA_ROOT_PREFIX = 'C:\Users\GHe\.local\share\mamba'
if ($env:CONDA_PREFIX -ne 'C:\Users\GHe\.local\share\mamba\envs\caustic-jax') {
    if (-not (Test-Path Function:\mamba)) {
        (& 'D:\Software\miniforge3\Library\bin\mamba.exe' shell hook -s powershell) | Out-String | Invoke-Expression
    }
    mamba activate caustic-jax
}
$env:LOOM_CACHE_DIR = "$PSScriptRoot\.cache\loom"
$env:LOOM_BUILD_DIR = "$PSScriptRoot\.cache\build"
$env:MPLCONFIGDIR = "$PSScriptRoot\.cache\matplotlib"
$env:JAX_ENABLE_X64 = 'true'
$env:JAX_PLATFORMS = 'cpu'

$env:CXX = 'D:\Software\msys2\ucrt64\bin\g++.exe'
$env:PATH = 'C:\Users\GHe\.local\share\mamba\envs\caustic-jax;D:\Software\msys2\ucrt64\bin;' + $env:PATH
$cores = [Environment]::ProcessorCount
$env:LOOM_NB_THREADS = "$cores"
$env:OMP_NUM_THREADS = "$cores"
$env:OPENBLAS_NUM_THREADS = "$cores"
$env:MKL_NUM_THREADS = "$cores"
$env:XLA_FLAGS = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$cores"

$env:PYTHONUTF8 = '1'
