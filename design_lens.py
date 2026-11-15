"""Generic CLI. No JAX/NumPy imports occur until runtime options are configured."""
from pathlib import Path
import argparse
from caustic.settings import RunSettings
def main():
    p=argparse.ArgumentParser(description='Square PNG + convex square mask -> SDOT -> JAX -> independent raytrace -> CNC STL',
                              formatter_class=argparse.RawDescriptionHelpFormatter,
                              epilog="Defaults: n=1.5, L=60 mm, distance=100 mm, grid=600, profile=high, threads=all. CLI values override JSON.")
    p.add_argument('--config',help='Unified JSON config (relative paths resolve relative to this file)')
    p.add_argument('--image','--target',dest='image')
    p.add_argument('--mask')
    p.add_argument('--out')
    p.add_argument('--n','--refractive-index',dest='refractive_index',type=float)
    p.add_argument('--n-air',dest='air_index',type=float)
    p.add_argument('--L','--length',dest='L_mm',type=float,help='Physical side length of the whole mask square [mm]')
    p.add_argument('--distance','--focal-length',dest='distance_mm',type=float,help='Screen z from flat entrance z=0 [mm]')
    p.add_argument('--target-size',dest='target_size_mm',type=float)
    p.add_argument('--grid','--mesh-n',dest='mesh_n',type=int,help='Grid intervals per L; mesh spacing=L/grid')
    p.add_argument('--threads',type=int)
    p.add_argument('--profile',choices=['quick','balanced','high'])
    p.add_argument('--no-center',action='store_true')
    a=p.parse_args();settings=RunSettings.load(a.config) if a.config else RunSettings()
    for key in ['image','mask','out','refractive_index','air_index','L_mm','distance_mm','target_size_mm','mesh_n','threads','profile']:
        value=getattr(a,key)
        if value is not None:setattr(settings,key,value)
    if a.no_center:settings.center_target=False
    if not settings.image or not settings.mask or not settings.out:p.error('--image, --mask and --out are required, directly or in JSON')
    settings.validate()
    from caustic.settings import configure_runtime
    configure_runtime(settings.threads)
    from caustic.general_pipeline import run
    run(settings)
if __name__=='__main__':main()
