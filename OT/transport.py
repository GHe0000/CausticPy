"""Paper-style reverse SDOT: source Voronoi capacities, target Laguerre barycenters."""
import numpy as np
import logging
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
from . import backend
import sdot
def solve(gray, c, source_domain):
    from .target import resize, dimensions
    seeds = source_domain.seeds(c.ot_sites, c.seed)
    source = source_domain.sdot_distribution()
    pd=sdot.PowerDiagram(seeds,distribution=source,kernel_dtype='FP64')
    masses=np.asarray(pd.measures);masses=masses/masses.sum()
    src=np.c_[np.asarray(pd.moment(0)),np.asarray(pd.moment(1))]/masses[:,None]
    reduced=resize(gray,c.ot_resolution)
    w,h=dimensions(gray,c)
    # SDOT axis 0 is x; PNG row 0 is +y. Flip and transpose explicitly.
    logging.info('SDOT solve: %s sites, raster %s',len(seeds),reduced.shape)
    # Explicit density homotopy supplies an admissible start for a disconnected target.
    # Final stage is EXACTLY the original raster, with zero dark density.
    weights0=None;stages=[]
    mean=reduced.mean()
    for alpha in c.solver["sdot"]["density_stages"]:
        logging.info("SDOT solving density stage alpha=%g, sites=%s",alpha,len(seeds))
        values=alpha*reduced+(1-alpha)*mean
        density=sdot.Image(np.ascontiguousarray(values[::-1].T),origin=np.array([-w/2,-h/2]),frame=np.diag([w,h]))
        sol=sdot.ot_solve(sdot.SumOfDiracs(seeds,weights=masses),density,tol=c.solver["sdot"]["tol"],max_iter=c.solver["sdot"]["max_iter"],
                          weights0=weights0,continuation='never')
        weights0=np.asarray(sol.weights)
        stages.append(dict(alpha=alpha,**dict(sol.stats)))
        logging.info('SDOT alpha %g residual %.3e',alpha,sol.stats['residual'])
    dst=np.asarray(sol.barycenters)
    actual=np.asarray(sol.cell_masses)
    stats=dict(sol.stats);stats['density_continuation']=stages
    stats.update(capacity_L1=float(abs(actual-masses).sum()),capacity_max_relative=float(np.max(abs(actual-masses)/masses)),
                 source_mass=float(masses.sum()),target_mass=float(actual.sum()),black_floor=0.,
                 source_polygon_area_ratio=None,
                 source_mask_area_mm2=source_domain.area,
                 mapping='reverse SDOT barycentric correspondence; piecewise linear interpolation')
    if not np.isfinite(dst).all(): raise RuntimeError('Invalid transport barycenters')
    return Correspondence(src,dst),stats
class Correspondence:
    def __init__(self,src,dst):
        self.src=src;self.dst=dst
        self.linear=LinearNDInterpolator(src,dst)
        self.nearest=NearestNDInterpolator(src,dst)
    def __call__(self,xy):
        y=self.linear(xy);bad=~np.isfinite(y).all(axis=1)
        y[bad]=self.nearest(xy[bad])
        return y
