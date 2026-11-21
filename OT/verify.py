"""Independent Sobol facet ray tracer."""
import numpy as np
import logging
from scipy.spatial import Delaunay
from scipy.stats import qmc
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from common.projection import refract, screen_hits, screen_pixels
from common.target import target_dimensions
def trace(xy,f,z,gray,c,source_domain):
    w,h=target_dimensions(gray,c.target_size_mm);ny,nx=gray.shape
    tri=Delaunay(xy)
    lookup={tuple(sorted(face)):i for i,face in enumerate(f)}
    ids=np.array([lookup.get(tuple(sorted(face)),-1) for face in tri.simplices])
    v=np.c_[xy,z][f];normal=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    d,q,trans=refract(normal,c.refractive_index,c.air_index)
    counts=np.zeros((ny,nx));physical=np.zeros_like(counts)
    total=2**c.trace_power;engine=qmc.Sobol(3,scramble=True,seed=c.seed+1)
    stats=dict(incident_rays=total,aperture_miss=0,tir=0,backward=0,outside_screen=0,captured=0)
    # Warm SciPy point-location cache before multithreaded reads.
    tri.find_simplex(xy[:1])
    def batch(u):
        p=source_domain.sample(u)
        s=tri.find_simplex(p);valid=(s>=0)&(ids[np.maximum(s,0)]>=0)
        st=dict(aperture_miss=int((~valid).sum()))
        p=p[valid];s=s[valid];fi=ids[s]
        coeff=tri.transform[s,:2];b=np.einsum('nij,nj->ni',coeff,p-tri.transform[s,2])
        bary=np.c_[b,1-b.sum(axis=1)]
        hitz=(z[tri.simplices[s]]*bary).sum(axis=1)
        ok=(q[fi]>0)&(d[fi,2]>0)
        st['tir']=int((q[fi]<=0).sum());st['backward']=int(((q[fi]>0)&(d[fi,2]<=0)).sum())
        hits=screen_hits(p,hitz,d[fi],c.distance_mm,min_dz=1e-12)
        iy,ix=screen_pixels(hits,w,h,gray.shape)
        inside=ok&(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)
        st['outside_screen']=int((ok&~inside).sum());st['captured']=int(inside.sum())
        pixel=iy[inside]*nx+ix[inside]
        image=np.bincount(pixel,minlength=nx*ny).reshape(ny,nx)/total
        entrance=1-((c.refractive_index-c.air_index)/(c.refractive_index+c.air_index))**2
        fresnel=np.bincount(pixel,weights=entrance*trans[fi[inside]],minlength=nx*ny).reshape(ny,nx)/total
        return st,image,fresnel
    import os
    from concurrent.futures import ThreadPoolExecutor
    def samples():
        for start in range(0,total,2**18):
            yield engine.random(min(2**18,total-start))
    requested=os.environ.get('CAUSTIC_THREADS')
    workers=min(int(requested),max(1,total//(2**18))) if requested else None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        workers=min(pool._max_workers,max(1,total//(2**18)))
        for batch_index,(st,image,fresnel) in enumerate(pool.map(batch,samples(),buffersize=workers),start=1):
            for k,vv in st.items():stats[k]+=vv
            counts+=image;physical+=fresnel
            logging.info('Raytrace %s/%s rays (%.1f%%)',f'{min(batch_index*2**18,total):,}',f'{total:,}',100*min(batch_index*2**18,total)/total)
    stats['raytrace_workers']=workers
    target=gray/gray.sum();dark=gray==0
    gradient_error=float(sum(np.sum(abs(np.diff(counts,axis=ax)-np.diff(target,axis=ax))) for ax in [0,1])/
                         max(sum(np.sum(abs(np.diff(target,axis=ax))) for ax in [0,1]),1e-20))
    edge_mask=(abs(np.gradient(gray)[0])+abs(np.gradient(gray)[1]))>.05

    bright=gray>.5*gray.max()
    thin=skeletonize(gray>.05*gray.max())&(distance_transform_edt(gray>.05*gray.max())<2)
    expected=target[thin].sum()
    stats.update(captured_flux=float(counts.sum()),physical_transmitted_screen_flux=float(physical.sum()),
                 dark_leak_incident=float(counts[dark].sum()),dark_leak_captured=float(counts[dark].sum()/max(counts.sum(),1e-20)),
                 target_l1=float(abs(counts-target).sum()),edge_gradient_relative_l1=gradient_error,
                 edge_band_flux=float(counts[edge_mask].sum()),rmse_relative=float(np.sqrt(np.mean((counts-target)**2))/np.sqrt(np.mean(target**2))),
                 thin_pixels=int(thin.sum()),thin_recall_at_quarter_target=float(np.mean(counts[thin]>=.25*target[thin])) if thin.any() else None,thin_flux_ratio=float(counts[thin].sum()/expected) if expected else None,
                 bright_mean_dark_mean_ratio=float(counts[bright].mean()/max(counts[dark].mean(),1e-20)) if dark.any() else None,
                 min_snell_discriminant=float(q.min()),max_surface_slope=float(np.max(np.linalg.norm(normal[:,:2],axis=1)/normal[:,2])),
                 flux_accounting_error=float(abs(sum(stats[k] for k in ['aperture_miss','tir','backward','outside_screen','captured'])-total)/total))
    scale=target.max()
    stats['display_saturation_fraction']=float(np.mean(counts>scale))
    return stats,counts,target
