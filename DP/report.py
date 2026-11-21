"""DP figures and independent raytrace."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import logging
import numpy as np
from scipy.stats import qmc
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from common.projection import refract, screen_hits, screen_pixels

def save_figures(out,v0,v1,f,target,image0,image1,history):
    limits=(min(v0[:,2].min(),v1[:,2].min()),max(v0[:,2].max(),v1[:,2].max()))
    fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,v,title in zip(axs[:2],[v0,v1],['Before DP','After DP']):
        im=ax.tripcolor(mtri.Triangulation(v[:,0],v[:,1],f),v[:,2],cmap='viridis',vmin=limits[0],vmax=limits[1],rasterized=True)
        ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
    fig.colorbar(im,ax=axs[:2],label='Height (mm)')
    delta=v1[:,2]-v0[:,2]
    im=axs[2].tripcolor(mtri.Triangulation(v0[:,0],v0[:,1],f),delta,cmap='coolwarm',rasterized=True)
    axs[2].set_aspect('equal');axs[2].set_title('Height change');fig.colorbar(im,ax=axs[2],label='mm')
    fig.savefig(out/'height.png',dpi=160);plt.close(fig)
    target_flux=target/target.sum();vmax=float(target_flux.max())
    fig,axs=plt.subplots(1,4,figsize=(18,5),layout='constrained')
    for ax,a,title in zip(axs[:3],[target_flux,image0,image1],['Target','Before DP: independent rays','After DP: independent rays']):
        ax.imshow(a,cmap='gray',vmin=0,vmax=vmax);ax.set_title(title);ax.axis('off')
    im=axs[3].imshow(image1-target_flux,cmap='coolwarm',vmin=-vmax,vmax=vmax)
    axs[3].set_title('After minus target');axs[3].axis('off');fig.colorbar(im,ax=axs[3])
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    steps = [row['iteration'] for row in history]
    axes[0].plot(steps, [row['loss'] for row in history])
    axes[0].set_title('Total loss')
    for name in history[0]['terms']['weighted']:
        values = np.array([row['terms']['weighted'][name] for row in history])
        base = abs(values[0]) or np.max(np.abs(values)) or 1.
        axes[1].plot(steps, values / base, label=name)
    axes[1].set_title('Weighted loss terms / initial magnitude (or maximum)')
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel('Accepted iteration')
        ax.grid(alpha=.3)
    fig.savefig(out/'loss.png',dpi=160)
    plt.close(fig)

def save_mapping(out,hits0,hits1,flux0,flux1):
    fig,axs=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    ids=np.linspace(0,len(flux0)-1,min(15000,len(flux0)),dtype=int)
    for ax,h,flux,title in zip(axs,[hits0,hits1],[flux0,flux1],['Before DP refracted face centers','After DP refracted face centers']):
        p=h.mean(1)[ids]
        ax.scatter(p[:,0],p[:,1],s=.4,c=flux[ids],cmap='viridis',rasterized=True)
        ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('Screen x (mm)');ax.set_ylabel('Screen y (mm)')
    fig.savefig(out/'mapping.png',dpi=160);plt.close(fig)


def raytrace(v,f,gray,c):
    tri=v[f]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=.5*normal[:,2]
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    optics=c['optics']
    n1,n2=optics['refractive_index'],optics['air_index']
    d,q,exit_transmission=refract(normal,n1,n2)
    cdf=np.cumsum(area);cdf/=cdf[-1]
    engine=qmc.Sobol(3,scramble=True,seed=c['validation']['seed'])
    total=2**c['validation']['trace_power'];res=len(gray)
    image=np.zeros((res,res));physical=image.copy()
    tir=back=outside=0
    transmission=exit_transmission*(1-((n1-n2)/(n1+n2))**2)
    for start in range(0,total,2**18):
        u=engine.random(min(2**18,total-start))
        ids=np.searchsorted(cdf,u[:,0],side='right')
        r=np.sqrt(u[:,1]);bary=np.c_[1-r,r*(1-u[:,2]),r*u[:,2]]
        p=np.einsum('ni,nij->nj',bary,tri[ids])
        hits=screen_hits(p[:,:2],p[:,2],d[ids],optics['screen_z_mm'])
        py,px=screen_pixels(hits,optics['target_size_mm'],optics['target_size_mm'],gray.shape)
        good=(q[ids]>0)&(d[ids,2]>0)
        inside=good&(px>=0)&(px<res)&(py>=0)&(py<res)
        tir+=int((q[ids]<=0).sum());back+=int(((q[ids]>0)&(d[ids,2]<=0)).sum())
        outside+=int((good&~inside).sum())
        pixels=py[inside]*res+px[inside]
        image+=np.bincount(pixels,minlength=res*res).reshape(res,res)/total
        physical+=np.bincount(pixels,weights=transmission[ids[inside]],minlength=res*res).reshape(res,res)/total
    target=gray/gray.sum();dark=gray==0
    foreground=gray>.05
    skeleton=skeletonize(foreground)
    thin=skeleton&(distance_transform_edt(foreground)<2)
    metric=dict(incident_rays=total,tir=tir,tir_faces=int((q<=0).sum()),
        min_snell_discriminant=float(q.min()),backward=back,outside_screen=outside,
        captured_flux=float(image.sum()),physical_transmitted_screen_flux=float(physical.sum()),
        dark_leak_incident=float(image[dark].sum()),
        rmse_relative=float(np.sqrt(np.mean((image-target)**2)/np.mean(target**2))),
        thin_pixels=int(thin.sum()),
        thin_flux_ratio=float(image[thin].sum()/target[thin].sum()) if thin.any() else None,
        min_thickness_mm=float(v[:,2].min()),max_thickness_mm=float(v[:,2].max()))
    logging.info('Independent trace: RMSE %.6f; leakage %.4f%%; flux %.4f%%; TIR faces %d',
        metric['rmse_relative'],100*metric['dark_leak_incident'],100*metric['captured_flux'],metric['tir_faces'])
    return image,metric
