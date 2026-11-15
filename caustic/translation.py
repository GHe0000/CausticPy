"""Low-dimensional physical warm start for translating an already verified caustic."""
import numpy as np
from numpy.polynomial.legendre import legvander2d
from scipy.optimize import minimize
from .backend import jax,jnp
from .geometry import operators
from .optics import screen_hits
def translated_surface(xy,f,z,translation,c,out):
    gx,gy,area,_=operators(xy,f)
    degree=7
    basis=legvander2d(xy[:,0]/(c.diameter_mm/2),xy[:,1]/(c.diameter_mm/2),[degree,degree])
    ids=[i*(degree+1)+j for i in range(degree+1) for j in range(degree+1) if i+j<=degree]
    B=jnp.asarray(basis[:,ids]);F=jnp.asarray(f);Z=jnp.asarray(z)
    Gx=jnp.asarray(gx[np.arange(len(f))[:,None],f].toarray());Gy=jnp.asarray(gy[np.arange(len(f))[:,None],f].toarray())
    mass=jnp.asarray(area/area.sum());ctr=jnp.asarray(xy[f].mean(axis=1))
    def projection(height):
        hf=height[F];slopes=jnp.stack([(Gx*hf).sum(1),(Gy*hf).sum(1)],axis=1)
        hit,q,_=screen_hits(ctr,hf.mean(1),slopes,c.screen_z_mm,c.n_lens,c.n_air)
        return hit,q
    old,_=projection(Z);goal=old+jnp.asarray(translation)
    def loss(coeff):
        height=Z+B@coeff;hit,q=projection(height)
        error=jnp.sum(mass*jnp.sum((hit-goal)**2,axis=1))
        thickness=1e5*jnp.mean(jax.nn.relu(c.min_thickness_mm-height)**2+jax.nn.relu(height-c.max_thickness_mm)**2)
        tir=1e5*jnp.sum(mass*jax.nn.relu(.04-q)**2)
        return error+thickness+tir+1e-7*jnp.sum(coeff**2)
    vg=jax.jit(jax.value_and_grad(loss));count=0
    def fun(coeff):
        nonlocal count
        val,g=vg(jnp.asarray(coeff));count+=1
        if count%50==0:print('Global physical translation',count,float(val),flush=True)
        return float(val),np.asarray(g).copy()
    initial=np.zeros(len(ids))
    for k,index in enumerate(ids):
        if index==1:initial[k]=translation[1]/((c.screen_z_mm-c.base_mm)*(c.n_lens/c.n_air-1))*c.diameter_mm/2
        if index==degree+1:initial[k]=translation[0]/((c.screen_z_mm-c.base_mm)*(c.n_lens/c.n_air-1))*c.diameter_mm/2
    result=minimize(fun,initial,jac=True,method='L-BFGS-B',bounds=[(-4,4)]*len(initial),options=dict(maxiter=400,maxls=35,ftol=1e-13,gtol=1e-9,maxcor=30))
    height=np.asarray(Z+B@jnp.asarray(result.x));hit,q=projection(jnp.asarray(height))
    rms=float(jnp.sqrt(jnp.sum(mass*jnp.sum((hit-goal)**2,axis=1))))
    np.savez_compressed(out/'translation_warmstart.npz',basis_indices=ids,coefficients=result.x,z=height)
    return height,dict(coefficients=len(ids),degree=degree,screen_translation_rms_mm=rms,loss=float(result.fun),message=str(result.message))
