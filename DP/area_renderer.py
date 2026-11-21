"""Exact triangle/pixel overlap rasterizer with custom JAX VJP."""
import jax
import jax.numpy as jnp
import numpy as np

def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def pixel_overlap(tri):
    """Overlap area with [0,1]^2 for CCW triangles (...,3,2)."""
    a=tri;b=jnp.roll(tri,-1,axis=-2);v=b-a
    parallel=jnp.abs(v)<1e-14;safe=jnp.where(parallel,1.,v)
    lower=-a/safe;upper=(1-a)/safe
    enter=jnp.where(parallel,-jnp.inf,jnp.minimum(lower,upper))
    leave=jnp.where(parallel,jnp.inf,jnp.maximum(lower,upper))
    t0=jnp.maximum(0.,jnp.max(enter,axis=-1));t1=jnp.minimum(1.,jnp.min(leave,axis=-1))
    valid=jnp.all((~parallel)|((a>=0)&(a<=1)),axis=-1)&(t1>t0)
    boundary=jnp.sum(jnp.where(valid,(t1-t0)*cross(a,v),0.),axis=-1)
    # Only top/right edges contribute to Green's theorem integral.
    starts=jnp.array([[1.,0.],[1.,1.]])
    directions=jnp.array([[0.,1.],[-1.,0.]])
    f0=cross(v[...,None,:,:],starts[:,None,:]-a[...,None,:,:])
    fd=cross(v[...,None,:,:],directions[:,None,:])
    parallel=jnp.abs(fd)<1e-14;safe=jnp.where(parallel,1.,fd);root=-f0/safe
    enter=jnp.where(fd>1e-14,root,-jnp.inf);leave=jnp.where(fd< -1e-14,root,jnp.inf)
    t0=jnp.maximum(0.,jnp.max(enter,axis=-1));t1=jnp.minimum(1.,jnp.min(leave,axis=-1))
    valid=jnp.all((~parallel)|(f0>1e-14),axis=-1)&(t1>t0)
    boundary+=jnp.sum(jnp.where(valid,t1-t0,0.),axis=-1)
    return jnp.clip(.5*boundary,0.,1.)

def pair_flux(tri,flux,pixel_xy):
    area=.5*cross(tri[...,1,:]-tri[...,0,:],tri[...,2,:]-tri[...,0,:])
    return flux*pixel_overlap(tri-pixel_xy[...,None,:])/jnp.maximum(area,1e-30)

def make_rasterizer(res,chunk=8192):
    def values(hit,flux,fi,pi):
        valid=fi>=0;face=jnp.maximum(fi,0)
        xy=jnp.stack([pi%res,pi//res],axis=-1)
        return jnp.where(valid,pair_flux(hit[face],flux[face],xy),0.)
    def forward(hit,flux,face_ids,pixel_ids):
        def body(image,ids):
            fi,pi=ids;vals=values(hit,flux,fi,pi)
            return image.at[pi].add(vals),None
        image,_=jax.lax.scan(body,jnp.zeros(res*res,jnp.float64),(face_ids,pixel_ids))
        return image.reshape(res,res)
    @jax.custom_vjp
    def raster(hit,flux,face_ids,pixel_ids):return forward(hit,flux,face_ids,pixel_ids)
    def fwd(hit,flux,face_ids,pixel_ids):
        return forward(hit,flux,face_ids,pixel_ids),(hit,flux,face_ids,pixel_ids)
    local_grad=jax.grad(lambda tri,mass,xy,g:jnp.sum(pair_flux(tri,mass,xy)*g),argnums=(0,1))
    def bwd(saved,adjoint):
        hit,flux,face_ids,pixel_ids=saved;image_grad=adjoint.reshape(-1)
        def body(carry,ids):
            fi,pi=ids;valid=fi>=0;face=jnp.maximum(fi,0)
            xy=jnp.stack([pi%res,pi//res],axis=-1)
            dh,df=local_grad(hit[face],flux[face],xy,jnp.where(valid,image_grad[pi],0.))
            return (carry[0].at[face].add(dh),carry[1].at[face].add(df)),None
        result,_=jax.lax.scan(body,(jnp.zeros_like(hit),jnp.zeros_like(flux)),(face_ids,pixel_ids))
        return *result,None,None
    raster.defvjp(fwd,bwd)
    return raster

def bin_triangles(hit_mm,res,width,chunk=8192,capacity=None,growth_factor=0.):
    """Face/pixel bounding-box pairs."""
    tri=(np.asarray(hit_mm)/width+.5)*res
    signed=cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    if not np.isfinite(tri).all() or np.any(np.asarray(signed)<=0):raise ValueError('Expected finite, positively oriented projected triangles')
    lo=np.maximum(np.floor(tri.min(1)).astype(np.int64),0)
    hi=np.minimum(np.floor(tri.max(1)).astype(np.int64),res-1)
    size=np.maximum(hi-lo+1,0);counts=np.prod(size,axis=1)
    face=np.repeat(np.arange(len(tri),dtype=np.int32),counts)
    start=np.repeat(np.cumsum(counts)-counts,counts);offset=np.arange(len(face))-start
    nx=size[face,0];pixel=(lo[face,1]+offset//nx)*res+lo[face,0]+offset%nx
    used=len(face);required=((used+chunk-1)//chunk)*chunk
    if capacity is None or (required>capacity and growth_factor>0):
        capacity=max(chunk,int(np.ceil(required*(1+growth_factor)/chunk))*chunk)
    if required>capacity:raise ValueError(f'Area pair capacity {capacity} < required {required}')
    fi=np.full(capacity,-1,np.int32);pi=np.zeros(capacity,np.int32)
    fi[:used]=face;pi[:used]=pixel
    return (fi.reshape(-1,chunk),pi.reshape(-1,chunk)),dict(pairs=used,capacity=capacity)

def make_parallel_rasterizer(res,chunk=8192):
    """Parallelize rasterization across local JAX devices."""
    base=make_rasterizer(res,chunk)
    devices=jax.local_devices()
    if len(devices)==1:return base
    lanes=len(devices)
    parallel=jax.pmap(base,in_axes=(None,None,0,0),devices=devices)
    def render(hit,flux,face_ids,pixel_ids):
        rows=((face_ids.shape[0]+lanes-1)//lanes)*lanes
        pad=rows-face_ids.shape[0]
        fi=jnp.pad(face_ids,((0,pad),(0,0)),constant_values=-1)
        pi=jnp.pad(pixel_ids,((0,pad),(0,0)),constant_values=0)
        shape=(lanes,rows//lanes,face_ids.shape[1])
        return parallel(hit,flux,fi.reshape(shape),pi.reshape(shape)).sum(axis=0)
    return render
