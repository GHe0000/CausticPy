import numpy as np
import logging
import optax
from scipy.sparse import vstack,csr_matrix
from scipy.sparse.linalg import lsmr
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
from scipy.optimize import minimize
from .backend import jax,jnp
from .geometry import disk_mesh,operators
from .optics import screen_hits
from .target import resize,dimensions
def splat(hits,weights,w,h,nx,ny):
    uv=(hits*jnp.array([1.,-1.])+jnp.array([w/2,h/2]))/jnp.array([w,h])*jnp.array([nx,ny])-.5
    lo=jnp.floor(uv).astype(int);t=uv-lo
    img=jnp.zeros((ny,nx))
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        ij=lo+jnp.array([dx,dy]);ok=(ij[:,0]>=0)&(ij[:,0]<nx)&(ij[:,1]>=0)&(ij[:,1]<ny)
        val=weights*(t[:,0] if dx else 1-t[:,0])*(t[:,1] if dy else 1-t[:,1])*ok
        img=img.at[jnp.clip(ij[:,1],0,ny-1),jnp.clip(ij[:,0],0,nx-1)].add(val)
    return img
def optimize(mapping,gray,c,initial=None,mesh_override=None,mesh_factory=None):
    history=[];old=initial
    w,h=dimensions(gray,c)
    for level,n in enumerate(c.mesh_levels):
        xy,f,boundary,tri=mesh_override if mesh_override is not None else mesh_factory(c,n) if mesh_factory is not None else disk_mesh(c,n);gx,gy,area,edges=operators(xy,f)
        ctr=xy[f].mean(axis=1);goal=mapping(ctr)
        # Sparse gradient integration provides a physically informed initializer.
        if old is None:
            desired=goal-ctr;direction=np.c_[desired,np.full(len(ctr),c.screen_z_mm-c.base_mm)]
            direction/=np.linalg.norm(direction,axis=1,keepdims=True)
            slope=direction[:,:2]/(c.n_lens/c.n_air-direction[:,2,None])
            anchor=csr_matrix(np.ones((1,len(xy)))/len(xy))
            A=vstack([gx.multiply(np.sqrt(area[:,None])),gy.multiply(np.sqrt(area[:,None])),anchor])
            b=np.r_[slope[:,0]*np.sqrt(area),slope[:,1]*np.sqrt(area),0.]
            heights=c.base_mm+lsmr(A,b,**c.solver["lsmr"])[0]
        else:
            if np.array_equal(xy,old[0]):
                heights=np.array(old[1],copy=True)
            else:
                heights=LinearNDInterpolator(old[0],old[1])(xy)
            bad=~np.isfinite(heights)
            if bad.any():heights[bad]=NearestNDInterpolator(old[0],old[1])(xy[bad])
        heights=np.clip(heights,c.min_thickness_mm+.1,c.max_thickness_mm-.1)
        # Dense local gradient coefficients: three nonzero entries per face.
        Gx=jnp.asarray(gx[np.arange(len(f))[:,None],f].toarray());Gy=jnp.asarray(gy[np.arange(len(f))[:,None],f].toarray())
        F=jnp.asarray(f);XY=jnp.asarray(ctr);Y=jnp.asarray(goal);mass=jnp.asarray(area/area.sum())
        E=jnp.asarray(edges)
        target=resize(gray,c.image_resolution);ny,nx=target.shape
        T=jnp.asarray(target/target.sum())
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize
        support=target>.05*target.max()
        thin=skeletonize(support)&(distance_transform_edt(support)<2)
        importance=jnp.asarray(1+(c.thin_weight-1)*thin.astype(float))
        # Fixed independent stratified barycentric samples per face for differentiable image refinement.
        rng=np.random.default_rng(c.seed)
        base_samples=8 if c.image_resolution>512 else 4 if len(f)>150000 else 16
        u=rng.random((max(base_samples,c.fine_quadrature),2));rt=np.sqrt(u[:,0])
        allbary=np.c_[1-rt,rt*(1-u[:,1]),rt*u[:,1]]
        bary=allbary[:base_samples]
        B=jnp.asarray(bary);samplexy=jnp.asarray(np.einsum('sk,fkd->fsd',bary,xy[f]).reshape(-1,2))
        # More quadrature only on source faces illuminating the fine-line neighborhood.
        # Base + extra rays still carry exactly each face's projected source mass.
        fine_ids=np.empty(0,dtype=int)
        if c.image_resolution>512 and c.fine_quadrature>base_samples and thin.any():
            from scipy.ndimage import map_coordinates
            from common.projection import refract as numpy_refract
            slopes0=np.c_[gx@heights,gy@heights]
            normals=np.c_[-slopes0,np.ones(len(f))];normals/=np.linalg.norm(normals,axis=1,keepdims=True)
            dirs0,_,_=numpy_refract(normals,c.n_lens,c.n_air)
            hits0=ctr+(c.screen_z_mm-heights[f].mean(1))[:,None]*dirs0[:,:2]/dirs0[:,2,None]
            distance=distance_transform_edt(~thin)
            def distance_at(p):
                uv=np.c_[(.5-p[:,1]/h)*ny-.5,(p[:,0]/w+.5)*nx-.5]
                return map_coordinates(distance,uv.T,order=1,mode='constant',cval=1e6)
            fine_ids=np.flatnonzero(np.minimum(distance_at(hits0),distance_at(goal))<4)
        denom=np.full(len(f),base_samples,dtype=float)
        denom[fine_ids]=c.fine_quadrature
        smass=jnp.repeat(mass/jnp.asarray(denom),len(bary))
        extra_count=c.fine_quadrature-base_samples if len(fine_ids) else 0
        extra_mass=jnp.zeros(0)
        if extra_count:
            extra_bary=jnp.asarray(allbary[base_samples:]);extra_ids=jnp.asarray(fine_ids)
            extra_xy=jnp.asarray(np.einsum('sk,fkd->fsd',allbary[base_samples:],xy[f[fine_ids]]).reshape(-1,2))
            extra_mass=jnp.repeat(mass[extra_ids]/c.fine_quadrature,extra_count)
        sampling_mass=float(smass.sum()+extra_mass.sum())
        if abs(sampling_mass-1)>1e-12:raise RuntimeError('Quadrature source flux error')
        logging.info('Image quadrature: base=%s, fine_faces=%s, fine_samples=%s, flux=%.15g',base_samples,len(fine_ids),c.fine_quadrature,sampling_mass)
        def components(z,image=False):
            zh=z[F];slopes=jnp.stack([(Gx*zh).sum(axis=1),(Gy*zh).sum(axis=1)],axis=1)
            hits,q,d=screen_hits(XY,zh.mean(axis=1),slopes,c.screen_z_mm,c.n_lens,c.n_air)
            fit=jnp.sum(mass*jnp.sum((hits-Y)**2,axis=1))
            # Normal consistency is evaluated against the Snell-required normal at current z.
            td=jnp.concatenate([Y-XY,(c.screen_z_mm-zh.mean(axis=1))[:,None]],axis=1)
            td/=jnp.linalg.norm(td,axis=1,keepdims=True)
            ns=jnp.concatenate([-slopes,jnp.ones((len(f),1))],axis=1);ns/=jnp.linalg.norm(ns,axis=1,keepdims=True)
            want=c.n_lens*jnp.array([0.,0.,1.])-c.n_air*td;want/=jnp.linalg.norm(want,axis=1,keepdims=True)
            normal=jnp.sum(mass*jnp.sum((ns-want)**2,axis=1))
            barrier=c.tir_weight*jnp.sum(mass*jax.nn.relu(c.tir_margin-q)**2)
            # Mild first derivative regularization; no global blur of target or normals.
            smooth=c.slope_regularization*jnp.mean((z[E[:,1]]-z[E[:,0]])**2)
            loss=fit+c.normal_weight*normal+barrier+smooth+c.mean_height_weight*(z.mean()-c.base_mm)**2
            if image:
                sz=(zh@B.T).reshape(-1)
                sh,_,_=screen_hits(samplexy,sz,jnp.repeat(slopes,len(bary),axis=0),c.screen_z_mm,c.n_lens,c.n_air)
                I=splat(sh,smass,w,h,nx,ny)
                if extra_count:
                    ez=(zh[extra_ids]@extra_bary.T).reshape(-1)
                    eh,_,_=screen_hits(extra_xy,ez,jnp.repeat(slopes[extra_ids],extra_count,axis=0),c.screen_z_mm,c.n_lens,c.n_air)
                    I=I+splat(eh,extra_mass,w,h,nx,ny)
                # No flux renormalization after tracing. Dark pixels receive additional weight.
                image_loss=jnp.mean((I-T)**2*(1+c.dark_weight*(T==0))*importance)*nx*ny
                edge=jnp.sum((jnp.diff(I,axis=1)-jnp.diff(T,axis=1))**2)+jnp.sum((jnp.diff(I,axis=0)-jnp.diff(T,axis=0))**2)
                loss=c.mapping_anchor*loss+c.image_weight*image_loss+c.edge_weight*edge
            return loss
        def run(z,image,steps):
            vg=jax.jit(jax.value_and_grad(lambda z:components(z,image)))
            count=0
            def fun(z):
                nonlocal count
                v,g=vg(jnp.asarray(z));v=float(v)
                if not np.isfinite(v) or not np.isfinite(g).all(): raise RuntimeError('Nonfinite objective')
                count+=1;history.append([level,int(image),count,v])
                if count%getattr(c,"log_every",25)==0: logging.info('level=%s phase=%s eval=%s loss=%.12g',n,'image' if image else 'mapping',count,v)
                return v,np.asarray(g).copy()
            # Optax Adam warm start, followed by quasi-Newton convergence.
            if not image:
                optimizer=optax.adam(**{key:value for key,value in c.solver["adam"].items() if key!="steps"})
                zj=jnp.asarray(z);state=optimizer.init(zj)
                @jax.jit
                def adam_step(zj,state):
                    value,gradient=vg(zj)
                    updates,state=optimizer.update(gradient,state,zj)
                    return jnp.clip(optax.apply_updates(zj,updates),c.min_thickness_mm,c.max_thickness_mm),state
                for _ in range(c.solver["adam"]["steps"]): zj,state=adam_step(zj,state)
                z=np.asarray(zj)
            result=minimize(fun,z,jac=True,method='L-BFGS-B',bounds=[(c.min_thickness_mm,c.max_thickness_mm)]*len(z),
                            options=dict(c.solver["lbfgsb"],maxiter=steps))
            logging.info('optimizer grid=%s image=%s: %s; loss=%.12g',n,image,result.message,result.fun)
            return result.x
        logging.info('Mesh grid=%s: %s vertices / %s faces',n,len(xy),len(f))
        if c.fit_iterations: heights=run(heights,False,c.fit_iterations)
        if c.image_iterations: heights=run(heights,True,c.image_iterations)
        old=(xy,heights,f)
    return xy,f,boundary,heights,np.asarray(history).reshape(-1,4)
