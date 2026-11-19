from .backend import jax,jnp
@jax.jit
def refract(slopes,n_lens=1.5,n_air=1.):
    normal=jnp.concatenate([-slopes,jnp.ones((len(slopes),1))],axis=1)
    normal=normal/jnp.linalg.norm(normal,axis=1,keepdims=True)
    eta=n_lens/n_air;cosi=normal[:,2]
    discriminant=1-eta**2*(1-cosi**2)
    direction=eta*jnp.array([0.,0.,1.])+(jnp.sqrt(jnp.maximum(discriminant,1e-12))-eta*cosi)[:,None]*normal
    return direction,discriminant
@jax.jit
def screen_hits(xy,z,slopes,screen_z,n_lens,n_air):
    d,q=refract(slopes,n_lens,n_air)
    hits=xy+(screen_z-z)[:,None]*d[:,:2]/jnp.maximum(d[:,2,None],1e-6)
    return hits,q,d
