"""DP mesh topology, refraction geometry and curvature initialization."""

import numpy as np
import jax.numpy as jnp


def topology(faces, vertex_count):
    edges = {}
    neighbors = [set() for _ in range(vertex_count)]
    for i, face in enumerate(np.asarray(faces)):
        for a, b in zip(face, np.roll(face, -1)):
            a, b = int(a), int(b)
            edges.setdefault(tuple(sorted((a, b))), []).append(i)
            neighbors[a].add(b)
            neighbors[b].add(a)
    if any(len(fs) > 2 for fs in edges.values()):
        raise ValueError('Nonmanifold mesh')
    boundary = sorted({v for edge, fs in edges.items() if len(fs) == 1 for v in edge})
    interior = sorted(set(range(vertex_count)) - set(boundary))
    src, dst = [], []
    for i, ns in enumerate(neighbors):
        for j in sorted(ns):
            src.append(i)
            dst.append(j)
    return dict(adjacent=jnp.asarray([fs for fs in edges.values() if len(fs) == 2], dtype=jnp.int32).reshape(-1, 2),
                boundary=np.asarray(boundary, np.int32), interior=jnp.asarray(interior, dtype=jnp.int32),
                src=jnp.asarray(src), dst=jnp.asarray(dst),
                degree=jnp.asarray([len(ns) for ns in neighbors]))


def geometry(vertices, faces, eta, screen_z, aperture_area):
    tri = vertices[faces]
    u, v = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cross = jnp.cross(u, v)
    norm = jnp.linalg.norm(cross, axis=1)
    normal = cross / norm[:, None]
    e1 = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    basis = jnp.stack([e1, jnp.cross(normal, e1)], axis=2)
    q = 1 + eta**2 * (normal[:, 2]**2 - 1)
    direction = (jnp.sqrt(q) - eta * normal[:, 2])[:, None] * normal + jnp.array([0., 0., eta])
    hit = tri[..., :2] + (screen_z - tri[..., 2:]) * direction[:, None, :2] / direction[:, None, 2:]
    area = .5 * cross[:, 2]
    return dict(vertices=vertices, normal=normal, basis=basis, centroid=tri.mean(1),
                area=area, surface_area=.5 * norm, q=q, direction=direction,
                hit=hit, flux=area / aperture_area)


def feasible(vertices, faces, eta, eps1):
    t = np.asarray(vertices)[np.asarray(faces)]
    c = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    norms = np.linalg.norm(c, axis=1)
    if not np.isfinite(t).all() or np.any(norms == 0) or np.any(.5*c[:, 2] <= eps1):
        return False
    return bool(np.all(1+eta**2*((c[:, 2]/norms)**2-1) > 0))


def fit_weingarten(g,adjacent,chunk=8192):
    normal=np.asarray(g['normal']);centroid=np.asarray(g['centroid']);basis=np.asarray(g['basis'])
    face_count=len(normal)
    adjacent=np.asarray(adjacent,dtype=np.int64)
    src=np.r_[adjacent[:,0],adjacent[:,1]];dst=np.r_[adjacent[:,1],adjacent[:,0]]
    order=np.argsort(src,kind='stable');src=src[order];dst=dst[order]
    degree=np.bincount(src,minlength=face_count)
    if np.max(degree,initial=0)>3:raise ValueError('Expected triangular manifold adjacency')
    offsets=np.r_[0,np.cumsum(degree)]
    neighbors=np.zeros((face_count,3),dtype=np.int64)
    slots=np.arange(len(src))-np.repeat(offsets[:-1],degree)
    neighbors[src,slots]=dst
    result=np.zeros((face_count,3));zero_sum=fit_sum=0.;rank_deficient=0
    for start in range(0,face_count,chunk):
        stop=min(start+chunk,face_count);ids=np.arange(start,stop)
        nb=neighbors[ids];valid=np.arange(3)[None,:]<degree[ids,None]
        delta=centroid[nb]-centroid[ids,None]
        dn=normal[nb]-normal[ids,None]
        d=np.einsum('nki,njk->nji',basis[ids],delta)
        b=np.einsum('nki,njk->nji',basis[ids],dn)
        distance=np.linalg.norm(d,axis=-1)
        if np.any(valid&(distance==0)):raise ValueError('Coincident projected face centroids')
        distance=np.where(valid,distance,1.)
        d=np.where(valid[...,None],d/distance[...,None],0.)
        b=np.where(valid[...,None],b/distance[...,None],0.)
        A=np.zeros((len(ids),3,2,3))
        A[:,:,0,0]=d[:,:,0];A[:,:,0,2]=d[:,:,1]
        A[:,:,1,1]=d[:,:,1];A[:,:,1,2]=d[:,:,0]
        A=A.reshape(-1,6,3);rhs=b.reshape(-1,6)
        u,s,vh=np.linalg.svd(A,full_matrices=False)
        cutoff=1e-12*s[:,:1];keep=s>cutoff
        inv=np.divide(1.,s,out=np.zeros_like(s),where=keep)
        sol=np.einsum('nki,nk->ni',vh,inv*np.einsum('nki,nk->ni',u,rhs))
        result[ids]=sol
        residual=np.einsum('nki,ni->nk',A,sol)-rhs
        zero_sum+=float(np.sum(rhs**2));fit_sum+=float(np.sum(residual**2))
        rank_deficient+=int(np.sum(keep.sum(axis=1)<3))
    if not np.isfinite(result).all():raise ValueError('Nonfinite fitted curvature')
    return result,dict(method='per-face symmetric weighted least squares; SVD minimum norm',
        coefficients='a,b,c for [[a,c],[c,b]] in current optimizer tangent frame',
        units='inverse optimizer length (pixel units during DP)',relative_svd_cutoff=1e-12,
        zero_directed_residual_sum=zero_sum,fitted_directed_residual_sum=fit_sum,
        rank_deficient_faces=rank_deficient,faces_without_neighbors=int((degree==0).sum()),
        coefficient_max_abs=float(np.max(np.abs(result),initial=0)))
