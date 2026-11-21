import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix

def operators(xy,f):
    p=xy[f];m=np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0]],axis=1)
    inv=np.linalg.inv(m)
    # dh = inv(M) @ [h1-h0,h2-h0]
    coeff=np.concatenate([-inv.sum(axis=2,keepdims=True),inv],axis=2)
    rows=np.repeat(np.arange(len(f)),3);cols=f.ravel()
    gx=coo_matrix((coeff[:,0,:].ravel(),(rows,cols)),shape=(len(f),len(xy))).tocsr()
    gy=coo_matrix((coeff[:,1,:].ravel(),(rows,cols)),shape=(len(f),len(xy))).tocsr()
    area=abs(np.linalg.det(m))/2
    edges=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    edges=np.unique(edges,axis=0)
    return gx,gy,area,edges

def clean_convex_mesh(points):
    """Drop degenerate boundary triangles and return a single convex boundary."""
    points=np.asarray(points);kept=np.arange(len(points))
    for _ in range(8):
        tri=Delaunay(points);f=tri.simplices.copy()
        a=points[f[:,1]]-points[f[:,0]];b=points[f[:,2]]-points[f[:,0]]
        cross=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        scale=np.maximum.reduce([(a*a).sum(1),(b*b).sum(1),((a-b)**2).sum(1)])
        f=f[abs(cross)>1e-10*scale]
        if not len(f):raise ValueError('Mask is numerically flat')
        used=np.unique(f)
        if len(used)==len(points):break
        points=points[used];kept=kept[used]
    else:raise RuntimeError('Unable to clean boundary triangulation')
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    _,inverse,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary_edges=edges[count[inverse]==1]
    successor=dict(boundary_edges.tolist())
    if len(successor)!=len(boundary_edges):raise RuntimeError('Nonmanifold boundary')
    start=int(boundary_edges[0,0]);ring=[start];current=successor[start]
    while current!=start:
        ring.append(current)
        if len(ring)>len(boundary_edges):raise RuntimeError('Boundary does not close')
        current=successor[current]
    if len(ring)!=len(boundary_edges):raise RuntimeError('Expected one convex boundary')
    return points,f,np.asarray(ring),tri,kept
