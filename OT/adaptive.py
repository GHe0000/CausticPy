"""Local source-mesh refinement near independently predicted thin-line screen regions."""
import numpy as np
import logging
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import distance_transform_edt,map_coordinates
from skimage.morphology import skeletonize
from common.projection import refract as numpy_refract
from .target import dimensions
def refine_mesh(xy,f,z,boundary,gray,c,rounds):
    support=gray>.05
    thin=skeletonize(support)&(distance_transform_edt(support)<2)
    distance=distance_transform_edt(~thin)
    w,h=dimensions(gray,c);ny,nx=gray.shape
    record=[]
    for level in range(rounds):
        v=np.c_[xy,z][f];normal=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        direction,q,_=numpy_refract(normal,c.n_lens,c.n_air)
        center=v.mean(axis=1)
        hit=center[:,:2]+(c.screen_z_mm-center[:,2,None])*direction[:,:2]/direction[:,2,None]
        uv=np.c_[(.5-hit[:,1]/h)*ny-.5,(hit[:,0]/w+.5)*nx-.5]
        dist=map_coordinates(distance,uv.T,order=1,mode='constant',cval=1e6)
        selected=np.flatnonzero((dist<6)&(q>0))
        # Add shared edge midpoints, then retriangulate. This produces a conforming mesh,
        # with no hanging vertices; all subsequent normals and traces use the new triangles.
        edges=np.sort(np.vstack([f[selected][:,[0,1]],f[selected][:,[1,2]],f[selected][:,[2,0]]]),axis=1)
        edges=np.unique(edges,axis=0)
        on_boundary=np.zeros(len(xy),bool);on_boundary[boundary]=True
        edges=edges[~(on_boundary[edges[:,0]]&on_boundary[edges[:,1]])]
        mid=xy[edges].mean(axis=1);mz=z[edges].mean(axis=1)
        newxy=np.vstack([xy,mid]);newz=np.r_[z,mz]
        newxy,idx=np.unique(newxy,axis=0,return_index=True);newz=newz[idx]
        # Original boundary positions are unchanged, in their original CCW order.
        lookup={tuple(p):i for i,p in enumerate(newxy)}
        boundary=np.array([lookup[tuple(p)] for p in xy[boundary]])
        from .geometry import clean_convex_mesh
        xy,f,boundary,tri,kept=clean_convex_mesh(newxy);z=newz[kept]
        record.append(dict(round=level,selected_faces=len(selected),vertices=len(xy),faces=len(f)))
        logging.info('Adaptive refinement: %s',record[-1])
    return xy,f,boundary,z,tri,record
