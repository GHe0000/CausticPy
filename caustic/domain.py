"""Convex lens footprint from a square binary mask, using only NumPy/SciPy."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull,Delaunay
from scipy.ndimage import binary_erosion
from scipy.stats import qmc

class MaskDomain:
    def __init__(self,path,length_mm,threshold=127):
        self.path=Path(path).resolve();self.length=float(length_mm)
        if not self.length>0:raise ValueError('L must be positive')
        image=Image.open(path).convert('RGBA')
        if image.width!=image.height:raise ValueError(f'Mask must be square, got {image.size}')
        if image.width<2:raise ValueError('Mask must be at least 2x2')
        a=np.asarray(image,dtype=np.float64)
        gray=(.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2])*a[:,:,3]/255
        self.mask=gray>threshold;self.n=image.width;self.pixel=self.length/self.n
        if not self.mask.any():raise ValueError('Mask has no white lens region')
        row,col=np.nonzero(self.mask&~binary_erosion(self.mask,border_value=0))
        corners=np.vstack([np.c_[col+dx,row+dy] for dx,dy in [(0,0),(1,0),(1,1),(0,1)]])
        corners=np.unique(corners,axis=0)
        points=np.c_[-self.length/2+corners[:,0]*self.pixel,self.length/2-corners[:,1]*self.pixel]
        hull=ConvexHull(points)
        self.vertices=points[hull.vertices] # CCW
        self.directions=hull.equations[:,:2];self.offsets=-hull.equations[:,2]
        self.area=float(hull.volume);self.center=self.vertices.mean(axis=0)
        edge=np.roll(self.vertices,-1,axis=0)-self.center
        p=self.vertices-self.center
        self.fan_area=(p[:,0]*edge[:,1]-p[:,1]*edge[:,0])/2
        self.fan_cdf=np.cumsum(self.fan_area);self.fan_cdf/=self.fan_cdf[-1]
    def contains(self,xy,margin=0):
        inside=np.ones(len(xy),bool)
        for direction,offset in zip(self.directions,self.offsets):
            inside&=xy@direction<=offset-margin
        return inside
    def sample(self,u):
        index=np.minimum(np.searchsorted(self.fan_cdf,u[:,0],side='right'),len(self.vertices)-1)
        root=np.sqrt(u[:,1]);b1=root*(1-u[:,2]);b2=root*u[:,2]
        return (1-root)[:,None]*self.center+b1[:,None]*self.vertices[index]+b2[:,None]*self.vertices[(index+1)%len(self.vertices)]
    def seeds(self,count,seed):
        u=qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]
        return self.sample(u)
    def sdot_distribution(self):
        import sdot
        return sdot.Polytope(self.directions,self.offsets,volume=self.area)
    def metadata(self):
        raw_area=float(self.mask.sum()*self.pixel**2)
        return dict(path=str(self.path),shape=list(self.mask.shape),L_mm=self.length,pixel_mm=self.pixel,
                    area_mm2=self.area,raster_white_area_mm2=raw_area,hull_area_added_mm2=self.area-raw_area,
                    hull_vertices=len(self.vertices),components=1,holes=0,
                    geometry='convex hull of white pixel squares; mask assumed convex')
    def mesh(self,c,n):
        spacing=self.length/n
        axis=np.linspace(-self.length/2,self.length/2,n+1)
        xx,yy=np.meshgrid(axis,axis);inside=np.c_[xx.ravel(),yy.ravel()]
        inside=inside[self.contains(inside,margin=.35*spacing)]
        ring=[]
        for a,b in zip(self.vertices,np.roll(self.vertices,-1,axis=0)):
            count=max(1,int(np.ceil(np.linalg.norm(b-a)/spacing)))
            ring.append(a+np.arange(count)[:,None]/count*(b-a))
        ring=np.vstack(ring)
        from .geometry import clean_convex_mesh
        xy,faces,boundary,tri,_=clean_convex_mesh(np.vstack([inside,ring]))
        p=xy[faces]
        a=p[:,1]-p[:,0];b=p[:,2]-p[:,0]
        area=abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]).sum()/2
        if not np.isclose(area,self.area,rtol=1e-9,atol=1e-10):raise RuntimeError('Mask mesh area mismatch')
        return xy,faces,boundary,tri
