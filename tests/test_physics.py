import unittest,tempfile
from pathlib import Path
import numpy as np
from caustic.backend import jax,jnp
from caustic.config import Config
from caustic.optics import screen_hits,numpy_refract
from caustic.geometry import disk_mesh,operators
from caustic.export import export
from caustic.optimize import splat
class Physics(unittest.TestCase):
    def test_plane(self):
        p=jnp.array([[2.,3.],[-4.,1.]])
        h,q,d=screen_hits(p,jnp.array([7.,19.]),jnp.zeros((2,2)),100.,1.5,1.)
        np.testing.assert_allclose(h,p);np.testing.assert_allclose(d,[[0,0,1],[0,0,1]])
        self.assertTrue(jax.config.jax_enable_x64)
    def test_snell_and_numpy(self):
        slopes=np.array([[.2,.1],[-.4,.3]])
        n=np.c_[-slopes,np.ones(2)];n/=np.linalg.norm(n,axis=1,keepdims=True)
        d,q,_=numpy_refract(n,1.5,1)
        hits,qj,dj=screen_hits(jnp.zeros((2,2)),jnp.array([8.,13.]),jnp.asarray(slopes),100.,1.5,1.)
        np.testing.assert_allclose(d,dj,atol=1e-13)
        np.testing.assert_allclose(hits,(100-np.array([8,13]))[:,None]*d[:,:2]/d[:,2,None])
        # Tangential momentum conservation at the dielectric interface.
        ki=np.broadcast_to([0.,0.,1.],d.shape)
        np.testing.assert_allclose(1.5*(ki-(ki*n).sum(1)[:,None]*n),d-(d*n).sum(1)[:,None]*n,atol=1e-13)
    def test_image_coordinate_and_no_renormalization(self):
        # (+x,+y) goes to upper right in PNG coordinates.
        I=np.asarray(splat(jnp.array([[2.,2.],[20.,20.]]),jnp.array([.4,.6]),10.,10.,40,40))
        self.assertAlmostEqual(I.sum(),.4)
        row,col=np.unravel_index(I.argmax(),I.shape)
        self.assertLess(row,20);self.assertGreater(col,20)
    def test_pixel_centering_preserves_detail(self):
        from caustic.target import center_pattern
        image=np.zeros((19,23));image[2:8,1]=np.arange(6)+1;image[4,2:9]=.25
        centered,shift=center_pattern(image)
        np.testing.assert_array_equal(np.sort(centered.ravel()),np.sort(image.ravel()))
        yy,xx=np.nonzero(centered)
        self.assertLessEqual(abs((yy.min()+yy.max())/2-(image.shape[0]-1)/2),.5)
        self.assertLessEqual(abs((xx.min()+xx.max())/2-(image.shape[1]-1)/2),.5)
    def test_tir(self):
        n=np.array([[-1.,0.,1.]])/np.sqrt(2)
        d,q,t=numpy_refract(n,1.5,1)
        self.assertLess(q[0],0)
    def test_gradient_and_closed_mesh(self):
        c=Config();xy,f,b,tri=disk_mesh(c,16);gx,gy,area,edges=operators(xy,f)
        z=8+.04*xy[:,0]-.03*xy[:,1]
        np.testing.assert_allclose(gx@z,.04,atol=1e-10);np.testing.assert_allclose(gy@z,-.03,atol=1e-10)
        self.assertAlmostEqual(area.sum(),.5*c.boundary_segments*30**2*np.sin(2*np.pi/c.boundary_segments),places=8)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            result=export(xy,f,b,z,c,Path(td));self.assertTrue(result['watertight']);self.assertEqual(result['euler_number'],2)
            import trimesh
            precise=trimesh.load_mesh(Path(td)/'lens_precision.stl')
            expected=np.vstack([np.c_[xy,z],np.c_[xy,np.zeros(len(xy))]])
            sort=lambda a:a[np.lexsort((a[:,2],a[:,1],a[:,0]))]
            np.testing.assert_allclose(sort(precise.vertices),sort(expected),rtol=0,atol=1e-14)
    def test_local_refinement_preserves_plane_and_solid(self):
        from caustic.adaptive import refine_mesh
        c=Config();c.boundary_segments=64
        xy,f,b,tri=disk_mesh(c,16);z=8+.02*xy[:,0]-.01*xy[:,1]
        gray=np.zeros((64,64));gray[:,32]=1
        xx,ff,bb,zz,tt,record=refine_mesh(xy,f,z,b,gray,c,1)
        self.assertGreater(len(xx),len(xy))
        np.testing.assert_allclose(zz,8+.02*xx[:,0]-.01*xx[:,1],atol=1e-12)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            result=export(xx,ff,bb,zz,c,Path(td));self.assertTrue(result['watertight'])
    def test_scatter_flux_and_derivative(self):
        f=lambda x:jnp.sum(splat(x,jnp.array([.3,.7]),10.,10.,32,32)**2)
        x=jnp.array([[1.123,.234],[-1.12,-.234]])
        I=splat(x,jnp.array([.3,.7]),10.,10.,32,32)
        self.assertAlmostEqual(float(I.sum()),1)
        g=np.asarray(jax.grad(f)(x));eps=1e-6
        xp=np.asarray(x).copy();xm=xp.copy();xp[0,0]+=eps;xm[0,0]-=eps
        self.assertAlmostEqual(g[0,0],float((f(xp)-f(xm))/(2*eps)),places=6)
if __name__=='__main__':unittest.main()
