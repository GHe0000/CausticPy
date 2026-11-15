import unittest,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import qmc
from caustic.domain import MaskDomain
from caustic.geometry import operators
from caustic.config import Config
from caustic.export import export
from caustic.verify import trace
ROOT=Path(__file__).resolve().parents[1]
class ConvexMaskTests(unittest.TestCase):
    def test_L_pixel_boundary_mesh_and_sampling(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            path=Path(td)/'mask.png';mask=np.zeros((16,16),np.uint8);mask[2:14,3:13]=255;Image.fromarray(mask).save(path)
            d=MaskDomain(path,32)
            self.assertAlmostEqual(d.area,480)
            np.testing.assert_allclose(d.vertices.min(0),[-10,-12]);np.testing.assert_allclose(d.vertices.max(0),[10,12])
            samples=d.sample(qmc.Sobol(3,scramble=True,seed=5).random_base2(12))
            self.assertTrue(d.contains(samples,margin=-1e-12).all())
            xy,f,b,_=d.mesh(Config(),16);gx,gy,areas,_=operators(xy,f)
            self.assertAlmostEqual(areas.sum(),d.area,places=9)
            self.assertTrue(d.contains(xy,margin=-1e-10).all())
            c=Config();z=8+.02*xy[:,0]-.03*xy[:,1]
            np.testing.assert_allclose(gx@z,.02,atol=1e-11);np.testing.assert_allclose(gy@z,-.03,atol=1e-11)
            model=export(xy,f,b,z,c,Path(td));self.assertTrue(model['watertight']);self.assertEqual(model['euler_number'],2)
    def test_flat_mask_raytrace_never_illuminates_black_region(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            out=Path(td);mask=np.zeros((16,16),np.uint8);mask[2:14,3:13]=255;Image.fromarray(mask).save(out/'mask.png')
            d=MaskDomain(out/'mask.png',32);c=Config(diameter_mm=32,target_width_mm=32,trace_power=12)
            xy,f,b,_=d.mesh(c,16);z=np.full(len(xy),8.)
            m,I,T=trace(xy,f,z,np.ones((16,16)),c,out,source_domain=d)
            self.assertEqual(m['aperture_miss'],0);self.assertEqual(m['tir'],0);self.assertAlmostEqual(m['captured_flux'],1)
            self.assertEqual(I[mask==0].sum(),0)
            self.assertIsNone(m['bright_mean_dark_mean_ratio'])
    def test_oblique_boundary_has_finite_gradients_and_closed_solid(self):
        from PIL import ImageDraw
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            out=Path(td);image=Image.new('L',(64,64),0)
            ImageDraw.Draw(image).polygon([(8,10),(54,4),(60,50),(12,58)],fill=255);image.save(out/'mask.png')
            domain=MaskDomain(out/'mask.png',40);c=Config(diameter_mm=40,target_width_mm=50,trace_power=12)
            for n in [24,32]:
                xy,f,b,_=domain.mesh(c,n);gx,gy,area,_=operators(xy,f)
                self.assertTrue(np.isfinite(gx.data).all());self.assertTrue(np.isfinite(gy.data).all())
                self.assertAlmostEqual(area.sum(),domain.area,places=8)
            z=np.full(len(xy),8.);self.assertTrue(export(xy,f,b,z,c,out)['watertight'])
            metrics,_,_=trace(xy,f,z,np.ones((32,32)),c,out,source_domain=domain)
            self.assertEqual(metrics['aperture_miss'],0);self.assertEqual(metrics['captured'],2**12)
    def test_non_square_and_empty_mask_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            path=Path(td)/'mask.png';Image.new('L',(10,12),255).save(path)
            with self.assertRaisesRegex(ValueError,'square'):MaskDomain(path,30)
            Image.new('L',(10,10),0).save(path)
            with self.assertRaisesRegex(ValueError,'white'):MaskDomain(path,30)
if __name__=='__main__':unittest.main()
