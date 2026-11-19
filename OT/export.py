import numpy as np
import trimesh
from common.mesh_io import export_surface


def export(xy, f, boundary, z, c, out):
    if z.min() < c.min_thickness_mm - 1e-8:
        raise RuntimeError('Thin lens')
    mesh = export_surface(np.c_[xy, z], f, boundary, out / 'lens.stl',
                          ascii_stl=c.ascii_stl, fix_normals=True)
    reread=trimesh.load_mesh(out/'lens.stl')
    if not reread.is_watertight: raise RuntimeError('STL round-trip failed')
    return dict(watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),
                euler_number=int(mesh.euler_number),volume_mm3=float(mesh.volume),vertices=len(mesh.vertices),faces=len(mesh.faces),
                min_thickness_mm=float(z.min()),max_thickness_mm=float(z.max()),
                cnc_height_field=True,tool_radius_compensation=False,ascii_precision_stl=c.ascii_stl,stl_units='mm (STL itself is unitless)')
