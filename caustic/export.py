import numpy as np
import trimesh
def export(xy,f,boundary,z,c,out):
    nv=len(xy)
    vertices=np.vstack([np.c_[xy,z],np.c_[xy,np.zeros(nv)]])
    bottom=f[:,::-1]+nv
    a,b=boundary.T if np.asarray(boundary).ndim==2 else (boundary,np.roll(boundary,-1))
    sides=np.vstack([np.c_[a,a+nv,b+nv],np.c_[a,b+nv,b]])
    mesh=trimesh.Trimesh(vertices,np.vstack([f,bottom,sides]),process=False)
    mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0: raise RuntimeError('Invalid closed solid')
    if z.min()<c.min_thickness_mm-1e-8: raise RuntimeError('Thin lens')
    # Single STL; preserve high precision when ascii_stl is enabled.
    mesh.export(out/'lens.stl',file_type='stl_ascii' if c.ascii_stl else 'stl')
    reread=trimesh.load_mesh(out/'lens.stl')
    if not reread.is_watertight: raise RuntimeError('STL round-trip failed')
    return dict(watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),
                euler_number=int(mesh.euler_number),volume_mm3=float(mesh.volume),vertices=len(vertices),faces=len(mesh.faces),
                min_thickness_mm=float(z.min()),max_thickness_mm=float(z.max()),
                cnc_height_field=True,tool_radius_compensation=False,ascii_precision_stl=c.ascii_stl,stl_units='mm (STL itself is unitless)')
