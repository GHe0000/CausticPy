"""STL surface I/O and target loading."""
from pathlib import Path
import hashlib
import numpy as np
import trimesh
from .target import center_pattern, load_gray

def load_surface(path):
    solid=trimesh.load_mesh(path,process=True)
    if not solid.is_watertight or not solid.is_winding_consistent:
        raise ValueError('Input STL must be closed with consistent winding')
    tri=solid.vertices[solid.faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    selected=solid.faces[cross[:,2]>0]
    used,inv=np.unique(selected,return_inverse=True)
    v=np.asarray(solid.vertices[used],float)
    f=inv.reshape(-1,3).astype(np.int32)
    if np.min(v[:,2])<=0: raise ValueError('Expected entrance plane z=0 and positive exit height')
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    _,idx,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
    boundary=edges[idx[count==1]]
    if np.any(count>2): raise ValueError('Nonmanifold exit surface')
    with open(path,'rb') as stream: sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return v,f,boundary,dict(input_stl=str(Path(path).resolve()),sha256=sha,
        vertices=len(v),faces=len(f),solid_vertices=len(solid.vertices),solid_faces=len(solid.faces),
        volume_mm3=float(solid.volume),stl_units='mm',
        min_thickness_mm=float(v[:,2].min()),max_thickness_mm=float(v[:,2].max()),
        watertight=bool(solid.is_watertight),winding_consistent=bool(solid.is_winding_consistent))

def export_surface(v, f, boundary, path, *, ascii_stl=True, fix_normals=False):
    n=len(v);base=v.copy();base[:,2]=0
    a,b=boundary.T if np.asarray(boundary).ndim==2 else (boundary,np.roll(boundary,-1))
    sides=np.vstack([np.c_[a,a+n,b+n],np.c_[a,b+n,b]])
    mesh=trimesh.Trimesh(np.vstack([v,base]),np.vstack([f,f[:,::-1]+n,sides]),process=False)
    if fix_normals:
        mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:
        raise ValueError('Invalid output solid')
    mesh.export(path,file_type='stl_ascii' if ascii_stl else 'stl')
    return mesh


def export_solid_stl(xy, f, boundary, z, min_thickness_mm, out, *, ascii_stl=True):
    if z.min() < min_thickness_mm - 1e-8:
        raise RuntimeError('Thin lens')
    stl_path = out / 'lens.stl'
    mesh = export_surface(np.c_[xy, z], f, boundary, stl_path,
                          ascii_stl=ascii_stl, fix_normals=True)
    reread = trimesh.load_mesh(stl_path)
    if not reread.is_watertight:
        raise RuntimeError('STL round-trip failed')
    return dict(watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                euler_number=int(mesh.euler_number), volume_mm3=float(mesh.volume),
                vertices=len(mesh.vertices), faces=len(mesh.faces),
                min_thickness_mm=float(z.min()), max_thickness_mm=float(z.max()),
                cnc_height_field=True, tool_radius_compensation=False,
                ascii_precision_stl=ascii_stl, stl_units='mm (STL itself is unitless)',
                body_count=1)


def load_target(path,center=True):
    gray = load_gray(path)
    gray, (dx, dy) = center_pattern(gray) if center else (gray, (0, 0))
    with open(path,'rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return gray,dict(image=str(path),image_sha256=sha,pixel_shift=[dx,dy],
        linear_grayscale='Rec709 RGB luminance multiplied by alpha; no blur',center_target=center)
