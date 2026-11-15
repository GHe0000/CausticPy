import numpy as np
from PIL import Image
def load(path,center=False):
    # Composite alpha onto black. Gray samples are interpreted as relative linear irradiance.
    im=Image.open(path).convert('RGBA')
    a=np.asarray(im,dtype=np.float64)/255
    gray=(.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2])*a[:,:,3]
    if not np.isfinite(gray).all() or gray.sum()<=0: raise ValueError('Target is empty or invalid')
    return center_pattern(gray)[0] if center else gray
def center_pattern(gray):
    yy,xx=np.nonzero(gray>0)
    dy=int(np.rint((gray.shape[0]-1-yy.min()-yy.max())/2))
    dx=int(np.rint((gray.shape[1]-1-xx.min()-xx.max())/2))
    result=np.zeros_like(gray)
    sy=slice(max(0,-dy),min(gray.shape[0],gray.shape[0]-dy))
    sx=slice(max(0,-dx),min(gray.shape[1],gray.shape[1]-dx))
    result[slice(sy.start+dy,sy.stop+dy),slice(sx.start+dx,sx.stop+dx)]=gray[sy,sx]
    if np.count_nonzero(result)!=np.count_nonzero(gray):raise ValueError("Centering clipped nonzero target pixels")
    return result,(dx,dy)
def resize(gray,width):
    height=max(1,round(gray.shape[0]*width/gray.shape[1]))
    # BOX integrates the pixels, does not add a blur or a black-density floor.
    return np.asarray(Image.fromarray(gray.astype('float32')).resize((width,height),Image.Resampling.BOX),dtype=float)
def dimensions(gray,c):
    return c.target_width_mm,c.target_width_mm*gray.shape[0]/gray.shape[1]
def save_gray(path,a):
    Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5)).save(path)
