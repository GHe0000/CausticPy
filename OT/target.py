import numpy as np
from PIL import Image
def resize(gray,width):
    height=max(1,round(gray.shape[0]*width/gray.shape[1]))
    # BOX integrates the pixels, does not add a blur or a black-density floor.
    return np.asarray(Image.fromarray(gray.astype('float32')).resize((width,height),Image.Resampling.BOX),dtype=float)
def dimensions(gray,c):
    return c.target_width_mm,c.target_width_mm*gray.shape[0]/gray.shape[1]
