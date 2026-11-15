"""Explicit helper: pad a target to a square without changing any original pixel."""
from pathlib import Path
import argparse
from PIL import Image,ImageDraw
def main():
    p=argparse.ArgumentParser();p.add_argument('--image',default='target_saturn.png');p.add_argument('--out',default='examples')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=Image.open(a.image).convert('RGBA');side=max(raw.size)
    square=Image.new('RGBA',(side,side),(0,0,0,255));square.paste(raw,((side-raw.width)//2,(side-raw.height)//2))
    square.save(out/'target_square.png')
    mask=Image.new('L',(256,256),0);ImageDraw.Draw(mask).ellipse((8,8,247,247),fill=255);mask.save(out/'mask_circle.png')
    quad=Image.new('L',(64,64),0);ImageDraw.Draw(quad).polygon([(8,10),(54,4),(60,50),(12,58)],fill=255);quad.save(out/'mask_convex.png')
    print('Explicitly padded target:',square.size,'original:',raw.size,'outputs:',out.resolve())
if __name__=='__main__':main()
