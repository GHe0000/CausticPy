import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plots(xy,f,z,mapping,history,I,T,out,c):
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    vmax=T.max()
    for a,data,title in zip(ax,[T,I,abs(I-T)],['Target','Independent facet raytrace','Absolute error']):
        a.imshow(data,vmin=0,vmax=vmax,cmap='gray',origin='upper');a.set_title(title);a.axis('off')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.tripcolor(xy[:,0],xy[:,1],f,z,shading='gouraud',cmap='viridis');fig.colorbar(im,ax=ax,label='z [mm]')
    ax.set_aspect('equal');ax.set_title('Freeform exit surface; entrance z=0');fig.savefig(out/'height.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots();ax.semilogy(history[:,3]);ax.set_xlabel('Objective evaluations (phase scales differ)');ax.set_ylabel('Loss')
    fig.savefig(out/'loss.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots();idx=np.linspace(0,len(mapping.src)-1,min(600,len(mapping.src))).astype(int)
    x=mapping.src[idx];y=mapping.dst[idx]
    ax.quiver(x[:,0],x[:,1],y[:,0]-x[:,0],y[:,1]-x[:,1],angles='xy',scale_units='xy',scale=1,width=.001)
    ax.set_aspect('equal');ax.set_title('SDOT reference correspondence [mm]');fig.savefig(out/'mapping.png',dpi=180);plt.close(fig)
