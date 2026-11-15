"""Plain Matplotlib figures for actual height-colored meshes and forward renders."""
from pathlib import Path
import json,numpy as np

def figure(xy,faces,z,I,T,settings):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig=plt.figure(figsize=(16,5.4),layout='constrained')
    ax=fig.add_subplot(131,projection='3d')
    art=ax.plot_trisurf(xy[:,0],xy[:,1],z,triangles=faces,cmap='viridis',
                       linewidth=.12 if len(faces)<30000 else 0,
                       edgecolor=(0,0,0,.18) if len(faces)<30000 else 'none',antialiased=False,shade=False)
    span=np.ptp(xy,axis=0);actual_z=max(float(np.ptp(z)),1e-9);shown_z=max(actual_z,max(span)*.30)
    ax.set_box_aspect((span[0],span[1],shown_z));ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_tick_params(pad=1,labelsize=8);ax.set_xlabel('x [mm]');ax.set_ylabel('y [mm]')
    title=f'Actual mesh: {len(xy):,} vertices'
    if shown_z/actual_z>1.05:title+=f'\nZ display aspect x{shown_z/actual_z:.1f}'
    ax.set_title(title)
    fig.colorbar(art,ax=ax,shrink=.70,pad=.03,label='Height z [mm]')
    width=settings.get('target_size_mm',50);height=width*T.shape[0]/T.shape[1]
    for index,data,title in [(2,T,'Target (normalized flux)'),(3,I,'Independent Snell raytrace')]:
        a=fig.add_subplot(1,3,index)
        a.imshow(data,cmap='gray',vmin=0,vmax=T.max(),origin='upper',
                 extent=[-width/2,width/2,-height/2,height/2],interpolation='nearest')
        a.set_title(title);a.set_xlabel('screen x [mm]');a.set_ylabel('screen y [mm]')
    fig.suptitle(f"n={settings.get('refractive_index',1.5):g}; mask L={settings.get('L_mm',60):g} mm; screen z={settings.get('distance_mm',100):g} mm")
    return fig

def load_result(path):
    path=Path(path).resolve()
    if (path/'result.json').exists():
        result=json.loads((path/'result.json').read_text(encoding='utf-8'))
        final=Path(result['final_directory']);settings=json.loads((path/'settings.json').read_text(encoding='utf-8'))
    else:
        final=path;config=json.loads((path/'config.json').read_text(encoding='utf-8'))
        settings=dict(refractive_index=config['n_lens'],L_mm=config['diameter_mm'],distance_mm=config['screen_z_mm'],
                      target_size_mm=config['target_width_mm'])
    config=json.loads((final/'config.json').read_text(encoding='utf-8'))
    data=np.load(final/f"surface_{config['mesh_levels'][-1]}.npz");ray=np.load(final/'raytrace.npz')
    return data['xy'],data['faces'],data['z'],ray['irradiance'],ray['target'],settings

