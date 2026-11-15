"""Single configuration schema for the generic mask-based program (no heavy imports)."""
from dataclasses import dataclass,field,asdict
from pathlib import Path
import json,os,sys
@dataclass
class RunSettings:
    image: str = ''
    mask: str = ''
    out: str = ''
    refractive_index: float = 1.5
    air_index: float = 1.
    L_mm: float = 60.
    distance_mm: float = 100.
    target_size_mm: float = 50.
    mesh_n: int = 600
    profile: str = 'high'
    threads: int = 0
    center_target: bool = True
    mask_threshold: int = 127
    optics: dict = field(default_factory=dict)
    stages: list | None = None
    def validate(self):
        for key in ['L_mm','distance_mm','target_size_mm']:
            if getattr(self,key)<=0:raise ValueError(key+' must be positive')
        if not self.refractive_index>self.air_index>0:raise ValueError('Need refractive_index > air_index > 0')
        if not isinstance(self.mesh_n,int) or self.mesh_n<8:raise ValueError('mesh_n must be an integer >=8')
        if self.threads<0:raise ValueError('threads must be >=0; 0 means all available cores')
        if not 0<=self.mask_threshold<255:raise ValueError('mask_threshold must be in [0,254]')
        if self.profile not in ['quick','balanced','high']:raise ValueError('Unknown profile')
    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'));c=cls(**data)
        for key in ['image','mask','out']:
            value=getattr(c,key)
            if value and not Path(value).is_absolute():setattr(c,key,str((Path(path).resolve().parent/value).resolve()))
        return c
    def to_dict(self):return asdict(self)
    def optical_config(self,pixels):
        from .config import Config
        presets={
          'quick':dict(ot_sites=128,ot_resolution=min(128,pixels),fit_iterations=40,image_iterations=10,trace_power=18),
          'balanced':dict(ot_sites=2048,ot_resolution=min(512,pixels),fit_iterations=220,image_iterations=80,trace_power=22),
          'high':dict(ot_sites=8192,ot_resolution=pixels,fit_iterations=400,image_iterations=80,trace_power=24)}
        values=presets[self.profile]|dict(image_resolution=max(16,min(384,pixels)),fine_quadrature=32)
        controlled={'diameter_mm','n_lens','n_air','screen_z_mm','target_width_mm','center_target','mesh_levels'}
        if controlled.intersection(self.optics):raise ValueError('Set physical parameters at the top level, not in optics')
        values.update(self.optics)
        values['ot_resolution']=max(16,values['ot_resolution'] or pixels)
        values['image_resolution']=max(16,values['image_resolution'] or pixels)
        levels=[24,self.mesh_n] if self.profile=='quick' else [48,120,300,self.mesh_n]
        values.update(diameter_mm=self.L_mm,n_lens=self.refractive_index,n_air=self.air_index,
                      screen_z_mm=self.distance_mm,target_width_mm=self.target_size_mm,center_target=self.center_target,
                      mesh_levels=sorted(set(min(x,self.mesh_n) for x in levels)))
        c=Config(**values)
        if not 0<c.min_thickness_mm<c.base_mm<c.max_thickness_mm<self.distance_mm:
            raise ValueError('Need 0 < min_thickness_mm < base_mm < max_thickness_mm < distance_mm; set thickness values in optics')
        c.validate();return c
    def refinement_plan(self):
        if self.stages is not None:return self.stages
        if self.profile=='quick':return []
        if self.profile=='balanced':
            return [dict(name='refine',adaptive=1,iterations=180,thin_weight=60,mapping_anchor=.01)]
        return [
          dict(name='adaptive',adaptive=2,iterations=400,thin_weight=60,mapping_anchor=.03),
          dict(name='adaptive_fine',adaptive=1,iterations=400,thin_weight=120,mapping_anchor=.03),
          dict(name='detail',adaptive=0,iterations=240,thin_weight=240,mapping_anchor=.003),
          dict(name='final',adaptive=0,iterations=400,thin_weight=1000,mapping_anchor=.001)]
def configure_runtime(threads=0):
    count=threads or (os.cpu_count() or 1)
    for key in ['CAUSTIC_THREADS','LOOM_NB_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key]=str(count)
    os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ['JAX_ENABLE_X64']='true'
    os.environ['XLA_FLAGS']=f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={count}'
    compiler=Path('D:/Software/msys2/ucrt64/bin/g++.exe')
    if os.name=='nt' and compiler.exists():
        os.environ.setdefault('CXX',str(compiler))
        os.environ['PATH']=str(Path(sys.executable).parent)+';'+str(compiler.parent)+';'+os.environ.get('PATH','')
    return count
