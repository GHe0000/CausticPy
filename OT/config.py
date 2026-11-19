from dataclasses import dataclass,field
@dataclass
class Config:
    center_target: bool = True
    diameter_mm: float = 60.
    screen_z_mm: float = 100. # measured from flat entrance plane z=0
    n_lens: float = 1.5
    n_air: float = 1.
    target_width_mm: float = 50.
    base_mm: float = 9.
    min_thickness_mm: float = 2.
    max_thickness_mm: float = 22.
    ot_sites: int = 4096
    ot_resolution: int = 512
    solver: dict = field(default_factory=dict)
    boundary_segments: int = 512
    mesh_levels: list = field(default_factory=lambda:[48,120,300,600])
    fit_iterations: int = 450
    image_iterations: int = 200
    mapping_anchor: float = .03
    normal_weight: float = 10.
    slope_regularization: float = 1e-5
    mean_height_weight: float = 1e-3
    image_weight: float = 100.
    edge_weight: float = 20.
    dark_weight: float = 3.
    tir_weight: float = 1e5
    tir_margin: float = .04
    fine_quadrature: int = 32
    ascii_stl: bool = True
    thin_weight: float = 1.
    image_resolution: int = 256
    trace_power: int = 23
    seed: int = 1701
    def validate(self):
        assert self.diameter_mm>0 and self.target_width_mm>0
        assert self.n_lens>self.n_air>0
        assert 0<self.min_thickness_mm<self.base_mm<self.max_thickness_mm<self.screen_z_mm
        assert self.fine_quadrature>=0 and self.thin_weight>=1 and self.mapping_anchor>=0
        assert self.ot_sites>=16 and self.boundary_segments>=32
        assert self.ot_resolution>=16 and self.image_resolution>=16
        assert self.mesh_levels and all(n>=8 for n in self.mesh_levels)
        assert 10<=self.trace_power<=28
        import math
        expected={
            'sdot':{'tol','max_iter','density_stages'},
            'lsmr':{'damp','atol','btol','conlim','maxiter'},
            'adam':{'learning_rate','steps','b1','b2','eps','eps_root','nesterov'},
            'lbfgsb':{'maxls','ftol','gtol','maxcor','maxfun'}}
        if not isinstance(self.solver,dict) or set(self.solver)!=set(expected):
            raise ValueError('solver must contain exactly sdot, lsmr, adam, lbfgsb')
        for group,keys in expected.items():
            values=self.solver[group]
            if not isinstance(values,dict) or set(values)!=keys:
                raise ValueError('Unexpected or missing solver.'+group+' keys')
            for key,value in values.items():
                if key in ['density_stages','nesterov']:continue
                if type(value) not in [int,float] or not math.isfinite(value):
                    raise ValueError('solver.'+group+'.'+key+' must be finite')
        sd=self.solver['sdot'];ls=self.solver['lsmr'];ad=self.solver['adam'];lb=self.solver['lbfgsb']
        stages=sd['density_stages']
        if not isinstance(stages,list) or len(stages)<2 or stages[0]!=0 or stages[-1]!=1 or any(type(a) not in [int,float] or not math.isfinite(a) for a in stages) or any(a>=b for a,b in zip(stages,stages[1:])):
            raise ValueError('solver.sdot.density_stages must increase strictly from 0 to 1')
        for group,key,minimum in [('sdot','max_iter',1),('lsmr','maxiter',1),('adam','steps',0),
                                  ('lbfgsb','maxls',1),('lbfgsb','maxcor',1),('lbfgsb','maxfun',1)]:
            value=self.solver[group][key]
            if type(value) is not int or value<minimum:raise ValueError('Invalid solver.'+group+'.'+key)
        if sd['tol']<=0 or ls['damp']<0 or ls['atol']<0 or ls['btol']<0 or ls['conlim']<=0:
            raise ValueError('Invalid SDOT/LSMR tolerances')
        if ad['learning_rate']<=0 or not 0<=ad['b1']<1 or not 0<=ad['b2']<1 or ad['eps']<=0 or ad['eps_root']<0 or type(ad['nesterov']) is not bool:
            raise ValueError('Invalid Adam parameters')
        if lb['ftol']<0 or lb['gtol']<0:raise ValueError('Invalid L-BFGS-B tolerances')
