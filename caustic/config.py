from dataclasses import dataclass,asdict,field
import json
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
    ot_tol: float = 1e-10
    ot_iterations: int = 500
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
    @classmethod
    def load(cls,path):
        c=cls(**json.loads(open(path,encoding='utf-8').read())) if path else cls()
        c.validate()
        return c
    def save(self,p): p.write_text(json.dumps(asdict(self),indent=2),encoding='utf-8')
