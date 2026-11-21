"""OT configuration loading and validation."""
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path

from common.runtime import resolve_runtime, validate_runtime

_ROOT = Path(__file__).resolve().parent.parent
_RUN_KEYS = {
    'image', 'mask', 'out', 'refractive_index', 'air_index', 'L_mm', 'distance_mm',
    'target_size_mm', 'mesh_n', 'center_target', 'mask_threshold', 'coarse_mesh_levels',
    'optics', 'stages', 'solver', 'runtime',
}
_TOP_LEVEL_OPTICS = {
    'L_mm', 'refractive_index', 'air_index', 'distance_mm', 'target_size_mm',
    'center_target', 'mesh_levels',
}


@dataclass
class Config:
    center_target: bool = True
    L_mm: float = 60.
    distance_mm: float = 100.  # from entrance z=0
    refractive_index: float = 1.5
    air_index: float = 1.
    target_size_mm: float = 50.
    base_mm: float = 9.
    min_thickness_mm: float = 2.
    max_thickness_mm: float = 22.
    ot_sites: int = 4096
    ot_resolution: int = 512
    solver: dict = field(default_factory=dict)
    boundary_segments: int = 512
    mesh_levels: list = field(default_factory=lambda: [48, 120, 300, 600])
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
        if self.L_mm <= 0 or self.distance_mm <= 0 or self.target_size_mm <= 0:
            raise ValueError('L_mm, distance_mm and target_size_mm must be positive')
        if not self.refractive_index > self.air_index > 0:
            raise ValueError('Need refractive_index > air_index > 0')
        if not 0 < self.min_thickness_mm < self.base_mm < self.max_thickness_mm < self.distance_mm:
            raise ValueError('Need 0 < min_thickness_mm < base_mm < max_thickness_mm < distance_mm; set thickness values in optics')
        assert self.fine_quadrature >= 0 and self.thin_weight >= 1 and self.mapping_anchor >= 0
        assert self.ot_sites >= 16 and self.boundary_segments >= 32
        assert self.ot_resolution >= 16 and self.image_resolution >= 16
        assert self.mesh_levels and all(n >= 8 for n in self.mesh_levels)
        assert 10 <= self.trace_power <= 28
        expected = {
            'sdot': {'tol', 'max_iter', 'density_stages'},
            'lsmr': {'damp', 'atol', 'btol', 'conlim', 'maxiter'},
            'adam': {'learning_rate', 'steps', 'b1', 'b2', 'eps', 'eps_root', 'nesterov'},
            'lbfgsb': {'maxls', 'ftol', 'gtol', 'maxcor', 'maxfun'},
        }
        if not isinstance(self.solver, dict) or set(self.solver) != set(expected):
            raise ValueError('solver must contain exactly sdot, lsmr, adam, lbfgsb')
        for group, keys in expected.items():
            values = self.solver[group]
            if not isinstance(values, dict) or set(values) != keys:
                raise ValueError('Unexpected or missing solver.' + group + ' keys')
            for key, value in values.items():
                if key in ['density_stages', 'nesterov']:
                    continue
                if type(value) not in [int, float] or not math.isfinite(value):
                    raise ValueError('solver.' + group + '.' + key + ' must be finite')
        sd, ls, ad, lb = (self.solver[k] for k in ('sdot', 'lsmr', 'adam', 'lbfgsb'))
        stages = sd['density_stages']
        if (not isinstance(stages, list) or len(stages) < 2 or stages[0] != 0 or stages[-1] != 1
                or any(type(a) not in [int, float] or not math.isfinite(a) for a in stages)
                or any(a >= b for a, b in zip(stages, stages[1:]))):
            raise ValueError('solver.sdot.density_stages must increase strictly from 0 to 1')
        for group, key, minimum in [('sdot', 'max_iter', 1), ('lsmr', 'maxiter', 1), ('adam', 'steps', 0),
                                    ('lbfgsb', 'maxls', 1), ('lbfgsb', 'maxcor', 1), ('lbfgsb', 'maxfun', 1)]:
            value = self.solver[group][key]
            if type(value) is not int or value < minimum:
                raise ValueError('Invalid solver.' + group + '.' + key)
        if sd['tol'] <= 0 or ls['damp'] < 0 or ls['atol'] < 0 or ls['btol'] < 0 or ls['conlim'] <= 0:
            raise ValueError('Invalid SDOT/LSMR tolerances')
        if (ad['learning_rate'] <= 0 or not 0 <= ad['b1'] < 1 or not 0 <= ad['b2'] < 1
                or ad['eps'] <= 0 or ad['eps_root'] < 0 or type(ad['nesterov']) is not bool):
            raise ValueError('Invalid Adam parameters')
        if lb['ftol'] < 0 or lb['gtol'] < 0:
            raise ValueError('Invalid L-BFGS-B tolerances')


def load_run_config(path=None):
    path = Path(path).resolve() if path else _ROOT / 'OT/config/general_high.json'
    settings = json.loads(path.read_text(encoding='utf-8'))
    missing = (_RUN_KEYS - {'runtime'}) - set(settings)
    unknown = set(settings) - _RUN_KEYS
    if missing:
        raise ValueError('Missing configuration keys: ' + ', '.join(sorted(missing)))
    if unknown:
        raise ValueError('Unknown configuration keys: ' + ', '.join(sorted(unknown)))
    for key in ['image', 'mask', 'out']:
        if settings[key]:
            value = Path(os.path.expandvars(settings[key])).expanduser()
            settings[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    settings['runtime'] = resolve_runtime(settings.get('runtime', {}), path.parent)
    validate_run_settings(settings)
    return settings


def validate_run_settings(s):
    if not isinstance(s['mesh_n'], int) or s['mesh_n'] < 8:
        raise ValueError('mesh_n must be an integer >=8')
    validate_runtime(s.get('runtime', {}))
    if not 0 <= s['mask_threshold'] < 255:
        raise ValueError('mask_threshold must be in [0,254]')
    if not isinstance(s['stages'], list):
        raise ValueError('stages must be a list; [] disables refinement')
    if not isinstance(s['optics'], dict):
        raise ValueError('optics must be an object')
    if not isinstance(s['solver'], dict):
        raise ValueError('solver must be an object')
    if (not isinstance(s['coarse_mesh_levels'], list)
            or not all(isinstance(n, int) and n >= 8 for n in s['coarse_mesh_levels'])):
        raise ValueError('coarse_mesh_levels must be a list of integers >=8; [] uses only mesh_n')
    build_config(s, pixels=16)


def build_config(s, pixels):
    values = dict(s['optics'])
    if _TOP_LEVEL_OPTICS.intersection(values):
        raise ValueError('Set physical parameters at the top level, not in optics')
    values['solver'] = s['solver']
    for key in ['ot_resolution', 'image_resolution']:
        values[key] = max(16, values[key] or pixels)
    levels = s['coarse_mesh_levels'] + [s['mesh_n']]
    values.update(
        L_mm=s['L_mm'],
        refractive_index=s['refractive_index'],
        air_index=s['air_index'],
        distance_mm=s['distance_mm'],
        target_size_mm=s['target_size_mm'],
        center_target=s['center_target'],
        mesh_levels=sorted(set(min(n, s['mesh_n']) for n in levels)),
    )
    c = Config(**values)
    c.validate()
    return c
