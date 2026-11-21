"""OT lens design CLI."""
from pathlib import Path
import argparse, contextlib, hashlib, json, logging, math, os, shutil, sys, time, traceback

sys.dont_write_bytecode = True

from common.runtime import configure_runtime, collect_environment
from OT.config import load_run_config, validate_run_settings, build_config


def write(path, data):
    import numpy as np
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                    default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item() if hasattr(v, 'item') else str(v)), encoding='utf-8')


@contextlib.contextmanager
def log_stage(name):
    started = time.monotonic()
    logging.info('%s | START', name)
    yield
    logging.info('%s | DONE (%.1fs)', name, time.monotonic() - started)


def no_total_internal_reflection(metrics):
    return metrics['tir'] == 0 and metrics['min_snell_discriminant'] > 0


def select_best(candidates):
    eligible = [row for row in candidates if no_total_internal_reflection(row['metrics'])
                and math.isfinite(row['metrics']['rmse_relative'])]
    return min(eligible, key=lambda row: row['metrics']['rmse_relative'], default=None)


def run(settings):
    from dataclasses import asdict, replace
    import jax
    from common.mesh_io import export_solid_stl
    from common.target import load_gray, center_pattern
    from OT.domain import MaskDomain
    from OT.transport import solve
    from OT.optimize import optimize
    from OT.adaptive import refine_mesh
    from OT.verify import trace
    from OT.report import plots

    out = Path(settings['out']).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for key in ['image', 'mask', 'out']:
        settings[key] = str(Path(settings[key]).resolve())
    started = time.monotonic()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S', stream=sys.stdout, force=True)
    candidates = []
    try:
        logging.info('Python %s; devices=%s; threads=%s; JAX float64=%s',
                     sys.executable, list(map(str, jax.devices())),
                     os.environ.get('CAUSTIC_THREADS', 'system default'), jax.config.jax_enable_x64)
        logging.info(f'Physics: flat entrance z=0; screen z={settings["distance_mm"]:g} mm; n={settings["refractive_index"]:g}/{settings["air_index"]:g}; mask L={settings["L_mm"]:g} mm')
        with log_stage('[1] Square input validation and convex mask'):
            raw = load_gray(settings['image'])
            gray, shift = center_pattern(raw) if settings['center_target'] else (raw, (0, 0))
            domain = MaskDomain(settings['mask'], settings['L_mm'], settings['mask_threshold'])
            config = build_config(settings, gray.shape[1])
            input_info = dict(image=settings['image'], mask=settings['mask'], shape=list(gray.shape), pixel_shift=shift,
                              image_sha256=hashlib.sha256(Path(settings['image']).read_bytes()).hexdigest(),
                              mask_sha256=hashlib.sha256(Path(settings['mask']).read_bytes()).hexdigest(),
                              illumination='uniform parallel +z light restricted to the convex lens aperture')
            logging.info(f'Input {gray.shape}; center shift {shift}; mask area {domain.area:.6f} mm^2; grid spacing L/n={settings["L_mm"]/settings["mesh_n"]:.6f} mm')
        with log_stage('[2] SDOT source capacities and target power diagram'):
            mapping, ot = solve(gray, config, domain)
            logging.info(f"OT mass L1 residual={ot['capacity_L1']:.3e}; source mass={ot['source_mass']:.15g}; target mass={ot['target_mass']:.15g}")
        base = out / 'base'
        base.mkdir()

        def factory(c, n):
            with log_stage(f'Mesh n={n}, spacing={settings["L_mm"]/n:.6f} mm'):
                mesh = domain.mesh(c, n)
                logging.info(f'Mesh: {len(mesh[0]):,} vertices / {len(mesh[1]):,} triangles')
                return mesh

        with log_stage('[3] Multi-resolution JAX/Optax surface design'):
            xy, f, b, z, history = optimize(mapping, gray, config, mesh_factory=factory)

        def evaluate(folder, c, state, hist, label, refinement=None):
            xx, ff, bb, zz = state
            with log_stage(f'{label}: CNC solid export'):
                mesh = export_solid_stl(xx, ff, bb, zz, c.min_thickness_mm, folder, ascii_stl=c.ascii_stl)
            with log_stage(f'{label}: independent forward raytrace ({2**c.trace_power:,} rays)'):
                metrics, I, T = trace(xx, ff, zz, gray, c, domain)
            with log_stage(f'{label}: save four diagnostic figures'):
                plots(xx, ff, zz, mapping, hist, I, T, folder, c)
            row = dict(stage=label, pipeline_complete=True, config=asdict(c), metrics=metrics, mesh=mesh,
                       no_total_internal_reflection=no_total_internal_reflection(metrics), cnc_release=False, model='lens.stl')
            if refinement is not None:
                row['refinement'] = refinement
            write(folder / 'result.json', row)
            candidates.append(row)
            logging.info(f"{label}: leak={metrics['dark_leak_captured']:.3%}; RMSE={metrics['rmse_relative']:.5f}; thin flux={metrics['thin_flux_ratio']}; TIR-free={row['no_total_internal_reflection']}")

        evaluate(base, config, (xy, f, b, z), history, 'base')
        for index, stage in enumerate(settings['stages'], 1):
            name = stage.get('name', f'refine_{index}')
            if Path(name).name != name or name in ['.', '..', 'base']:
                raise ValueError('Invalid stage name')
            folder = out / name
            folder.mkdir()
            c = replace(config, mesh_levels=[settings['mesh_n']], fit_iterations=0,
                        image_iterations=int(stage.get('iterations', 200)),
                        image_resolution=max(16, int(stage.get('resolution', 0)) or gray.shape[1]),
                        thin_weight=float(stage.get('thin_weight', 60)),
                        mapping_anchor=float(stage.get('mapping_anchor', .003)))
            if 'fine_samples' in stage:
                c.fine_quadrature = int(stage['fine_samples'])
            c.validate()
            record = None
            with log_stage(f'[4.{index}] Local mesh refinement and physical image optimization'):
                rounds = int(stage.get('adaptive', 0))
                if rounds:
                    xy, f, b, z, record = refine_mesh(xy, f, z, b, gray, c, rounds)
                    logging.info(f'After local refinement: {len(xy):,} vertices / {len(f):,} triangles')
                xy, f, b, z, history = optimize(mapping, gray, c, initial=(xy, z, f), mesh=(xy, f, b))
            evaluate(folder, c, (xy, f, b, z), history, name, record)
        selected = select_best(candidates)
        if selected is not None:
            for name in ['lens.stl', 'height.png', 'comparison.png', 'loss.png', 'mapping.png']:
                shutil.copyfile(out / selected['stage'] / name, out / name)
        result = dict(pipeline_complete=True, selected_stage=selected['stage'] if selected else None,
                      selection='lowest independent raytrace relative RMSE among TIR-free stages',
                      selection_status='selected' if selected else 'no_tir_free_candidate',
                      no_total_internal_reflection=selected is not None,
                      metrics=selected['metrics'] if selected else None, mesh=selected['mesh'] if selected else None,
                      config=selected['config'] if selected else None, settings=settings, input=input_info, domain=domain.metadata(),
                      transport=ot, environment=collect_environment(), candidates=candidates,
                      cnc_release=False, elapsed_seconds=time.monotonic() - started, model='lens.stl' if selected else None)
        write(out / 'result.json', result)
        if selected:
            logging.info(f'[5] FINISHED. Selected stage: {selected["stage"]}; RMSE={selected["metrics"]["rmse_relative"]:.8g}; model: {out/"lens.stl"}')
        else:
            logging.warning('[5] FINISHED. No TIR-free candidate; no root model selected. Inspect stage result.json files.')
        return result
    except BaseException:
        write(out / 'result.json', dict(pipeline_complete=False, error=traceback.format_exc(), settings=settings,
                                        candidates=candidates, elapsed_seconds=time.monotonic() - started))
        raise


def main():
    parser = argparse.ArgumentParser(description='Design a lens from a convex mask and target PNG using SDOT.')
    parser.add_argument('--config', type=Path, help='Run JSON (default: OT/config/general_high.json)')
    parser.add_argument('--out', type=Path, help='Write to a new result directory')
    args = parser.parse_args()
    settings = load_run_config(args.config)
    if args.out is not None:
        settings['out'] = str(args.out.resolve())
    if not all(settings[key] for key in ['image', 'mask', 'out']):
        parser.error('image, mask and out are required in the run configuration')
    validate_run_settings(settings)
    configure_runtime(settings['runtime'])
    run(settings)


if __name__ == '__main__':
    main()
