"""DP lens optimization CLI."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

from common.runtime import configure_runtime, resolve_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'DP/config/default.json')
    parser.add_argument('--iterations', type=int, help='Override the L-BFGS iteration budget')
    parser.add_argument('--out', type=Path, help='Write to a new result directory')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    opt = config['optimization']
    if args.iterations is not None:
        opt['iterations'] = args.iterations
    if opt['iterations'] < 1:
        parser.error('iterations must be positive')
    output = (args.out or config_path.parent / config['out']).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')

    configure_runtime(resolve_runtime(config.get('runtime', {}), config_path.parent))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S', stream=sys.stdout)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from DP.area_renderer import bin_triangles, make_parallel_rasterizer
    from DP.lbfgs import minimize_lbfgs
    from common.mesh_io import export_surface, load_surface, load_target
    from DP.loss import image_loss
    from DP.optics import feasible, fit_weingarten, geometry, topology
    from DP.report import raytrace, save_figures, save_mapping

    started = time.perf_counter()
    weights = config['loss_weights']
    if any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Loss weights must be finite and nonnegative')

    gray, target_info = load_target(
        (config_path.parent / config['image']).resolve(), config.get('center_target', True))
    logging.info('JAX %s, %s, float64 %s', jax.__version__, jax.devices(), jax.config.x64_enabled)
    logging.info('Reading STL and target (%d x %d, shift %s)',
                 *gray.shape, target_info['pixel_shift'])
    v0, faces, boundary, source = load_surface(config_path.parent / config['input_stl'])
    logging.info('Loaded %s vertices and %s exit faces', f'{len(v0):,}', f'{len(faces):,}')

    # Convert mm to screen-pixel units.
    optics = config['optics']
    res = len(gray)
    scale = optics['target_size_mm'] / res
    gamma = opt['gamma']
    vertices = v0 / scale
    vertices[:, :2] += res / 2
    eta = optics['refractive_index'] / optics['air_index']
    tris = vertices[faces]
    aperture = .5 * np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2].sum()
    del tris
    logging.info('Building mesh adjacency and fitting curvature')
    topo = topology(faces, len(vertices))
    geo = jax.jit(lambda v: geometry(v, faces, eta, optics['screen_z_mm'] / scale, aperture))
    initial_geometry = geo(jnp.asarray(vertices))
    nu = opt['nu_inverse_mm'] * scale
    min_area = float(initial_geometry['area'].min())
    eps1 = min_area * opt['eps1_min_area_fraction']
    eps2 = min_area * opt['eps2_min_area_fraction']
    curvature, initialization = fit_weingarten(initial_geometry, topo['adjacent'])
    initialization['coefficient_max_abs_inverse_mm'] = initialization['coefficient_max_abs'] / scale
    logging.info('Curvature fit residual %.8g -> %.8g',
                 initialization['zero_directed_residual_sum'], initialization['fitted_directed_residual_sum'])

    # Fix boundary XY; keep boundary Z and interior XYZ free.
    free = np.ones(vertices.shape, dtype=bool)
    free[np.unique(boundary), :2] = False
    free_ids = np.flatnonzero(free.ravel())
    base = jnp.asarray(vertices.ravel())
    x0 = np.concatenate((vertices.ravel()[free_ids], curvature.ravel()))

    def unpack(x):
        v = base.at[free_ids].set(x[:len(free_ids)]).reshape(vertices.shape)
        return v, x[len(free_ids):].reshape(curvature.shape)

    target = jnp.asarray(gray[::-1].copy())  # Flip PNG Y to screen +Y.
    raster = make_parallel_rasterizer(res, opt['raster_chunk'])

    def energy(x, pairs):
        v, w = unpack(x)
        return image_loss(geo(v), w, topo, nu, eps1, eps2, target, gamma, raster, pairs, weights)

    value_and_grad = jax.jit(jax.value_and_grad(energy, has_aux=True))
    capacity = None
    evaluations = 0
    history = []
    last_terms = None
    last_value = None

    def objective(x):
        nonlocal capacity, evaluations, last_terms, last_value
        v, _ = unpack(x)
        hit = np.asarray(geo(v)['hit'])
        pairs, info = bin_triangles(hit - res / 2, res, float(res),
                                    chunk=opt['raster_chunk'], capacity=capacity, growth_factor=.2)
        capacity = info['capacity']
        evaluations += 1
        if evaluations == 1:
            logging.info('Compiling JAX loss for %s face/pixel pairs', f"{info['pairs']:,}")
        (value, terms), gradient = value_and_grad(x, tuple(jnp.asarray(p) for p in pairs))
        last_value = float(value)
        last_terms = jax.tree.map(float, terms)
        if not history:
            record(x)
        return last_value, np.asarray(gradient)

    def feasible_x(x):
        v, _ = unpack(x)
        v = np.asarray(v)
        z = v[:, 2] * scale
        return (z.min() >= optics['min_thickness_mm'] and z.max() <= optics['max_thickness_mm']
                and z.max() < optics['screen_z_mm'] and feasible(v, faces, eta, eps1))

    def record(x):
        v, _ = unpack(x)
        displacement = np.linalg.norm((np.asarray(v) - vertices) * scale, axis=1)
        step = len(history)
        history.append(dict(iteration=step, elapsed_seconds=time.perf_counter()-started,
                            loss=last_value, terms=last_terms,
                            max_displacement_mm=float(displacement.max())))
        logging.info('Iteration %d/%d, loss %.10g, maximum displacement %.6g mm',
                     step, opt['iterations'], last_value, displacement.max())

    logging.info('Starting L-BFGS, budget %d', opt['iterations'])
    result = minimize_lbfgs(objective, x0, feasible=feasible_x, maxiter=opt['iterations'],
                            gtol=opt['gtol'], ftol=opt['ftol'], maxls=opt['maxls'],
                            maxcor=opt['maxcor'], initial_step=opt['initial_step'], callback=record)
    logging.info('%s', result.message)
    best_vertices, _ = unpack(result.best_x)
    v1 = np.asarray(best_vertices).copy()
    v1[:, :2] -= res / 2
    v1 *= scale

    logging.info('Tracing the input STL')
    density = gray**gamma
    image0, metrics0 = raytrace(v0, faces, density, config)
    output.mkdir(parents=True)
    stl = output / 'lens.stl'
    logging.info('Exporting and reloading %s', stl)
    export_surface(v1, faces, boundary, stl)
    exported_v, exported_faces, _, mesh = load_surface(stl)
    expected_counts = dict(vertices=len(v1), faces=len(faces), solid_vertices=2 * len(v1),
                           solid_faces=2 * (len(faces) + len(boundary)))
    if any(mesh[key] != expected for key, expected in expected_counts.items()):
        raise ValueError('STL round trip changed the mesh counts')
    mesh['exit_faces'] = mesh.pop('faces')
    mesh['exit_vertices'] = mesh.pop('vertices')
    mesh['faces'] = mesh.pop('solid_faces')
    mesh['vertices'] = mesh.pop('solid_vertices')
    mesh.pop('input_stl')
    logging.info('Tracing the exported STL')
    image1, metrics1 = raytrace(exported_v, exported_faces, density, config)
    if metrics1['tir_faces'] != 0:
        raise ValueError('Exported STL contains total internal reflection')
    logging.info('Saving plots and result JSON')
    save_figures(output, v0, v1, faces, density, image0, image1, history)
    final_geometry = geo(best_vertices)
    save_mapping(output, (np.asarray(initial_geometry['hit']) - res / 2) * scale,
                 (np.asarray(final_geometry['hit']) - res / 2) * scale,
                 np.asarray(initial_geometry['flux']), np.asarray(final_geometry['flux']))

    summary = dict(config=config, input=source, target=target_info,
                   curvature_initialization=initialization, before=metrics0, after=metrics1,
                   mesh_after=mesh, initial_loss=history[0]['loss'], final_loss=float(result.best_fun),
                   optimizer=dict(message=result.message, iterations=int(result.nit),
                                  evaluations=int(result.nfev), success=bool(result.success),
                                  best_iteration=int(result.best_iteration),
                                  invalid_trials=int(result.invalid_trials)),
                   history=history, elapsed_seconds=time.perf_counter()-started,
                   devices=str(jax.devices()), float64=jax.config.x64_enabled)
    (output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    logging.info('Complete: %s', output)


if __name__ == '__main__':
    main()
