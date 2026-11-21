"""DP loss and surface regularizers."""

import jax.numpy as jnp


def directed_residual(g, matrices, i, j):
    basis = g['basis'][i]
    d = jnp.einsum('fki,fk->fi', basis, g['centroid'][j] - g['centroid'][i])
    dn = jnp.einsum('fki,fk->fi', basis, g['normal'][j] - g['normal'][i])
    error = jnp.einsum('fij,fj->fi', matrices[i], d) - dn
    return jnp.sum(error**2, axis=1) / jnp.sum(d**2, axis=1)


def regularizers(g, weingarten, topo, nu, eps1, eps2):
    a, b, c = weingarten.T
    matrices = jnp.stack([a, c, c, b], axis=1).reshape(-1, 2, 2)
    i, j = topo['adjacent'].T
    h = directed_residual(g, matrices, i, j) + directed_residual(g, matrices, j, i)
    edge = jnp.sum(-jnp.expm1(-h / (2 * nu**2)))
    face = jnp.sum(((a + b) / 2)**2 * g['surface_area'])
    xy = g['vertices'][:, :2]
    average = jnp.zeros_like(xy).at[topo['src']].add(xy[topo['dst']]) / topo['degree'][:, None]
    ids = topo['interior']
    lap = jnp.sum((xy[ids] - average[ids])**2)
    area = g['area']
    denom = jnp.where(area > eps1, area - eps1, 1.)
    barr_a = jnp.sum(jnp.where(area >= eps2, 0., jnp.where(area > eps1, ((eps2-eps1)/denom-1)**2, jnp.inf)))
    barr_t = jnp.sum(jnp.where(g['q'] > 0, 1 / g['q'], jnp.inf))
    return dict(face=face, edge=edge, lap=lap, smooth=face+3*edge+.2*lap,
                barrier=barr_a+barr_t, residual=h)


def image_loss(geometry, curvature, topology, nu, eps1, eps2, target, gamma, raster, pairs, weights):
    target_total = jnp.sum(target**gamma)
    flux = raster(geometry['hit'], geometry['flux'], *pairs)
    image = (target_total * flux)**(1 / gamma)
    image_error = jnp.sum((image - target)**2)

    screen_size = jnp.array([target.shape[1], target.shape[0]])
    outside = geometry['hit'] - jnp.clip(geometry['hit'], 0., screen_size)
    boundary_error = jnp.sum(outside**2)
    shape = regularizers(geometry, curvature, topology, nu, eps1, eps2)
    terms = dict(image=image_error, boundary=boundary_error, face=shape['face'],
                 edge=shape['edge'], laplacian=shape['lap'], barrier=shape['barrier'])
    weighted = dict(image=weights['image'] * terms['image'],
                    boundary=weights['boundary'] * terms['boundary'],
                    face=weights['smooth'] * terms['face'],
                    edge=3 * weights['smooth'] * terms['edge'],
                    laplacian=.2 * weights['smooth'] * terms['laplacian'],
                    barrier=weights['barrier'] * terms['barrier'])
    total = (weighted['image'] + weighted['boundary']
             + weights['smooth'] * shape['smooth'] + weighted['barrier'])
    saturation = jnp.mean(jnp.exp(-shape['residual'] / (2 * nu**2)) < 1e-6)
    return total, dict(raw=terms, weighted=weighted, welsch_saturated_fraction=saturation)
