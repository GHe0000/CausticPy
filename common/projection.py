"""Forward geometry for rays travelling along +z from a flat entrance."""

import numpy as np


def refract(normals, n_lens, n_air):
    """Return exit directions, Snell discriminants, and exit Fresnel transmission."""
    eta = n_lens / n_air
    cos_i = normals[:, 2]
    discriminant = 1 - eta**2 * (1 - cos_i**2)
    cos_t = np.sqrt(np.maximum(discriminant, 0))
    direction = eta * np.array([0., 0., 1.]) + (cos_t - eta * cos_i)[:, None] * normals
    rs = (n_lens * cos_i - n_air * cos_t) / (n_lens * cos_i + n_air * cos_t)
    rp = (n_air * cos_i - n_lens * cos_t) / (n_air * cos_i + n_lens * cos_t)
    return direction, discriminant, 1 - .5 * (rs**2 + rp**2)


def screen_hits(xy, z, directions, screen_z, *, min_dz=None):
    dz = directions[:, 2, None]
    if min_dz is not None:
        dz = np.maximum(dz, min_dz)
    return xy + (screen_z - z[:, None]) * directions[:, :2] / dz


def screen_pixels(hits, width, height, shape):
    """Convert centered screen coordinates in mm to PNG row/column indices."""
    rows, cols = shape
    col = np.floor((hits[:, 0] / width + .5) * cols).astype(int)
    row = np.floor((.5 - hits[:, 1] / height) * rows).astype(int)
    return row, col
