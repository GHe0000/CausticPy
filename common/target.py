"""Read linear gray samples and optionally center them by an integer shift."""

import numpy as np
from PIL import Image


def load_gray(path):
    with Image.open(path) as image:
        rgba = np.asarray(image.convert('RGBA'), dtype=np.float64) / 255
    gray = (.2126 * rgba[:, :, 0] + .7152 * rgba[:, :, 1] + .0722 * rgba[:, :, 2]) * rgba[:, :, 3]
    if gray.shape[0] != gray.shape[1] or gray.shape[0] < 2:
        raise ValueError(f'Target must be a square image of at least 2x2, got {gray.shape}')
    if not np.isfinite(gray).all() or gray.sum() <= 0:
        raise ValueError('Target is empty or invalid')
    return gray


def center_pattern(gray):
    yy, xx = np.nonzero(gray > 0)
    rows, cols = gray.shape
    dy = int(np.rint((rows - 1 - yy.min() - yy.max()) / 2))
    dx = int(np.rint((cols - 1 - xx.min() - xx.max()) / 2))
    shifted = np.zeros_like(gray)
    sy = slice(max(0, -dy), min(rows, rows - dy))
    sx = slice(max(0, -dx), min(cols, cols - dx))
    shifted[sy.start + dy:sy.stop + dy, sx.start + dx:sx.stop + dx] = gray[sy, sx]
    if np.count_nonzero(shifted) != np.count_nonzero(gray):
        raise ValueError('Centering clipped nonzero target pixels')
    return shifted, (dx, dy)
