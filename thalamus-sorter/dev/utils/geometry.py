"""Coordinate conversion and spatial helpers."""

import math


def coords(i, width):
    """Linear index to (x, y) grid coordinates."""
    return (i % width, i // width)


def to_index(x, y, width):
    """(x, y) grid coordinates to linear index."""
    return y * width + x


def move_closer(ix, iy, jx, jy):
    """Move two points one step closer to each other along their connecting line.
    Returns (new_ix, new_iy, new_jx, new_jy)."""
    d = math.sqrt((ix - jx) ** 2 + (iy - jy) ** 2)
    if d == 0:
        return ix, iy, jx, jy
    ux = (jx - ix) / d
    uy = (jy - iy) / d
    return (round(ix + ux), round(iy + uy),
            round(jx - ux), round(jy - uy))
