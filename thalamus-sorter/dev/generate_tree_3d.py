"""Generate an 80x80x80 3D volume containing a simple tree shape.
Saved as .npy file for reuse. Values: 0.0 (empty) to 1.0 (solid).

Tree structure (y is up):
  - Trunk: brown cylinder at center, y=0..30
  - Canopy: 3 stacked cones getting smaller, y=20..70
  - Star: small bright spot at the top
"""

import numpy as np


def make_tree(size=80):
    vol = np.zeros((size, size, size), dtype=np.float32)
    cx, cz = size // 2, size // 2  # center x, z

    # Coordinate grids
    x = np.arange(size)
    y = np.arange(size)
    z = np.arange(size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Distance from vertical axis (x-z plane)
    r = np.sqrt((X - cx) ** 2 + (Z - cz) ** 2)

    # --- Trunk: cylinder, y=0..30, radius=3 ---
    trunk_mask = (r <= 3) & (Y >= 0) & (Y <= 30)
    vol[trunk_mask] = 0.45  # brown-ish

    # --- Canopy: 3 cone layers ---
    # Bottom cone: y=18..42, radius 28 at bottom to 4 at top
    for yi in range(18, 42):
        frac = (yi - 18) / (42 - 18)
        cone_r = 28 * (1 - frac) + 4 * frac
        mask = (r <= cone_r) & (Y == yi)
        vol[mask] = 0.7 + 0.1 * (1 - frac)

    # Middle cone: y=30..55, radius 22 at bottom to 3 at top
    for yi in range(30, 55):
        frac = (yi - 30) / (55 - 30)
        cone_r = 22 * (1 - frac) + 3 * frac
        mask = (r <= cone_r) & (Y == yi)
        vol[mask] = 0.75 + 0.1 * (1 - frac)

    # Top cone: y=42..68, radius 16 at bottom to 2 at top
    for yi in range(42, 68):
        frac = (yi - 42) / (68 - 42)
        cone_r = 16 * (1 - frac) + 2 * frac
        mask = (r <= cone_r) & (Y == yi)
        vol[mask] = 0.8 + 0.15 * (1 - frac)

    # --- Star at top ---
    star_mask = (r <= 3) & (Y >= 67) & (Y <= 72)
    vol[star_mask] = 1.0

    return vol


if __name__ == "__main__":
    vol = make_tree(80)
    np.save("tree_80.npy", vol)
    n_filled = (vol > 0).sum()
    n_total = vol.size
    print(f"Tree volume: {vol.shape}, filled: {n_filled}/{n_total} ({100*n_filled/n_total:.1f}%)")
    print(f"Value range: [{vol.min():.2f}, {vol.max():.2f}]")
