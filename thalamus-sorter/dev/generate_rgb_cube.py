"""Generate an RGB cube: x→red, y→green, z→blue.
Every cell is filled. Color directly encodes 3D position.

Saved as (size, size, size, 3) uint8 .npy file.
"""

import numpy as np


def make_rgb_cube(size=80):
    vol = np.zeros((size, size, size, 3), dtype=np.uint8)
    for d in range(3):
        sl = [None, None, None]
        sl[d] = slice(None)
        ramp = np.linspace(0, 255, size).astype(np.uint8)
        # Broadcast ramp along axis d
        shape = [1, 1, 1]
        shape[d] = size
        vol[:, :, :, d] = ramp.reshape(shape)
    return vol


if __name__ == "__main__":
    vol = make_rgb_cube(80)
    np.save("rgb_cube_80.npy", vol)
    print(f"RGB cube: {vol.shape}, dtype={vol.dtype}")
    print(f"Corner (0,0,0): {vol[0,0,0]} (black)")
    print(f"Corner (79,0,0): {vol[79,0,0]} (red)")
    print(f"Corner (0,79,0): {vol[0,79,0]} (green)")
    print(f"Corner (0,0,79): {vol[0,0,79]} (blue)")
    print(f"Corner (79,79,79): {vol[79,79,79]} (white)")
