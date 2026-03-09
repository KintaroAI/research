#!/usr/bin/env python3
"""Compose a grid of continuous drift experiment results into a single image.

Top-left cell: original image. Top row: Dims labels. Left column: K labels.
Each cell: the corresponding experiment frame at that tick.
Produces 1000 composite frames (one per saved frame across all experiments).
"""

import os
import cv2
import numpy as np

CELL_SIZE = 80
LABEL_W = CELL_SIZE
LABEL_H = CELL_SIZE
PADDING = 2

k_values = [1, 11, 21, 31, 41, 51]
dims_values = [2, 3, 10, 50, 100]

n_rows = len(k_values)
n_cols = len(dims_values)

img_w = LABEL_W + n_cols * (CELL_SIZE + PADDING) - PADDING
img_h = LABEL_H + n_rows * (CELL_SIZE + PADDING) - PADDING

original = cv2.imread("K_80_g.png", cv2.IMREAD_GRAYSCALE)

os.makedirs("output_2_grid", exist_ok=True)

n_frames = 1000

for frame_idx in range(n_frames):
    canvas = np.ones((img_h, img_w), dtype=np.uint8) * 255

    # Top-left: original image
    canvas[0:CELL_SIZE, 0:CELL_SIZE] = original

    # Top row: Dims labels
    for ci, d in enumerate(dims_values):
        x = LABEL_W + ci * (CELL_SIZE + PADDING)
        label = f"D={d}"
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = x + (CELL_SIZE - ts[0]) // 2
        ty = (CELL_SIZE + ts[1]) // 2
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)

    # Left column: K labels
    for ri, k in enumerate(k_values):
        y = LABEL_H + ri * (CELL_SIZE + PADDING)
        label = f"K={k}"
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = (LABEL_W - ts[0]) // 2
        ty = y + (CELL_SIZE + ts[1]) // 2
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)

    # Fill cells
    for ri, k in enumerate(k_values):
        for ci, d in enumerate(dims_values):
            folder = f"output_2_{k}_{d}"
            path = os.path.join(folder, f"frame_{frame_idx:06d}.png")
            if os.path.exists(path):
                cell = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if cell is not None:
                    cell = cv2.resize(cell, (CELL_SIZE, CELL_SIZE))
                    x = LABEL_W + ci * (CELL_SIZE + PADDING)
                    y = LABEL_H + ri * (CELL_SIZE + PADDING)
                    canvas[y:y+CELL_SIZE, x:x+CELL_SIZE] = cell

    out_path = os.path.join("output_2_grid", f"grid_{frame_idx:06d}.png")
    cv2.imwrite(out_path, canvas)

    if frame_idx == 0:
        print(f"First frame saved: {out_path} ({img_w}x{img_h})")
    if (frame_idx + 1) % 100 == 0:
        print(f"  {frame_idx + 1}/{n_frames} frames")

print("Done.")
