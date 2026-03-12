#!/usr/bin/env python3
"""Compose correlation grid search results into two grid images.

Sweep 1: sigma (rows) × threshold (cols) — 3×3
Sweep 2: k_sample (rows) × signal_T (cols) — 3×3

Produces 101 composite frames per sweep (one per saved frame).
"""

import os
import cv2
import numpy as np

CELL_SIZE = 80
LABEL_W = 100
LABEL_H = 30
PADDING = 2

# Disparity results for annotation
DISP = {
    # sweep 1
    ("s1", 5, 0.15): 0.2671, ("s1", 5, 0.3): 0.1174, ("s1", 5, 0.5): 0.0568,
    ("s1", 8, 0.15): 0.1949, ("s1", 8, 0.3): 0.0437, ("s1", 8, 0.5): 0.0225,
    ("s1", 12, 0.15): 0.1682, ("s1", 12, 0.3): 0.1486, ("s1", 12, 0.5): 0.0926,
    # sweep 2
    ("s2", 100, 50): 0.6063, ("s2", 100, 100): 0.1110, ("s2", 100, 200): 0.1628,
    ("s2", 200, 50): 0.2764, ("s2", 200, 100): 0.0404, ("s2", 200, 200): 0.1185,
    ("s2", 400, 50): 0.4747, ("s2", 400, 100): 0.0670, ("s2", 400, 200): 0.0768,
}


def put_text_centered(canvas, text, x, y, w, h, scale=0.4):
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    tx = x + (w - ts[0]) // 2
    ty = y + (h + ts[1]) // 2
    cv2.putText(canvas, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 1, cv2.LINE_AA)


def build_grid(sweep, row_param, row_values, col_param, col_values,
               dir_fmt, title, out_dir):
    n_rows = len(row_values)
    n_cols = len(col_values)
    n_frames = 101

    # Canvas size: title row + label row + data rows; label col + data cols
    title_h = 30
    img_w = LABEL_W + n_cols * (CELL_SIZE + PADDING) - PADDING
    img_h = title_h + LABEL_H + n_rows * (CELL_SIZE + PADDING) - PADDING

    # Add disparity label row at bottom
    disp_h = 20
    img_h += disp_h

    os.makedirs(out_dir, exist_ok=True)

    for frame_idx in range(n_frames):
        canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

        # Title
        put_text_centered(canvas, title, 0, 0, img_w, title_h, scale=0.45)

        # Column labels
        for ci, cv_val in enumerate(col_values):
            x = LABEL_W + ci * (CELL_SIZE + PADDING)
            label = f"{col_param}={cv_val}"
            put_text_centered(canvas, label, x, title_h, CELL_SIZE, LABEL_H, scale=0.35)

        # Row labels + cells
        for ri, rv in enumerate(row_values):
            y = title_h + LABEL_H + ri * (CELL_SIZE + PADDING)
            label = f"{row_param}={rv}"
            put_text_centered(canvas, label, 0, y, LABEL_W, CELL_SIZE, scale=0.35)

            for ci, cv_val in enumerate(col_values):
                x = LABEL_W + ci * (CELL_SIZE + PADDING)
                folder = dir_fmt(rv, cv_val)
                path = os.path.join(folder, f"frame_{frame_idx:06d}.png")

                if os.path.exists(path):
                    cell = cv2.imread(path)
                    if cell is not None:
                        cell = cv2.resize(cell, (CELL_SIZE, CELL_SIZE))
                        canvas[y:y+CELL_SIZE, x:x+CELL_SIZE] = cell

                # Disparity label at bottom of cell
                disp_key = (sweep, rv, cv_val)
                if disp_key in DISP:
                    d = DISP[disp_key]
                    disp_y = title_h + LABEL_H + n_rows * (CELL_SIZE + PADDING) - PADDING
                    put_text_centered(canvas, f"{d:.4f}", x, disp_y, CELL_SIZE, disp_h, scale=0.3)

        out_path = os.path.join(out_dir, f"grid_{frame_idx:06d}.png")
        cv2.imwrite(out_path, canvas)

        if frame_idx == 0:
            print(f"  First frame: {out_path} ({img_w}x{img_h})")
        if (frame_idx + 1) % 50 == 0 or frame_idx == n_frames - 1:
            print(f"  {frame_idx + 1}/{n_frames} frames")


def main():
    print("Sweep 1: sigma x threshold")
    build_grid(
        sweep="s1",
        row_param="sigma", row_values=[5, 8, 12],
        col_param="thresh", col_values=[0.15, 0.3, 0.5],
        dir_fmt=lambda sigma, thresh: f"output_9_grid_s1_s{sigma}_t{thresh}",
        title="Sweep 1: sigma x threshold (k=200, T=200)",
        out_dir="output_9_grid_sweep1",
    )

    print("\nSweep 2: k_sample x signal_T")
    build_grid(
        sweep="s2",
        row_param="k_sample", row_values=[100, 200, 400],
        col_param="T", col_values=[50, 100, 200],
        dir_fmt=lambda ks, st: f"output_9_grid_s2_k{ks}_T{st}",
        title="Sweep 2: k_sample x T (sigma=8, thresh=0.3)",
        out_dir="output_9_grid_sweep2",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
