#!/usr/bin/env python3
"""Grid search: K × dimensionality for continuous drift experiment."""

import subprocess
import os
import time

PYTHON = "venv/bin/python"
IMAGE = "K_80_g.png"
SIZE = 80
FRAMES = 100000
SAVE_EVERY = 100  # 100k / 100 = 1000 images
LR = 0.05

k_values = [1, 11, 21, 31, 41, 51]
dims_values = [2, 3, 10, 50, 100]

procs = []
for k in k_values:
    for d in dims_values:
        out_dir = f"output_2_{k}_{d}"
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            PYTHON, "main.py", "continuous",
            "--image", IMAGE,
            "-W", str(SIZE), "-H", str(SIZE),
            "--k", str(k),
            "--lr", str(LR),
            "--dims", str(d),
            "--frames", str(FRAMES),
            "--save-every", str(SAVE_EVERY),
            "-o", out_dir,
            "--gpu",
        ]
        print(f"Launching: k={k}, dims={d} -> {out_dir}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((k, d, out_dir, p))

print(f"\n{len(procs)} jobs launched. Waiting...")
t0 = time.time()

for k, d, out_dir, p in procs:
    stdout, stderr = p.communicate()
    status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
    out_text = stdout.decode().strip().split('\n')[-1] if stdout else ""
    print(f"  k={k:2d}, dims={d:3d}: {status}  {out_text}")
    if p.returncode != 0:
        print(f"    stderr: {stderr.decode()[:200]}")

elapsed = time.time() - t0
print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
