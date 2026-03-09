#!/usr/bin/env python3
"""Launch a grid of greedy drift experiments in parallel."""

import subprocess
import sys
import os
import time

PYTHON = "venv/bin/python"
IMAGE = "K_80_g.png"
SIZE = 80
FRAMES = 100000
SAVE_EVERY = 100  # 100k / 100 = 1000 images

move_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
k_values = [1, 11, 21, 31, 41, 51]

procs = []
for mf in move_fractions:
    for k in k_values:
        out_dir = f"output_80_{mf}_{k}"
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            PYTHON, "main.py", "greedy",
            "--image", IMAGE,
            "-W", str(SIZE), "-H", str(SIZE),
            "--k", str(k),
            "--move-fraction", str(mf),
            "--frames", str(FRAMES),
            "--save-every", str(SAVE_EVERY),
            "-o", out_dir,
            "--gpu",
        ]
        print(f"Launching: mf={mf}, k={k} -> {out_dir}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((mf, k, out_dir, p))

print(f"\n{len(procs)} jobs launched. Waiting...")
t0 = time.time()

for mf, k, out_dir, p in procs:
    stdout, stderr = p.communicate()
    status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
    print(f"  mf={mf}, k={k}: {status}")
    if p.returncode != 0:
        print(f"    stderr: {stderr.decode()[:200]}")

elapsed = time.time() - t0
print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
