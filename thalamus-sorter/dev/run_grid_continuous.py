#!/usr/bin/env python3
"""Launch a grid of continuous drift experiments in parallel.

Sweeps learning rate × K, matching experiment 00001's grid search structure.
"""

import subprocess
import sys
import os
import time

PYTHON = "venv/bin/python"
IMAGE = "K_80_g.png"
SIZE = 80
FRAMES = 100000
SAVE_EVERY = 100  # 100k / 100 = 1000 images

learning_rates = [0.01, 0.05, 0.1, 0.2]
k_values = [1, 11, 21, 31, 41, 51]

procs = []
for lr in learning_rates:
    for k in k_values:
        out_dir = f"output_cont_{lr}_{k}"
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            PYTHON, "main.py", "continuous",
            "--image", IMAGE,
            "-W", str(SIZE), "-H", str(SIZE),
            "--k", str(k),
            "--lr", str(lr),
            "--frames", str(FRAMES),
            "--save-every", str(SAVE_EVERY),
            "-o", out_dir,
            "--gpu",
        ]
        print(f"Launching: lr={lr}, k={k} -> {out_dir}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((lr, k, out_dir, p))

print(f"\n{len(procs)} jobs launched. Waiting...")
t0 = time.time()

for lr, k, out_dir, p in procs:
    stdout, stderr = p.communicate()
    status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
    out_text = stdout.decode().strip().split('\n')[-1] if stdout else ""
    print(f"  lr={lr}, k={k}: {status}  {out_text}")
    if p.returncode != 0:
        print(f"    stderr: {stderr.decode()[:200]}")

elapsed = time.time() - t0
print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
