"""Grid search for correlation-based neighbor sampling parameters.

Sweep 1: sigma × threshold (k_sample=200, T=200 fixed)
Sweep 2: k_sample × signal_T (sigma=8, threshold=0.3 fixed)

Usage:
    python run_correlation_grid.py -i K_80_g.png
    python run_correlation_grid.py -i K_80_g.png --sweep 1   # sigma × threshold only
    python run_correlation_grid.py -i K_80_g.png --sweep 2   # k_sample × T only
"""

import subprocess
import sys
import re
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_one(params, image, output_dir):
    """Run a single correlation experiment and extract final disparity."""
    cmd = [
        sys.executable, "main.py", "word2vec",
        "--mode", "correlation",
        "-W", "80", "-H", "80",
        "--dims", "8",
        "--k-neg", "5",
        "--lr", "0.001",
        "--normalize-every", "100",
        "--k-sample", str(params["k_sample"]),
        "--threshold", str(params["threshold"]),
        "--signal-T", str(params["signal_T"]),
        "--signal-sigma", str(params["sigma"]),
        "-i", image,
        "-f", "5000",
        "--save-every", "50",
        "-o", output_dir,
        "--render", "umap",
        "--align",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    # Extract final disparity from output
    output = result.stdout + result.stderr
    disparities = re.findall(r"disparity=([0-9.]+)", output)
    final_disp = float(disparities[-1]) if disparities else 999.0

    # Extract total pairs
    pairs_match = re.search(r"total_pairs=(\d+)", output)
    total_pairs = int(pairs_match.group(1)) if pairs_match else 0

    return {
        "disparity": final_disp,
        "pairs": total_pairs,
        "elapsed": elapsed,
        **params,
    }


def _run_job(job):
    """Wrapper for parallel execution."""
    params, image, output_dir, label, sweep = job
    r = run_one(params, image, output_dir)
    r["label"] = label
    r["sweep"] = sweep
    print(f"  done: {label:<20s} disp={r['disparity']:.4f} "
          f"pairs={r['pairs']//1_000_000}M time={r['elapsed']:.0f}s",
          flush=True)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="K_80_g.png")
    parser.add_argument("--sweep", type=int, default=0,
                        help="0=both, 1=sigma×threshold, 2=k_sample×T")
    parser.add_argument("-o", "--output-prefix", default="output_9_grid")
    parser.add_argument("-j", "--jobs", type=int, default=6,
                        help="Max parallel jobs (default: 6)")
    args = parser.parse_args()

    jobs = []

    # Sweep 1: sigma × threshold
    if args.sweep in (0, 1):
        sigmas = [5, 8, 12]
        thresholds = [0.15, 0.3, 0.5]
        for sigma in sigmas:
            for thresh in thresholds:
                label = f"s{sigma}_t{thresh}"
                out_dir = f"{args.output_prefix}_s1_{label}"
                params = {
                    "sigma": sigma, "threshold": thresh,
                    "k_sample": 200, "signal_T": 200,
                }
                jobs.append((params, args.image, out_dir, label, 1))

    # Sweep 2: k_sample × signal_T
    if args.sweep in (0, 2):
        k_samples = [100, 200, 400]
        signal_Ts = [50, 100, 200]
        for ks in k_samples:
            for st in signal_Ts:
                label = f"k{ks}_T{st}"
                out_dir = f"{args.output_prefix}_s2_{label}"
                params = {
                    "sigma": 8, "threshold": 0.3,
                    "k_sample": ks, "signal_T": st,
                }
                jobs.append((params, args.image, out_dir, label, 2))

    print(f"Running {len(jobs)} experiments, {args.jobs} in parallel")
    t0 = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.time() - t0
    print(f"\nAll done in {total_time:.0f}s")

    # Print sweep 1 table
    s1 = [r for r in results if r["sweep"] == 1]
    if s1:
        sigmas = [5, 8, 12]
        thresholds = [0.15, 0.3, 0.5]
        print("\n" + "=" * 60)
        print("SWEEP 1: sigma × threshold → disparity  (k_sample=200, T=200)")
        print(f"{'':>12s}", end="")
        for t in thresholds:
            print(f"  thresh={t:<6}", end="")
        print()
        for sigma in sigmas:
            print(f"  sigma={sigma:<4}", end="")
            for thresh in thresholds:
                r = next((x for x in s1
                          if x["sigma"] == sigma
                          and x["threshold"] == thresh), None)
                if r:
                    print(f"  {r['disparity']:>10.4f}", end="")
                else:
                    print(f"  {'---':>10s}", end="")
            print()

    # Print sweep 2 table
    s2 = [r for r in results if r["sweep"] == 2]
    if s2:
        k_samples = [100, 200, 400]
        signal_Ts = [50, 100, 200]
        print("\n" + "=" * 60)
        print("SWEEP 2: k_sample × signal_T → disparity  (sigma=8, threshold=0.3)")
        print(f"{'':>14s}", end="")
        for st in signal_Ts:
            print(f"  T={st:<8}", end="")
        print()
        for ks in k_samples:
            print(f"  k_sample={ks:<4}", end="")
            for st in signal_Ts:
                r = next((x for x in s2
                          if x["k_sample"] == ks
                          and x["signal_T"] == st), None)
                if r:
                    print(f"  {r['disparity']:>10.4f}", end="")
                else:
                    print(f"  {'---':>10s}", end="")
            print()

    # Full summary
    print("\n" + "=" * 60)
    print("ALL RESULTS")
    print(f"{'Label':<20s} {'Disp':>8s} {'Pairs':>8s} {'Time':>6s}")
    for r in sorted(results, key=lambda x: x["disparity"]):
        print(f"{r['label']:<20s} {r['disparity']:8.4f} "
              f"{r['pairs']//1_000_000:>6d}M {r['elapsed']:>5.0f}s")


if __name__ == "__main__":
    main()
