"""Auto-generate output directory paths following the convention:
    ~/data/research/visual-motor-agent/exp_NNNNN/{run:03d}_{short_desc}

Usage:
    from output_name import next_output
    path = next_output(1, "A_baseline_s1")
    # -> "/home/user/data/research/visual-motor-agent/exp_00001/004_A_baseline_s1"

CLI:
    python output_name.py 1 A_baseline_s1
"""

import os
import re
import sys

DATA_DIR = os.path.expanduser("~/data/research/visual-motor-agent")


def next_output(experiment, desc, base_dir=None):
    """Find next available run number for this experiment and return full path."""
    if base_dir is None:
        base_dir = DATA_DIR
    exp_dir = os.path.join(base_dir, f"exp_{experiment:05d}")
    os.makedirs(exp_dir, exist_ok=True)
    pattern = re.compile(r"^(\d{3})_")
    max_run = 0
    for entry in os.listdir(exp_dir):
        m = pattern.match(entry)
        if m:
            max_run = max(max_run, int(m.group(1)))
    run = max_run + 1
    return os.path.join(exp_dir, f"{run:03d}_{desc}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python output_name.py <experiment> <short_desc>")
        sys.exit(1)
    print(next_output(int(sys.argv[1]), sys.argv[2]))
