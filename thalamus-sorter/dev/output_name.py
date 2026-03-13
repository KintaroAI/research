"""Auto-generate output directory paths following the convention:
    ~/data/research/thalamus-sorter/output_{experiment}_{run:03d}_{short_desc}

Usage:
    from output_name import next_output
    path = next_output(13, "rgb_garden_50k_d16")
    # -> "/home/user/data/research/thalamus-sorter/output_13_004_rgb_garden_50k_d16"

CLI:
    python output_name.py 13 rgb_garden_50k_d16
"""

import os
import re
import sys

DATA_DIR = os.path.expanduser("~/data/research/thalamus-sorter")


def next_output(experiment, desc, base_dir=None):
    """Find next available run number for this experiment and return full path."""
    if base_dir is None:
        base_dir = DATA_DIR
    os.makedirs(base_dir, exist_ok=True)
    pattern = re.compile(rf"^output_{experiment}_(\d{{3}})_")
    max_run = 0
    for entry in os.listdir(base_dir):
        m = pattern.match(entry)
        if m:
            max_run = max(max_run, int(m.group(1)))
    run = max_run + 1
    return os.path.join(base_dir, f"output_{experiment}_{run:03d}_{desc}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python output_name.py <experiment> <short_desc>")
        sys.exit(1)
    print(next_output(int(sys.argv[1]), sys.argv[2]))
