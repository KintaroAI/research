#!/usr/bin/env python3
"""W&B wrapper for CUDA training binary.

Launches ./train (or any command) via subprocess, parses structured stdout
line-by-line, and logs metrics to Weights & Biases in real-time.

Usage:
    python wandb_train.py [wandb options] -- ./train [train flags...]

Examples:
    python wandb_train.py -- ./train
    python wandb_train.py --project gpt2-cuda --name "banded-256" -- ./train -1 256 -2 256
    python wandb_train.py --tags banded,fc1 --group sparsity -- ./train -1 256

If wandb is not installed, prints a warning and runs the command without logging.
"""

import argparse
import re
import subprocess
import sys

# Graceful degradation if wandb not installed
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# --- Regexes matching CUDA training output ---

# Config table row: "| parameter             | value                     |"
CONFIG_RE = re.compile(r'^\|\s*(.+?)\s*\|\s*(.+?)\s*\|$')

# Train step: "step    4/1000: train loss 5.123456 (12.34 ms, 84000 tok/s)"
TRAIN_RE = re.compile(
    r'step\s+(\d+)/(\d+):\s+train loss\s+([\d.]+)\s+\(([\d.]+) ms,\s+(\d+) tok/s\)'
)

# Validation: "val loss 4.567890 (avg_train 5.123456, gap +0.456789)"
VAL_RE = re.compile(
    r'val loss\s+([\d.]+)\s+\(avg_train\s+([\d.]+),\s+gap\s+([+-]?[\d.]+)\)'
)

# Test: "test loss 4.321000"
TEST_RE = re.compile(r'test loss\s+([\d.]+)')

# Config table separator lines
SEPARATOR_RE = re.compile(r'^\+[-+]+\+$')


def parse_config_value(value_str):
    """Try to parse a config value as int, float, or leave as string."""
    s = value_str.strip()
    if s == 'NULL':
        return None
    # Try int
    try:
        return int(s)
    except ValueError:
        pass
    # Try float (handles scientific notation like 1.000000e-04)
    try:
        return float(s)
    except ValueError:
        pass
    return s


def config_key(raw_key):
    """Normalize config key: lowercase, replace spaces with underscores."""
    return raw_key.strip().lower().replace(' ', '_')


def parse_args():
    """Parse wandb options before '--', leave the rest as the train command."""
    # Split on '--'
    argv = sys.argv[1:]
    if '--' in argv:
        split_idx = argv.index('--')
        wandb_args = argv[:split_idx]
        train_cmd = argv[split_idx + 1:]
    else:
        wandb_args = []
        train_cmd = argv

    if not train_cmd:
        print("Error: no training command provided", file=sys.stderr)
        print("Usage: python wandb_train.py [wandb options] -- ./train [flags...]", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description='W&B wrapper for training')
    parser.add_argument('--project', default='gpt2-cuda', help='W&B project name')
    parser.add_argument('--name', default=None, help='W&B run name')
    parser.add_argument('--group', default=None, help='W&B run group')
    parser.add_argument('--tags', default=None, help='Comma-separated tags')
    parser.add_argument('--entity', default=None, help='W&B entity (team/user)')
    args = parser.parse_args(wandb_args)

    if args.tags:
        args.tags = [t.strip() for t in args.tags.split(',')]

    return args, train_cmd


def run(wandb_args, train_cmd):
    config = {}
    in_config_table = False
    current_step = None

    # Initialize wandb
    use_wandb = HAS_WANDB
    if use_wandb:
        # Patch email in API-fetched user settings to avoid leaking personal
        # email in public run metadata. wandb fetches email from the API during
        # init and writes it to wandb-metadata.json; the env var and Settings
        # kwarg are both overwritten by this API response. Monkey-patching
        # _load_user_settings is the only reliable intercept point.
        PUBLIC_EMAIL = 'kin@kintaroai.com'
        _wl = wandb.setup()
        _orig_load = _wl._load_user_settings
        def _patched_load(_orig=_orig_load):
            result = _orig()
            if result and 'email' in result:
                result['email'] = PUBLIC_EMAIL
            return result
        _wl._load_user_settings = _patched_load

        wandb.init(
            project=wandb_args.project,
            name=wandb_args.name,
            group=wandb_args.group,
            tags=wandb_args.tags,
            entity=wandb_args.entity,
            config={'command': ' '.join(train_cmd)},
            settings=wandb.Settings(disable_git=True),
            save_code=False,
        )
    else:
        print("WARNING: wandb not installed. Running without logging.", file=sys.stderr)
        print("Install with: pip install wandb", file=sys.stderr)

    # Launch training subprocess
    proc = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )

    try:
        for line in proc.stdout:
            # Pass through to terminal
            sys.stdout.write(line)
            sys.stdout.flush()

            stripped = line.rstrip()

            # Config table detection
            if SEPARATOR_RE.match(stripped):
                in_config_table = True
                continue

            # Config table rows
            if in_config_table:
                m = CONFIG_RE.match(stripped)
                if m:
                    key = config_key(m.group(1))
                    val = parse_config_value(m.group(2))
                    if key not in ('parameter', 'value') and val is not None:
                        config[key] = val
                    continue
                else:
                    # Non-table line after table started — table is over,
                    # upload collected config
                    if config and use_wandb:
                        wandb.config.update(config)
                    in_config_table = False

            # Train step
            m = TRAIN_RE.search(stripped)
            if m:
                step = int(m.group(1))
                current_step = step
                if use_wandb:
                    wandb.log({
                        'train_loss': float(m.group(3)),
                        'step_time_ms': float(m.group(4)),
                        'tok_per_sec': int(m.group(5)),
                    }, step=step)
                continue

            # Validation
            m = VAL_RE.search(stripped)
            if m:
                if use_wandb:
                    log_step = current_step if current_step is not None else 0
                    wandb.log({
                        'val_loss': float(m.group(1)),
                        'avg_train_loss': float(m.group(2)),
                        'generalization_gap': float(m.group(3)),
                    }, step=log_step)
                continue

            # Test loss
            m = TEST_RE.search(stripped)
            if m:
                if use_wandb:
                    log_step = current_step if current_step is not None else 0
                    wandb.log({
                        'test_loss': float(m.group(1)),
                    }, step=log_step)
                continue

    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise
    finally:
        proc.wait()
        if use_wandb:
            wandb.finish()

    return proc.returncode


def main():
    wandb_args, train_cmd = parse_args()
    rc = run(wandb_args, train_cmd)
    sys.exit(rc)


if __name__ == '__main__':
    main()
