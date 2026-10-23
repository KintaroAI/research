"""Run a queue of training jobs, at most `--slots` at a time (default 2 x 64 threads on this machine).

Queue file: one job per line, `<run_name> <train.py args...>`; '#' starts a comment. Output dirs are
assigned up front (exp_NNNNN/RRR_<run_name>) so concurrent jobs never race for a run number.

    ./venv/bin/python run_queue.py queue.txt --exp 1 --slots 2 --threads 64
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

from output_name import DATA_DIR

HERE = os.path.dirname(os.path.abspath(__file__))


def external_jobs(running):
    """Training processes not started by this queue (they occupy slots too)."""
    try:
        out = subprocess.run(["pgrep", "-f", "[t]rain.py --config"], capture_output=True, text=True).stdout.split()
    except OSError:
        return 0
    mine = {str(p.pid) for p, _, _ in running}
    return len([pid for pid in out if pid not in mine])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--exp", type=int, default=1)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--threads", type=int, default=64)
    args = ap.parse_args()
    jobs = []
    with open(args.queue) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                name, rest = line.split(None, 1)
                jobs.append((name, rest))
    exp_dir = os.path.join(DATA_DIR, "exp_%05d" % args.exp)
    os.makedirs(exp_dir, exist_ok=True)
    existing = [int(e[:3]) for e in os.listdir(exp_dir) if e[:3].isdigit()]
    nxt = max(existing, default=0) + 1
    planned = []
    for name, rest in jobs:
        out = os.path.join(exp_dir, "%03d_%s" % (nxt, name))
        nxt += 1
        planned.append((name, out, rest))
    running = []
    log = open(os.path.join(exp_dir, "queue_%s.log" % time.strftime("%Y%m%d_%H%M%S")), "w")
    while planned or running:
        running = [(p, n, o) for p, n, o in running if p.poll() is None]
        while planned and len(running) + external_jobs(running) < args.slots:
            name, out, rest = planned.pop(0)
            os.makedirs(out, exist_ok=True)
            cmd = [os.path.join(HERE, "venv", "bin", "python"), os.path.join(HERE, "train.py")] + shlex.split(rest)
            cmd += ["--set", "train.threads=%d" % args.threads, "-o", out]
            fo = open(os.path.join(out, "stdout.log"), "w")
            p = subprocess.Popen(cmd, cwd=HERE, stdout=fo, stderr=subprocess.STDOUT)
            running.append((p, name, out))
            msg = "%s start %s pid=%d: %s" % (time.strftime("%H:%M:%S"), out, p.pid, " ".join(cmd))
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        time.sleep(5)
    log.write("%s queue done\n" % time.strftime("%H:%M:%S"))
    log.close()
    print("queue done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
