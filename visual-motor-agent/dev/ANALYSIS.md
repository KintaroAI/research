# Running and Analysing Universal-Model Experiments

How to run a universal-model lifetime experiment (vm-00019 on) and get its numbers, tables, videos and dream measures
again. Each experiment's own Makefile (`../NNNNN-*/Makefile`) has the exact commands and settings; this file explains
the pipeline they use. Older basic-agent tools (`evaluate.py`, `analyze_dynamics.py`, `make_video.py`) are in
`README.md`, "Analysis".

## What repeats and what does not

- **GPU lifetimes do not repeat bit for bit.** The same config and seed on the GPU give different lives (two 2,000-tick
  runs of `U_touch_ego.json`, seed 1: 12 against 20 food in the first 1k ticks). A rerun repeats an experiment's
  findings, not its per-seed numbers: compare arms over seeds (3 per arm so far), never single runs.
- **Frozen tests on a saved checkpoint do repeat exactly** (fixed world seeds, `EVAL_SEED`, and fixed sampling
  seeds): `forage_video.py` over 16 worlds reproduces `online.py`'s forage numbers to the last digit.
- **Where things are:** outputs in `~/data/research/visual-motor-agent/exp_NNNNN/` (not in git). Each run directory
  has `info.json` (the full config, the command line, the git commit, the seed), `metrics.jsonl` (one line per 1k
  ticks), `ckpt_NNNNNNNN.bin` (every `online.save_every` ticks) and `eval/` (the tests' JSON). The code of an
  experiment is its tag `exp/vm-NNNNN`.

## 1. Lifetimes

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe   # GPU 0 = 3090, 1 = 4090
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python lifetime.py --config configs/U_touch.json \
    --set online.steps=100000 --set online.save_every=25000 --seed 1 --set name=TB -o $OUT/001_TB_s1
```

- `--set name=ARM` names the arm: `tables.py` groups runs by it.
- Self-forcing (task 00039): `--set online.sf_windows=8 --set online.sf_steps=3 --set online.sf_jacobi=2`. The 4-step
  pair lookahead in life (task 00029/00038): `--set online.instinct_h=4 --set online.instinct_decode=pair`.
- Speeds (100k ticks): plain 62 ticks/s (~30 min; three at once on one GPU under MPS); self-forcing 22–30; self-forcing
  with the 4-step lookahead 13 (~2 h). A lifetime with the lookahead slows to ~1 tick/s next to frozen tests on the
  same GPU: run tests on the other GPU or after it.

## 2. Frozen forage tests (`online.py --eval`)

The checkpoint, frozen, forages in 16 fresh continual worlds × 1,000 steps (the same worlds for every run):

```bash
F="--eval-device cuda --eval-tests forage --eval-worlds 16"
./venv/bin/python online.py --eval RUN --ckpts 100000 $F --eval-instinct 0                    # the model's own moves
./venv/bin/python online.py --eval RUN --ckpts 100000 $F --eval-horizon 1                     # 1-step instinct
./venv/bin/python online.py --eval RUN --ckpts 100000 $F --eval-horizon 4 --eval-decode mean  # 4-step lookahead
./venv/bin/python online.py --eval RUN --ckpts 100000 $F --eval-horizon 4 --eval-decode pair
```

Each writes `RUN/eval/ckpt_00100000{_instinct0,_h1,_h4_mean,_h4_pair}_forage16.json`: `forage_hits_per_1k` (food:
successes), `forage_poison_per_1k`, `forage_hunger` (the mean at the end), `forage_capped` (share of steps at hunger 1). ~10 s to a
few minutes each (the lookahead is slowest).

**Chance** (`forage_chance.py --config configs/U_touch.json`): blind random walks in the same worlds. Food alone is no
evidence of steering (a fast walk turning every 5 steps eats as much as the best agents); the food : poison ratio is
(no blind walk in the touch world exceeds 0.78). The view does not change it (the egocentric world has the same
dynamics).

## 3. The dream curve (`eval_dream.py`)

```bash
./venv/bin/python eval_dream.py RUN --ckpts 100000          # --horizon 16 --decode pair by default
```

The frozen agent forages in 16 worlds; from 4 points of each life the model dreams 16 steps on its own predictions,
with the real moves that followed, so each dreamed frame has a true one. `RUN/eval/ckpt_00100000_dream.json`, per
dreamed step h: `exact` (share of pixels exactly right), `mae`, `hunger_err`, `cursor_dist` (px between the dreamed
and the real cursor's white pixels) and `cursor_lost`. In an egocentric world the cursor is always drawn at the center,
so `cursor_dist` is 0 by construction: compare `exact` and `hunger_err` there.

## 4. Tables (`tables.py`)

```bash
./venv/bin/python tables.py $OUT/00?_TB_s? $OUT/00?_TSF4_s? [--step 100000] [--segments 4]
```

Markdown, ready for a README: the frozen tests per run (food / poison and their ratio) and the means per arm; life by
equal segments (food / poison per 1k from `metrics.jsonl`'s `step_hits` and `poison`, and any 1k windows without food);
the dream curve at h = 1, 6, 11, 16. Runs from several experiments can be mixed (arms keep the order given).

## 5. Videos

```bash
# a frozen agent foraging, 1,000 steps (forage.mp4, forage_sheet.png: a panel per 100 steps)
./venv/bin/python forage_video.py RUN --step 100000 --steps 1000 --worlds 16 --horizon 4 --decode pair -o DIR
#   --vs RUN2 [--vs-title T] adds compare.mp4: two agents side by side in the same world (world 0)
# forage 200 steps, then dream 200 on the model's own predictions and moves, the real world beside it
./venv/bin/python dream_video.py RUN --step 100000 --real 200 --dream 200 -o DIR    # dream_pair.mp4, .npz, sheet
./venv/bin/python dream_measures.py DIR/dream_pair.npz
```

- `forage_video.py` prints world 0's counts and the means over its `--worlds`; with 16 they equal the forage test's.
  `--horizon` / `--decode` override the run's own instinct. Touch worlds show touches, egocentric worlds draw the path
  and touches relative to the centered cursor.
- `dream_measures.py` reads the dream video's frames: per dreamed step the cursor's white pixels (a whole cross is 13)
  and the objects' pixels (circles and dots), dreamed and real; how many dreamed steps draw the cursor (≥ 10 px), and
  the first step at which the dream is empty (< 30 object pixels). The dream's moves are sampled from the model's own
  move prediction (no instinct), so it is one sample, not an average.

## 5a. Text: next-letter cross-entropy across layouts (`letter_ce.py`)

```bash
./venv/bin/python letter_ce.py RUN [RUN ...] --ckpts 20000 30000 40000 100000   # also by letter column and line
```

The two-hot cross-entropy (nats per letter) at the token that predicts each letter's first appearance, on the same
800 val windows for every run (vm-00046, vm-00047). Use it to compare text runs whose layouts differ (scaled worlds,
summary tokens, splits): their training losses are not comparable. Text lifetimes overfit after ~30k ticks, so compare
at the best checkpoint, not only at 100k. In the normal text world (every patch a letter) it measures what `eval_predict.py`'s patch cross-entropy does.

## 5b. Pieces of one body (`eval_pieces.py`, task 00059)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_pieces.py RUN [RUN ...] --rule gain --touch
```

Frozen final checkpoints (a one-model run works too) in 16 worlds for 1,500 ticks: food, poison, move size, smoothness,
aim at food within 16 pixels, wins by the food's quadrant. `--rule` swaps the rule on frozen weights (mean, max, gain,
random, council, veto, sharp, gain_sharp). `--touch` takes each poison touch apart. Lifetime numbers come from each
run's `metrics.jsonl`, over 40k-60k: `all_hits_per_1k`, `all_poison_per_1k`, per piece `prop_dx_pK`, `agree_pK`,
`won_x_pK`.

## 6. Writing up

The experiment directory holds only README.md and Makefile (the tag keeps the code). The README is the report: goal,
method (arms, settings, GPUs, speeds), log, results (tables from `tables.py`), analysis, conclusions, next steps;
compare with earlier experiments' arms where they share a setting. State caveats: arms that differ in more than one
setting, seeds that behaved unusually, measures that mean nothing in a world (the egocentric cursor distance).
