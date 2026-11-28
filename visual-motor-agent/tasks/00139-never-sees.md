# Task 00139: A Life That Never Sees (experiment vm-00126)

**Date:** 2026-11-27
**Status:** Complete, tagged `exp/vm-00126` (2026-11-28; the owner's word of 2026-11-27): a model that is never given a frame paints the scenery, and without replay the court's paddle near its place; no ball, no disc. One seed an arm: a look.

## Context

The owner, 2026-11-27: "lets try different stuff - lets have empty input (black for example) and train model on a real image - so the idea is that weights of transformer are changing under the influence of real image - but model do not actually ever see the image in the first place - I want o see what will happen".

So the frame reaches the model by one way only: as the target of its updates. Whatever the model then paints for a frame, it paints from its weights.

## What is run

The one watching transformer of vm-00120 (`configs/W1_discs.json`, `configs/W1_breakout.json`), seed 1, 60k ticks, with `online.blind = [0, 0, 64, 64]` (task 00118's blind spot over the whole frame): every frame the model is given is black (grey level 0), while acting and in every update; the targets and memory are the real frames. The body's values and the move stay as they are: in the court the body's hunger is the score (a hit sets it to 0, a miss adds .5); in the disc world it is 0.

| Arm | World | The update |
|---|---|---|
| EB, ED | the court, the discs | as always: the newest window and 7 replayed from memory |
| EB1, ED1 (mine) | the court, the discs | `online.batch = 1`: the newest window only, no replay |

**Why the second pair** (mine): with black input the model cannot tell a replayed window from the present one, so 7 parts of 8 of every update pull its picture toward the average of all it has lived. Without replay the weights follow the present alone: the plainest form of "the weights change under the influence of the real image".

## How it is seen

- **In the life, at every tick** (`online.pred_score = 1`, new): before the model learns from a frame, its picture of that frame is read: the sharp decoding (`pair`) of what it predicts for the current frame from the window as it sees it (black). Its summed squared error against the real frame, in grey levels, beside two that need no model: the last real frame (copy) and the mean of all frames so far. In the metrics by 1,000 ticks: `pred_err`, `pred_copy`, `pred_mean`.
- **Frames kept** (`online.pred_record = N` with it; `online.pred_every = K`): the real frame and the model's picture at the first and last N ticks and at every K-th tick (`pred_frames.npz`), for a strip (`pred_show.py RUN -o PNG`) and a time-lapse video (`pred_show.py RUN ... -o MP4`).
- **After the life, frozen:** `wm_measures.py score --lived` shows the exam as the life saw frames (black); the life's own world after its end (`--set home`) and the working set. And the exam shown whole real frames, which the model has never been given.

## Readings, set before the results

One seed an arm: a look.

1. **What does the model paint?** The pictures and the video.
2. **Is its picture better than repeating the last frame, and than the average frame?** The error in the life against copy and against the mean frame, by 1,000 ticks.
3. **Do the changing weights carry the present?** The newest window alone against the usual update; and the life's last 1,000 ticks (learning on) against the same weights frozen, on the life's own world right after.
4. **Shown real frames for the first time** (the exam with whole frames): what does it do?

**What I expect** (written before any life of this experiment ran, but for a trial of 400 ticks that only showed the option runs):

- With the usual update the picture settles within a few thousand ticks to the scenery without its things: the court's walls and background with no ball and no paddle, the discs' world an empty background. Each pixel's most probable value is the background, and `pair` draws that.
- Its error is then the things themselves. Against copy it loses clearly in both worlds (copy is wrong only where a thing moved); against the mean frame it is about level.
- Without replay the picture follows the slow things with a lag: a paddle where it has lately been, faint trails of the slow discs. A lower error than with replay, still above copy's.
- Frozen at the end: with replay no different from the life; without replay worse than the life's last ticks, since the picture stops following.
- Shown whole frames: nothing useful. The input weights of every grey but black were never trained.

## Changed on the way (2026-11-28, after the first two lives were seen)

- **Two more lives of mine: EB1x and ED1x,** the newest window only with the learning rate 10 times larger (`ppo.lr = .003`). The life of the court without replay painted the paddle near its place, the life of the discs nothing: do faster weights follow the discs?
- **A reading of the court's paddle and ball in the kept pictures** (`pred_show.court_things`): is a paddle painted, how far its place is from the real one's, is any ball pixel painted.

## To build

- `online.py`, `lifetime.py`: `online.pred_score` (one transformer on the GPU with the ring: the picture of the current frame at every tick and its errors in the metrics), `online.pred_record` and `online.pred_every` with it (the frames kept, `pred_frames.npz`). The four quarter pieces' `pred_record` (task 00118) stays as it is.
- `wm_measures.py`: `Pieces(..., lived=True)` for a run with `online.blind`: the real ticks shown with the box black, as the life saw them.
- `pred_show.py`: the strip and the video.
- `tests/test_never_sees.py`.
- `00126-never-sees/`: README, Makefile.

## Results (2026-11-28)

Six lives of 60k ticks, one seed an arm (a look). The README has the tables, the pictures and the readings one by one.

- **1. What the model paints:** with the usual update the scenery without its things, settled by 2,000 ticks: the court's walls and background (a paddle in .005 of the kept pictures), the disc world an empty background. Without replay the court life paints a paddle (in .980 of its pictures). No ball in any court life, no disc in any disc life.
- **2. Against copy and the mean frame:** worse than both with replay (15.6 grey levels a pixel against 10.2 and 14.3 in the court; 33.4 against 16.8 and 29.8 in the disc world). Without replay the court's picture is between them (13.2; better than copy in 1 window of 59).
- **3. Do the changing weights carry the present?** Where it is slow and nothing is replayed: the court's paddle is painted a median of 2.5 px from its place (a paddle at its mean place: 13.6 px). Frozen, EB1's picture is as good as in the life for 8 ticks (11.1) and worse than the scenery from 65 on (19.1 against 15.6). Of the discs next to nothing: a few pixels of their grey in 18 and 25 of 644 kept pictures without replay (a count of mine), and no gain right after the life's end in the sharp picture or in the expected value.
- **4. Shown real frames for the first time:** the court lives paint far worse (28.0 against 15.6), the disc lives the same empty background.
- **What I expected:** right for the scenery, for copy, for the paddle without replay and for the frozen weights; wrong for the discs (I had faint trails; there are none worth the name) and a little for the mean frame (the picture is below it with replay, not level).
- **Found on the way:** the exam's dream feeds the model its own picture while it paints a frame, which a blind life was never given: its exam numbers are not its pictures'. `pred_show.py after` reads the frozen weights the way the life makes its pictures.

## Verification

- `tests/test_never_sees.py`: with the whole frame blind no pixel of a real frame reaches the model (two worlds that differ only in their frames give the same pictures while learning is off; the update's inputs are black); the scored picture is the one the life's window gives; the errors in the metrics are those of the kept frames; `--lived` blacks the box in the exam. The whole dev suite.
- Every life: 7 checkpoints and the final one, the last log line at tick 60,000, no error; `pred_err`, `pred_copy`, `pred_mean` in every row.
- Done: `tests/test_never_sees.py` (11), the whole dev suite 2,181 passed, 34 skipped; 120 checks over the six lives' files, 0 failed (the points above, each life's settings against the plain life of vm-00122, the 727 kept frames, the frozen scorings).
