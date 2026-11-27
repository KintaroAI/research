# Task 00134: Training on Its Own Dreamed Frames, Watching (experiment vm-00121)

**Date:** 2026-11-27
**Status:** Complete, tagged `exp/vm-00121` (2026-11-27): 20 lives of 60k ticks scored against vm-00120's by seed; `00121-self-forcing-watching/README.md`. Started by the owner's word of the night before ("lets do as you suggest so we keep GPU loaded and have some results by morning"). **Seeds 1–6 only:** after the disc world's first six seeds I recommended skipping seeds 7–12, and the owner agreed (2026-11-27, morning: "lets skip skipping seeds 7–12", which I first read as "do not skip", and then "no lets skil seeds 7-12"). Four lives of seeds 7 to 10 that the queue had started at 08:49 were stopped within four minutes, before their first 1,000 ticks, and removed. **The court's 7-step arm has seeds 1–2:** minutes later the owner set the rule for what follows ("I think we need to stop wasting resources on so many seeds runs - I do not need scientific and statistically significant results - just either it's working or not - we will be running multiple seeds as soon as we find something meaningful to run it about"), and its four other lives (SB7 seeds 3 to 6, at 32k of 60k ticks, two hours from their end) were stopped and removed. As run: SD3, SD7 and SB3 on seeds 1–6, SB7 on seeds 1–2: 20 lives.

## Context

- **The order of work** (`ARCHITECTURE.md` section 15, item 4): the first change to the watching model is the dream in the disc world. vm-00120 settled the model it starts from: one transformer on the whole frame (`configs/W1_discs.json`, `configs/W1_breakout.json`).
- **Where the dream stands (vm-00120, 12 seeds, 60k ticks, decoded with `pair`):** a food disc is still in place at dreamed tick 4 in .666 of the dreams and at tick 8 in .298; its horizon is 4.8 ticks, the poison's 5.0. A straight line through the disc's places in the last two frames as seen holds it for 10 and 13 ticks (.807 at tick 4, .635 at tick 8 for the food). In the court the ball is held for 10.8 ticks (a line through the 8 frames of the window: 11; through the true places: 15).
- **The suspected cause:** the model has only ever read real frames. In a dream its inputs are its own frames, a little wrong, and it was never trained on those; the errors feed themselves.
- **The remedy exists** (task 00039, vm-00032): self-forcing. `online.sf_windows` replayed windows of every update get the inputs (frame and body) of their last 1 to `online.sf_steps` steps from the model's own predictions, dreamed in order with `pair` decoding and `online.sf_jacobi` passes a frame; the targets stay real. In vm-00032 (the click world, the single model of that time) the dreamed cursor stayed 3.6–5.2 px from the real one after 16 steps against 7.6–18.6. `lifetime.py` refuses it with `online.pieces` (the four) and runs it with one transformer.

## The arms

All: one transformer, watching (`online.watch = 1`), 60k ticks, a checkpoint every 10k; set on the command line (`--set`), no new config.

| Arm | World | Config | Self-forcing |
|---|---|---|---|
| SD3 | discs | `configs/W1_discs.json` | `online.sf_windows = 2`, `sf_steps = 3` (vm-00032's rule: its share of the batch, a quarter; up to 3 dreamed steps) |
| SD7 | discs | the same | `online.sf_windows = 2`, `sf_steps = 7` (as deep as the window of 8 allows: one real step is left) |
| SB3, SB7 | the scripted court | `configs/W1_breakout.json` | the same two |
| OD, OB | | | none: vm-00120's lives of the same seeds, not run again |

- An update trains on 8 windows: the newest and 7 replayed. 2 of the 7 are self-forced; the number of dreamed steps of each is drawn from 1 to `sf_steps`. Dreamed input steps are 6% of the batch's steps with `sf_steps = 3` (vm-00032: 6%) and 12.5% with 7.
- `sf_jacobi = 2` (the default, vm-00032's): a dreamed frame is two passes from the frame before it. This is cheaper than the exam's exact decoding (about 11 passes a frame in the disc world) and not the same frame; a third arm with more passes is left for later.
- **Seeds 1–6 first** (24 lives; the disc world's before the court's), then seeds 7–12 behind them on the same queue (not run: skipped by the owner's word after the first table, see Status). A seed's lives watch the same world as vm-00120's and start from the same weights.
- Why depth is the second arm: at dreamed tick h the window holds h − 1 dreamed frames (all 8 from tick 9 on). Up to 3 dreamed steps train the model for dreamed ticks 2 to 4, up to 7 for ticks 2 to 8. The disc's horizon now is 5 ticks.

## The numbers

vm-00119's measures (`dev/wm_measures.py`: `reset`, `observe`, `dream`; exam seed 1001, the final set stays unopened; `pair`, exact decoding, 64 windows a pass): the regular starts (next frame, the dream by tick, the things held, valid frames), the events, the warm-up, the dream after a short warm-up, home, learning (the checkpoints of 10k to 50k), the dream of 200 ticks for seeds 1–3. Not scored: the seams and borders (no quarters here) and the other decoders.

`wm_measures.py compare` pairs each arm with vm-00120's lives by seed, and SD7 with SD3.

## Readings, set before the results

Paired by seed, self-forcing − none, with the 95% interval over the seeds, per world and arm.

1. **The discs held (the aim).** The horizon of the food and of the poison in ticks and the shares held at dreamed ticks 4 and 8. Above 0 with the interval: self-forcing holds the discs longer. **The mark:** the straight line through the last two frames (10 and 13 ticks; .807 and .635 of the food at ticks 4 and 8). Reached or not is said apart from "longer".
2. **The price.** Next-frame skill (on the exam's new discs and at home), the error where nothing moves, the valid frames; in the court the forecasts of a hit and of a miss. A difference below 0 with its interval is a price and is named with its size.
3. **The court.** The ball's and the paddle's horizon, the ball held at tick 8, the dream's horizon by pixels: is the gain there too, and is the court unharmed?
4. **The depth.** SD7 − SD3 and SB7 − SB3, paired: does dreaming up to 7 steps add to up to 3?
5. **What follows.**
   - The discs are held longer in an arm and the next frame pays nothing (its interval includes 0 or lies above): self-forcing joins the watching model for the next changes, in the arm with the longer horizon (the cheaper `sf_steps = 3` if the two do not differ). For the owner's word.
   - Held longer at a price: both numbers are reported and the owner decides.
   - Not held longer: exposure to its own frames is not what loses the discs. Then the next suspects are the decoding (each pixel decoded apart: patch codes, section 15) and the window (the model reads 2 to 4 of its 8 ticks).

**What I expect** (written before the lives ran): the discs held longer in both arms, at tick 4 about .75 to .80 of the food in place of .666 and a horizon of 6 to 7 ticks; SD7 a little ahead of SD3 at tick 8; the mark of the straight line not reached. The next frame level or slightly worse (a quarter of the batch reads inputs that are a little wrong). The court: the ball's horizon up by a tick or two, the forecasts level.

## Cost

Self-forcing adds forward passes to every update: up to 3 a dreamed step (one for the body, two for the frame). Measured before the start with four lives together on the RTX 3090 (see the experiment's README).

## To build

- Nothing in the model or the life script: `--set online.sf_windows=2 --set online.sf_steps=K`.
- `00121-self-forcing-watching/`: README, Makefile (`runs`, `anchors`, `score`, `warm-dream`, `home`, `learn`, `free`, `table`, `compare`, `images`; added after the first results: `course`, `course-base`).
- A queue for the night (scratch, not in the repo): the lives by priority on both cards, one scoring process a card beside them.

## Added after the first results (2026-11-27, seeds 1–6 of the disc world; mine, not in the plan above)

- **A fault in the tool, found by the first comparison:** `wm_measures.py compare` stopped with `KeyError: 0` when only one of the two arms has the seam part scored (vm-00120's lives have it, these have not): a part that is not there reads as nan, and nan passed the check for a number. Corrected; `tests/test_wm_measures.py::test_two_arms_are_compared_on_the_parts_both_have` fails on the tool as committed and passes now.
- **A new part, `course`** (`wm_measures.py score --parts course`): is a dreamed thing behind the real one on its course, ahead of it or aside? The question came from a look at vm-00120's dreams while the lives ran: the dreamed discs drift, and mostly backward along their course.
  - On the regular starts, at dreamed tick h, every thing that is there and whose true steps up to tick h are all the step it made into the last real frame, at least .1 px long (it keeps the course the last two real frames show: no turn, no push, not standing).
  - The reader's shift (the dreamed place less the true place, whole px) is split along that step and across it. Per kind, by tick: the things on a straight course, found, held; and over the found: px along the course (below 0: behind), the shares more than half a px behind and ahead, px aside, and the way gone along the course since the last real frame as a share of the true way (1 keeps the speed, 0 stands still).
  - It reads the things through the same reader as the other parts and needs nothing of the model but its dream. It works in the court too (the ball between bounces, the paddle while it runs at a steady speed).
  - `--home-starts N`: the home set with N regular starts in place of 16 (a longer stretch of the life's own world; kept under its own name, `--name home64`). 16 starts hold about 20 discs of a kind on a straight course at tick 8, too few to read a share.
  - Read with care: the reader works in whole pixels, so at tick 1 a poison disc (.75 px a tick) is read at 0 or 1 px behind and nothing between; the way gone is a fair number from the tick at which the true way is 3 px or more. A thing more than 8 px off (the reader's reach) is read at the reach's edge or not found.
  - `compare` prints the block "On its course" for the working set and for `home64`. Makefile: `course`, `course-base` (vm-00120's lives, the other side of every pair; their summaries get the part added, nothing else in them changes).
  - Tests (no card needed): the world itself keeps every thing's speed; a dream at half speed has gone half the way; the copy none; a frame moved by 4 px has its things 4 px off, mirrored when moved the other way.

## Results (2026-11-27)

20 lives ran to 60k ticks and were scored (SD3, SD7, SB3 on seeds 1–6, SB7 on seeds 1–2); the write-up is `00121-self-forcing-watching/README.md`. Self-forcing − none, paired by seed, `pair`, the working set:

- **Reading 3, the court: it works.** Up to 3 steps: the ball held for 16.2 ticks against 10.8 (+5.3 ± 2.9, 6 of 6 seeds), at tick 8 in .826 of the dreams against .645; past the straight line through the window's 8 frames (11) and through the true places (15). The next frame pays nothing (+.009 ± .016); a hit is forecast better within 16 dreamed ticks (+.093 ± .072). SB7 on two seeds: the ball held 13, 12 ticks (+4.0 and +0.0).
- **Reading 1, the discs held: not longer on the exam's new discs.** Up to 3 steps: the food at tick 4 −.003 ± .081, at tick 8 +.010 ± .069. Up to 7 steps: shorter (the food at tick 8 −.123 ± .070, the poison −.143 ± .052), and the food is not found at all in .357 of the dreams at tick 8 (.013 without).
- **The life's own discs: both arms dream better** (skill at tick 8 at home +.104 ± .049 and +.090 ± .054, 6 of 6 and 6 of 6 seeds).
- **Reading 2, the price:** none for up to 3 steps in either world (+.066 ± .071 on new discs). Up to 7 steps cost on new discs (next-frame skill on the food −.256 ± .373).
- **Reading 4, the depth:** 7 steps add nothing to 3 at home and harm on new discs (−.133 ± .100 at tick 8).
- **Reading 5, what follows.** The condition is met in the court (held longer, the next frame pays nothing) and not on the disc world's exam. The cause found there is not one of the two suspects named beforehand: the exam's discs and dots are new to the life (next-frame skill on the dots .264 on the exam against .780 at home). For the owner's word: the disc world with variety, with and without up to 3 dreamed steps, on 1 to 2 seeds an arm.
- **The course part** (added): without self-forcing a dreamed poison disc has gone .786 of its true way at tick 4 on own discs and .719 on new ones; self-forcing reduces the lag on own discs (+.071 ± .014, +.079 ± .056) and up to 7 steps make it worse on new ones (−.201 ± .191).
- **What I expected** (the discs held longer in both arms on the exam) was wrong there and about right on the life's own discs; the court's gain I had put too low (a tick or two).

## Verification

- A short self-forced life is scored end to end before the queue starts (the measures read a self-forced life's weights like any other).
- Every life: 6 checkpoints, the last log line at tick 60,000, `sf_dreamed` above 0 in its metrics, no error in its log.
- Every scored file names its life and seed, one transformer, 64 windows a pass, `pair`, nothing unsettled.
- **Done, 2026-11-27.** Every life's files checked before the tables were made (the final and 6 earlier checkpoints; the info naming one watching transformer with the arm's self-forcing and nothing else changed against its pair of vm-00120; the log's last line at tick 60,000 and no error in it; 2,000 dreamed windows in every row of the metrics; every scored part; the course part of the life and of its pair): 1,398 items, 0 failed. No life of a skipped seed is left in the output folder.
- The README's numbers and tables are filled from the lives' files by a builder (0 missing), and so are the lines of the other documents. Typed by hand: the costs measured before the start and vm-00120's speed.
- The whole dev suite on the final code: 2,125 passed, 34 skipped, in two runs (the measures' own file, 35 passed; the rest, 2,090 passed, 34 skipped). vm-00120's count was 2,123.
- `tests/test_wm_measures.py` with the comparison's correction and the course part: 35 passed (2026-11-27, on the RTX 3090 beside four lives, 22 minutes; 33 before). New: two arms compared on the parts both have; the things on their course (predictors made by hand); the course part scored from the regular part's dreams and for itself, the home set with more starts; the comparison's course block.
- The course part against the look it came from (a scratch script that took a disc's turns from the world's own marks): the same numbers to the second decimal on vm-00120's first life (the way gone at tick 8: food .866 against .865, poison .639 against .639), before the course was tied to the step into the last real frame.
