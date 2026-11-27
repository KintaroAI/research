# Task 00135: Lives of 200k Ticks, Watching (experiment vm-00122)

**Date:** 2026-11-27
**Status:** Complete (2026-11-27), pending tag `exp/vm-00122` (the owner: "how about we run 200k runs for both worlds to see if it improves over time?"): a longer life keeps improving what the life knows (the court, its own discs) and does nothing for discs it has never seen. One seed an arm: a look.

## Context

- vm-00119 to vm-00121 are lives of 60k ticks. At 60k the watching transformer's dream was still getting better: in vm-00120 the food held at dreamed tick 4 on new discs went .346, .562, .666 after 10k, 30k and 60k ticks.
- vm-00121: training on its own dreamed frames (up to 3 steps) holds the court's ball for 16.2 dreamed ticks against 10.8, and does nothing for discs the life has never seen.
- The owner's question: does it improve over time? And the owner's rule of the same day: few seeds first.

## The lives

One seed (1), 200k ticks, a checkpoint every 10k; `configs/W1_discs.json` and `configs/W1_breakout.json` with `--set online.steps=200000`:

| Arm | World | Self-forcing |
|---|---|---|
| OD | discs | none |
| SD3 | discs | `online.sf_windows = 2`, `online.sf_steps = 3` |
| OB | the scripted court | none |
| SB3 | the scripted court | as SD3 |

Seed 1's worlds are those of vm-00120's and vm-00121's seed 1, so the first 60k ticks repeat those lives (not bit for bit: lives on a card do not repeat exactly). The lives' memory holds 100k ticks; from there on the oldest go.

## The numbers

`dev/wm_measures.py` with `pair`, the same exam (seed 1001), at 20k, 40k, ..., 200k: the regular part (next-frame skill, the dream of 32 ticks, the things held) and the course part. The disc lives also on their own world after the life's end (64 starts from tick 200,000 on; `wm_measures.home_skip`, new: a life's home set starts after its last tick).

## Readings, set before the results

One seed an arm: a look, not a measurement. A change over time counts as seen only if it is larger than the spread among the six seeds of vm-00120 and vm-00121 at 60k (the 95% half-widths there: the ball's horizon 1.2 ticks, the food held at tick 4 on new discs .04, next-frame skill on new discs .15, in the court .013).

1. **Does it improve after 60k?** Per world and arm, 200k against 60k: the things held (the ball's horizon; the food and the poison held at ticks 4 and 8), next-frame skill, the way gone.
2. **Does self-forcing keep its gain?** SB3 − OB and SD3 − OD by tick: the court's gain of 60k (+5.3 ticks of the ball's horizon) kept, grown or gone; the disc world's exam still without a gain or not.
3. **New discs against own discs:** does a longer life close the gap, or widen it (more of its scene learnt by heart)?
4. **The marks:** a straight line through the last two frames holds a food disc for 10 ticks and a poison disc for 13; in the court the line through the window's 8 frames holds the ball for 11.

**What I expect** (written before the lives ran): the court keeps improving in both arms and self-forcing stays ahead; the life's own discs keep improving; on new discs the things held rise a little more and level off far below the mark, next-frame skill stays flat (it was flat from 20k on), and self-forcing still adds nothing there.

## To build

- `wm_measures.home_skip` (the home set after the life's last tick) and its test; `wm_video.py` uses it.
- `00122-long-lives/`: README, Makefile (`runs`, `anchors`, `score`, `home`, `curves`, `images`, `videos`).
- Added while the lives ran: `dev/wm_curves.py` (the scored checkpoints of an output folder as tables, a row a measure and arm and a column a tick, and as a picture) and `tests/test_wm_curves.py`.

## Results (2026-11-27)

Four lives of 200k ticks, one seed an arm (a look). The README has the tables, the picture and the readings one by one.

- **Does it improve after 60k?** Where the life knows the things, yes, to the end. The court: the ball held for 16 dreamed ticks at 200k against 9 at 60k (OB) and 26 against 15 (SB3); next-frame skill +.061 and +.067. The life's own discs: the food held at tick 8 in .516 and .523 of the dreams against .266 and .375. New discs: nothing that one life can show (the food held at tick 4 +.014 and +.027, below the .04 set beforehand; next-frame skill −.048 and −.038).
- **Does self-forcing keep its gain?** In the court yes: +6 ticks of the ball's horizon at 60k, +10 at 200k; about 65k ticks with it do what 200k do without. On the life's own discs it shrinks (skill against the copy at tick 8 +.103 at 60k, +.052 at 200k). On the exam SD3 is ahead at every tick, which is the seed: seed 1 had the largest such gain of vm-00121's six at 60k.
- **New against own discs:** the gap widens (next-frame skill +.322 at 60k, +.446 at 200k without self-forcing; +.237 and +.341 with it).
- **The marks:** the court's ball is past the line through the window's frames (11 ticks) from 80k on without self-forcing and from 40k on with it; own discs are close to theirs (the poison 10 and 11 ticks against 13); new discs far below (the food 6 and 7 against 10).
- **What I expected:** right for the court and the life's own discs, about right for new discs (next-frame skill is not flat but falls a little); whether self-forcing adds nothing on new discs is not answered by this seed.
- **Added after the results:** how far one life can be trusted. The same seed run twice differs at 60k (the ball's horizon 15 against 17 in SB3, the food held at tick 8 .398 against .463 in SD3).
- **Added on the owner's request after the results** ("I want you generate video where it starts for 20 tickes and then dream 1000 ticks"): `wm_video.py --real 20 --warm 20 --dream 1000 --outline 32` (`make videos-long`), three videos of the 200k lives. A look at them is in the README: the court's dream loses its ball within 57 to 585 ticks and never serves a new one; the discs' dream goes on to the end, loses the dots within about 100 ticks and grows new discs toward the size of the life's own.
- **Built on the way:** `dev/wm_curves.py` and its test; the picture's first run needed two corrections (a title that ran into the next panel, a label on the dotted line that sat on the curves: the legend names it now).

## Verification

- Every life: 20 checkpoints and the final one, the last log line at tick 200,000, no error in its log. **Done:** each life has its 20 checkpoints, the one of tick 0 and the final one, a log that ends at tick 200,000 and 200 rows of metrics (711 checks in all, 0 failed).
- Every scored file names its life, one transformer, 64 windows a pass, `pair`, nothing unsettled. **Done** (the same 711 checks): 40 checkpoints on the exam and 20 on the disc lives' own world, both parts whole, the exam's floors in every file equal to the anchors'.
- Added: the home set is after the life. **Done:** its floors are the same in all 20 files and equal those worked out again from the world from tick 200,000 on (no model); from tick 60,000 on they differ.
- At 60k the lives without self-forcing are within the six seeds' range of vm-00120. **OB: yes. OD: above it on two numbers of six** (next-frame skill .367 against the six seeds' highest .355, the food held at tick 4 .711 against .676): seed 1 is the best of the six in the disc world and this is its second run. With the self-forced lives against vm-00121's six, 19 of 24 numbers are within the range.
- Tests: `tests/test_wm_curves.py` (4), `tests/test_wm_video.py` (4, with the long dream), `tests/test_wm_measures.py` (36); the whole dev suite 2,134 passed, 34 skipped, run on the final code.
