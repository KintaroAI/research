# Task 00133: One Transformer on the Whole Frame, Watching (experiment vm-00120)

**Date:** 2026-11-26
**Status:** Complete (2026-11-27), pending tag `exp/vm-00120`: 24 lives of 60k ticks scored and set against vm-00119's four transformers by seed; `00120-one-transformer/README.md`.

## Context

- **The owner, 2026-11-26, after vm-00119:** "I think we need to do a step back - it is expected that adding transformers will reduce accuracy - lets do the same set of tests (world prediction) on one transformer and 64x64 field and see where it will get us".
- **Why the model has four transformers.** They were taken for acting, not for prediction: in vm-00051 four quarter transformers whose moves compete (the winner by gain) ate 42.8 food per 1k where one model ate 19.0, and the cause was the competition while learning. For prediction the split was tested only in the text world (vm-00049, vm-00050: quarters with every summary lose nothing; summaries near the piece only cost) and judged "no loss that could be seen at 64 x 64, a mild test" by foraging (vm-00073). It has never been measured on pictures by a world-model measure.
- **vm-00119** gave the measures and the four transformers' table: next-frame skill .667 in the scripted court and .337 in the disc world; the ball held for 9.1 ticks, a disc for 3 to 4.

## The arms

- **OD, OB:** `configs/W1_discs.json` and `configs/W1_breakout.json`: `W_discs.json` and `W_breakout.json` with one transformer. The same model (8 layers, width 128, 4 heads, context 8 ticks, 32 bins, the pyramid of summaries, the age key, the ring) without `model.band_n`, `band_cols`, `band_win` and without `online.pieces`. It only watches (`online.watch`). Seeds 1–12, 60k ticks, a checkpoint every 10k; seeds 1–8 on the RTX 4090 and 9–12 on the RTX 3090, as vm-00119's.
- **Against WD, WB** (vm-00119), paired by seed: a seed's lives watched the same world (the same discs; the same serves).

What differs between one and four:

| | One transformer | Four quarter transformers |
|---|---|---|
| Tokens a tick | 87 (the body, 64 patches, 21 summaries, the move) | 4 x 32 (each: the body, its 16 patches, 14 summaries near it, the move) |
| What a patch token can read | every patch of the frame | the patches of its quarter; the rest as summaries near its quarter, coarser with distance |
| Parameters | 2.17M | 8.64M (4 x 2.16M) |
| A dreamed frame | one sequence, decoded patch by patch | four sequences, their quarters joined into one frame |
| Ticks per second, one life alone on the RTX 4090 | 54 (a first look, 1,500 ticks) | 40 |

So the one transformer has a quarter of the parameters and full sight; the four have four times the parameters and narrow sight.

## The numbers

The measures of vm-00119 (`dev/wm_measures.py`, `tasks/00132-watching-worlds.md`), with `pair`, the reference decoder, on the same exam (seed 1001; the final set stays unopened): the regular starts (next-frame skill, the dream tick by tick, the things held, the valid frames), the events, the warm-up, the dream after a short warm-up, home, the dream of 200 ticks (seeds 1–3), learning (the checkpoints of 10k to 50k).

**Not run:** the other decoders (`mean`, `instinct`, `instinct_pair`). vm-00119 settled that question; it is not what is asked here.

**New, the seams.** If the split costs, it should cost where the quarters meet. The next frame and the first 8 dreamed ticks of the regular starts, with the squared error summed apart over the pixels within 4 px of a quarter's border (x or y from 28 to 35: 960 of the 4,096 pixels) and over the rest, for the model and for the copy; the skill of each part. The four transformers' lives of vm-00119 are scored again for this part only.

**The scoring batch.** The one transformer's attention over 696 tokens a window does not fit 256 windows on a card (15.9 GB for the attention alone), so it is scored 64 windows at a time (`--chunk 64`; 7.9 GB). The four were scored 256 at a time. The decoding is exact in both; only the last bits can differ.

## Readings, set before the results

All paired by seed, one − four, with the 95% interval over the 12 seeds, per world.

1. **The next frame.** One − four in next-frame skill above 0 with its interval: the split costs accuracy, as the owner expects. The interval includes 0: no difference to be seen at this size. Below 0: the four predict better (they have four times the parameters).
2. **The dream.** The same reading for the things held (the horizon of each kind; the share held at ticks 4 and 8) and for the dream's horizon by pixels.
3. **The seams.** If the four lose to the one, the loss should be larger near the seams than away from them (one − four skill near, against far).
4. **The window.** The next-frame skill and the dream from 2, 4 and 8 real ticks: does one transformer use more of its window than the four (who use 4 ticks, in the court's dream 2)?
5. **What follows.**
   - One is ahead on 1 and 2: the world-model work goes on with one transformer (training on its own dreamed frames exists for it in `lifetime.py`; the four refuse it), and the four stay a matter of acting.
   - Not ahead: the split is not the limit at this size. Then a wider single transformer with the four's parameters (about 8.6M) is the arm that separates the split from the size.
   - Mixed (ahead in one world only, or on the next frame and not in the dream): reported as it is, by world.

**What I expect** (written before the lives ran): one transformer ahead in the disc world, on the next frame and on the discs held, because a disc of 5 to 11 px radius lies across a seam much of the time; about level in the court on the next frame (the ball is 2 px wide) and ahead on the ball's horizon; and the four's loss sitting near the seams.

## To build

- `wm_measures.py`: `score --chunk N`; the part `seam`; `compare OUT_A ARM_A OUT_B ARM_B` (the paired table of two arms). Tests. Added after the first results: `borders` (the things held by whether they have lain across a border between the quarters, from the kept distances) and `reach` (the ticks after which one arm's mean curve is at the other's final level).
- `00120-one-transformer/`: README, Makefile (`runs`, `anchors`, `score`, `warm-dream`, `seam`, `home`, `learn`, `free`, `seam-four`, `borders`, `table`, `compare`, `images`).

## Results (2026-11-27)

All 24 lives ran to 60k ticks and were scored; the write-up is `00120-one-transformer/README.md`. One − four, paired by seed over 12 seeds, `pair`, the working set:

- **Reading 1, the next frame.** The court: +.092 ± .010 (12 of 12 seeds): the split costs accuracy. The disc world: −.015 ± .043 on the exam's new discs, no difference to be seen; on the life's own discs +.131 ± .021.
- **Reading 2, the dream.** One transformer ahead in both worlds: a food disc held at tick 4 +.228 ± .045, the ball at tick 8 +.065 ± .049; the dream's horizon by pixels +2.3 ± 1.6 ticks (discs) and +2.9 ± 1.3 (court).
- **Reading 3, the seams.** In the court the four's loss is larger near the seams at the first dreamed tick (+.045 ± .022 (11 of 12 seeds)) and on the balls that have crossed a border (+.080 ± .050 (10 of 12 seeds) at tick 8). In the disc world it is not: a disc inside one quarter all the way is lost as fast (+.015 ± .064 (5 of 12 seeds)).
- **Reading 4, the window.** Hardly: at the court's next frame the one transformer gains +.012 ± .008 from the 5th to the 8th real tick and the four −.000 ± .008; its dream starts as well from 2 real ticks (the court) or 4 (the discs) as from 8.
- **Reading 5, what follows.** One transformer is ahead on the dream in both worlds and on the next frame in the court. The four are ahead on the forecast of a hit (−.010 ± .008 one tick ahead) and on what is drawn where nothing moves on new discs. By the reading the world-model work goes on with one transformer, for the owner's word.
- **Added after the first results:** `wm_measures.py borders` (the things held by whether they have lain across a border between the quarters) and, under `compare`'s Learning, the ticks after which the one transformer is at the four's level of 60k (about 17k ticks in the disc world's dream, about 27k ticks in the court's next frame).

## Verification

- `tests/test_wm_measures.py`: 33 passed; with `tests/test_watch.py` 51, both files run on the free RTX 4090 with the final tool. New in it: the seams, the things by where they have been, the ticks to a level, one transformer through the same three calls, and the comparison printed from scored lives.
- The whole dev suite before the commit: 2,123 passed, 34 skipped (2026-11-27; 4 more than vm-00119's 2,119, the new tests of this file).
- Every life's files checked before the tables were made (`pieces == 1`, 64 windows a pass, every part, no frame whose decoding did not settle; the four's seam part): 1,458 items, 0 failed.
- The README's results are filled from the lives' files by a builder (0 missing). Typed by hand are only the costs (ticks per second, card memory), the sizes, and the look at seed 5's frames.
- All 24 lives: 6 checkpoints each, the last log line at tick 60,000, no error in a log.
- Self-forcing with one watching transformer starts and runs (a 96-tick run of `configs/W1_discs.json` with `online.sf_windows = 8`); with `configs/W_discs.json` it is refused.
- A scoring fault on the way: two scoring processes started beside six lives on the RTX 4090 and four learning-curve jobs died of memory; they were run again (the queue reads the card's memory, which falls between two checkpoints of one job).
