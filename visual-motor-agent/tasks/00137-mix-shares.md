# Task 00137: What the Mix Is Made of: More Prediction, and Random Grey (experiment vm-00124)

**Date:** 2026-11-27
**Status:** Complete (2026-11-27), pending tag `exp/vm-00124` (the owner's word, given while vm-00123 ran): neither make does better than the half-and-half mix, and both pay more. One seed an arm: a look.

## Context

- vm-00123 (task 00136, running): the watching transformer is only ever shown frames in which each pixel comes from its own prediction of that frame with probability .5 (`online.mix = .5`), the targets real, the mix made again at every update.
- Its first numbers from the lives' own logs: the half it is shown is soon nearly right. In the court 0.12% of the pixels taken from the prediction are not the real ones at 14k ticks (about 3 a frame), in the disc world 3.6% at 8k.
- The owner, 2026-11-27: "and next experiment once this one is finished - lets do 70% mix from next frame and 30% keep from real and yet another experiment - 30% real, 50% from prediction and 20% just grey noise".

## The two mixes

| Arms | From its prediction | Random grey | Real | Options |
|---|---|---|---|---|
| M7D, M7B | .7 | none | .3 | `online.mix = .7` |
| MND, MNB | .5 | .2 | .3 | `online.mix = .5`, `online.mix_noise = .2` |

D: the disc world (`configs/W1_discs.json`); B: the scripted court (`configs/W1_breakout.json`). One seed (1), 60k ticks. Filed as one experiment with two arms a world: both ask what the mix should be made of, against the same lives (said to the owner; two experiments in the owner's words).

**Random grey** (`online.mix_noise = r`, new): each pixel of a frame as seen is the prediction's with probability q, a random grey level (0 to 255, uniform, as `env.noise` and `noise_in` draw it) with probability r, the real one otherwise; one draw a pixel, fresh at every tick and at every update. The greys are part of the frame as seen: the model carries them in its window and predicts the next frame from them. Without the share nothing is drawn for it, so vm-00123's lives repeat as they were made.

**Why these two** (my reading of the owner's idea): vm-00123's noise is the model's own error and all but vanishes as it learns. More of the frame from the prediction leaves less to lean on (.3 real: a ball of 4 pixels has no real pixel in 24% of the frames, against 6% at .5). Random grey is noise that does not shrink.

## The lives and the cards

A life with the mix alone on the RTX 4090 runs 12.7 ticks per second, on the RTX 3090 8.0.

1. When vm-00123's court life ends (about 17:40): M7B and MNB together on the RTX 4090.
2. When its disc life ends (about 18:25): M7D on the RTX 3090.
3. When the two court lives end: MND on the RTX 4090.

## The numbers

As vm-00123's, so that all the lives stand in one table: the exam shown whole real frames at 20k, 40k and 60k (the regular part and the course part); the disc lives on their own world (`--home-from 200000`, `home64`); at 60k the exam shown as the life saw frames (`--lived`: for MND and MNB with their random grey) and the free dream of 200 ticks. Against vm-00123's MD and MB (.5) and vm-00122's OD, SD3, OB, SB3 at the same ticks.

## Readings, set before the results

One seed an arm: a look. A difference counts only if it is outside what two runs of one seed differ by at 60k (2 ticks of the ball's horizon, .06 of the things held at a tick, .025 of next-frame skill; vm-00122).

1. **More prediction (.7 against .5):** are the things held longer in the dream, and what does the next frame pay?
2. **Random grey (.5 with .2 grey against .5):** the same two questions. Does noise that does not shrink do more than noise that does?
3. **Shown whole frames or frames as lived:** a model that lived with random grey may need its grey to see as it learnt to; whole frames are the fair comparison with every other model.
4. **The long dream:** is a ball in the court longer, is one ever served?

**What I expect** (written before any life of this experiment ran, with vm-00123's court life scored only at 10k):

- .7: the dream a little better than at .5 and the next frame a little worse; the differences small, because late in life the prediction is nearly the real frame at either share.
- Random grey: next-frame skill shown whole frames clearly lower than without it (a fifth of every input pixel carries nothing); the dream no better than at .5. A random grey is not an error the model makes, so learning to see through it does not teach it to mend its own dream.
- Neither serves a ball in the long dream.

## Results (2026-11-27)

Four lives of 60k ticks, one seed an arm (a look). The README has the tables, the pictures and the readings one by one.

- **1. More prediction (.7 against .5):** the things are held less long, not longer. The court's ball 12 dreamed ticks against 15 at 60k (held at tick 16 in .306 of the dreams against .492); the food at tick 4 on own discs .516 against .555, on new discs .514 against .564 (inside the yardstick, the same way at every tick). The next frame pays more: the court .696 against .742, new discs .055 against .166.
- **2. Random grey (.5 with .2 grey against .5):** worse on both questions. The court's ball 11 ticks (no ball in its dream at 20k, 3 ticks at 40k: still climbing at the end), next-frame skill .656; own discs: the food at tick 4 .367, next-frame skill .445 against .639. On new discs its next frame is level with .5 (.151 against .166) and is the only mix's that rises through the life. Noise that does not shrink does not do more than noise that does.
- **3. Whole frames or frames as lived:** whole frames, for the next frame in every life and for the court's ball (the court's next-frame skill .656 whole against .466 as lived with random grey, .696 against .511 at .7). The model that lived with random grey does not need its grey.
- **4. The long dream:** not mended. The ball pixels left at tick 200 of the free dream (a ball is 4): 0.00 with random grey, 1.34 at .7, 0.47 at .5, 2.03 and 1.78 without a mix. No dream serves a new ball (the world serves 11 times in the same 32 stretches; a scratch count of mine). Why, by a scratch test on the 60k checkpoints: every life expects the serve when it is due (its expected next frame holds a band worth 0.6 to 1.2 ball pixels in the court's top rows in the last two ticks before a serve) and not where, and `pair` draws each pixel's most probable value, which is empty. This corrects my reading in vm-00122, which blamed the window as well. By the same count the noisier makes sometimes draw a ball again that was gone for 3 to 5 ticks (twice in 32 dreams at .7, 4 times with random grey, never at .5).
- **What I expected:** wrong for .7's dream (I had it a little better; it is worse) and for the size of what its next frame pays; right for random grey (the next frame clearly lower, the dream no better) and for the missing serve. Not expected: how late the life with random grey learns to see, and the balls drawn again.
- **Changed on the way: the picture.** The plan gave each make a colour of its own. Five hues in one panel cannot all be told apart (a palette check of every pair), so the makes of the mix share the mix's green and differ by a marker and a label at the line's end, and a world with more than three arms has a row a set (`wm_curves.py`; vm-00122's and vm-00123's pictures come out as they were, pixel for pixel).

## To build

- `online.mix_noise` in `online.mix_inputs`, `lifetime.py` (the metric `noise_share`) and `wm_measures.Pieces` (`--lived` draws the greys too); `tests/test_mix.py`.
- `wm_curves.py`: the new arms' names and colours (as built: one colour for every make of the mix, a marker and a label each; see Results).
- `00124-mix-shares/`: README, Makefile.

## Verification

- `tests/test_mix.py` (24); `tests/test_wm_curves.py` (6, one new); the whole dev suite: 2,161 passed, 34 skipped (2026-11-27, after the last change to `wm_curves.py`).
- Every life: 7 checkpoints and the final one, the last log line at tick 60,000, no error; `mix_share` .7 or .5 and, for MND and MNB, `noise_share` .2 in every row.
- Every scored file names its life, one transformer, 64 windows a pass, `pair`, nothing unsettled; the files scored as lived say so.
- Done: 373 checks over the four lives' files, 0 failed (the points above, each life's settings against the plain life of the same seed, the record of what it was shown, the exam's floors against the anchors', the home stretch against vm-00122's).
