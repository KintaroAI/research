# Task 00137: What the Mix Is Made of: More Prediction, and Random Grey (experiment vm-00124)

**Date:** 2026-11-27
**Status:** Running (the owner's word of 2026-11-27, while vm-00123 ran; the court lives started at 17:40, the disc lives follow as the cards come free).

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

## To build

- `online.mix_noise` in `online.mix_inputs`, `lifetime.py` (the metric `noise_share`) and `wm_measures.Pieces` (`--lived` draws the greys too); `tests/test_mix.py`.
- `wm_curves.py`: the new arms' names and colours.
- `00124-mix-shares/`: README, Makefile.

## Verification

- `tests/test_mix.py`; the whole dev suite.
- Every life: 7 checkpoints and the final one, the last log line at tick 60,000, no error; `mix_share` .7 or .5 and, for MND and MNB, `noise_share` .2 in every row.
- Every scored file names its life, one transformer, 64 windows a pass, `pair`, nothing unsettled; the files scored as lived say so.
