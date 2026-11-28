# vm-00124: What the Mix Is Made of: More Prediction, and Random Grey

**Date:** 2026-11-27
**Status:** Running (four lives, one seed an arm; the two court lives started on 2026-11-27 at 17:40, the disc lives follow as the cards come free)
**Source:** `dev/` (not tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00124/`
**Task plan:** [`../tasks/00137-mix-shares.md`](../tasks/00137-mix-shares.md)

## Goal

The owner, 2026-11-27, while vm-00123 ran: "and next experiment once this one is finished - lets do 70% mix from next frame and 30% keep from real and yet another experiment - 30% real, 50% from prediction and 20% just grey noise".

vm-00123 shows the watching transformer only frames in which half the pixels come from its own prediction of that frame. What should that mix be made of: more of the prediction, or noise that is not the model's own?

## Method

**The mix** is vm-00123's (`online.mix`, `online.mix_inputs`): at every tick the model predicts the coming frame from what it has seen, and each pixel of the frame it is shown is drawn anew; the targets and memory are the real frames, and the mix is made again at every update.

| Arms | From its prediction | Random grey | Real | Options |
|---|---|---|---|---|
| M7D, M7B | .7 | none | .3 | `online.mix = .7` |
| MND, MNB | .5 | .2 | .3 | `online.mix = .5`, `online.mix_noise = .2` |

D: the disc world (`configs/W1_discs.json`); B: the scripted court (`configs/W1_breakout.json`). Seed 1, 60k ticks (few seeds first).

**Random grey** (`online.mix_noise`, new here): such a pixel is a grey level drawn uniformly from 0 to 255, fresh at every tick and at every update. It is part of the frame as seen: the model carries it in its window.

Against vm-00123's MD and MB (.5 from its prediction, .5 real) and vm-00122's OD, SD3, OB and SB3 at the same ticks.

**The numbers** are vm-00123's: vm-00119's measures at 20k, 40k and 60k with the exam shown whole real frames; the disc lives also on their own world; at 60k the exam shown as the life saw frames, and the free dream of 200 ticks.

## Readings, set before the results

One seed an arm: a look. A difference counts only if it is outside what two runs of one seed differ by at 60k (2 ticks of the ball's horizon, .06 of the things held at a tick, .025 of next-frame skill; vm-00122).

1. **More prediction (.7 against .5):** are the things held longer in the dream, and what does the next frame pay?
2. **Random grey (.5 with .2 grey against .5):** the same two questions. Does noise that does not shrink do more than noise that does?
3. **Shown whole frames or frames as lived:** a model that lived with random grey may need its grey to see as it learnt to.
4. **The long dream:** is a ball in the court longer, is one ever served?

**What I expect** (written before any life of this experiment ran, with vm-00123's court life scored only at 10k):

- .7: the dream a little better than at .5 and the next frame a little worse; the differences small, because late in life the prediction is nearly the real frame at either share.
- Random grey: next-frame skill shown whole frames clearly lower than without it; the dream no better than at .5. A random grey is not an error the model makes, so learning to see through it does not teach it to mend its own dream.
- Neither serves a ball in the long dream.

## Results

None yet: the lives are running.

## Files

- `Makefile`: `runs`, `score`, `home`, `lived`, `free`, `curves`, `images`, `videos-long`.
