# vm-00088: The Fresh Value at Eight Layers, Twelve Seeds

**Date:** 2026-11-12
**Status:** Running (started 2026-11-12)
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00088/`
**Task plan:** [`../tasks/00099-value-refresh-depth.md`](../tasks/00099-value-refresh-depth.md)

## Goal

- **vm-00086:** eight layers in each quadrant helped K4 in every lived measure. The old value at eight layers stayed optimistic, and its candidates collapsed.
- **vm-00087:** a tail recomputed at every update, as the mean of the four transformers, made the value calibrated, but at two layers it added no food beyond its interval.

Here the fresh value runs at eight layers, against K4 at eight layers, over twelve seeds, because six were too few to resolve differences of this size. The design and the readings set before the results are in the task plan.

## Method

| Arm | Label | Tail | Layers | Seeds |
|---|---|---|---|---|
| K4D8 | mean hunger over the next 4 ticks | none | 8 | 1–6 from vm-00086 (RTX 4090), 7–12 new (RTX 3090) |
| V90RD8 | value, γ 0.9, n 4 | the four's mean prediction of the move made, fresh at every update | 8 | 1–6 on the RTX 4090, 7–12 on the RTX 3090 |

- **Every seed pair runs on one card:** seeds 1–6 on the 4090, seeds 7–12 on the 3090, all under MPS.
- **Everything else is the default:** four quadrant transformers, gain, log summaries, the ring, the egocentric touch world, `online.future_k = 4`, 60k ticks.
- **Frozen tests:** `eval_hunger.py chain` and `eval_hunger.py value`, as in vm-00085–00087.
