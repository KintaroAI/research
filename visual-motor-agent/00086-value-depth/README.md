# vm-00086: Eight Layers in Each Quadrant, With and Without the Value

**Date:** 2026-11-11
**Status:** Queued (starts on the RTX 4090 once 00021's seed-1 runs there end)
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00086/`
**Task plan:** [`../tasks/00097-value-depth.md`](../tasks/00097-value-depth.md)

## Goal

Two earlier results point the same way:
- In vm-00084, eight layers made a single transformer forage better than two, in all six seeds.
- In vm-00085, a bootstrapped value (γ 0.9) made the four-quadrant default aim further, but its estimate was badly optimistic.

The owner asked for both together. Here each of the default's four transformers is made 8 layers deep, with K4's label and with vm-00085's value. With vm-00085's 2-layer arms this gives a 2 × 2 of label and depth. The design and the readings set before the results are in the task plan.

## Method

| Arm | Label | Layers per transformer | Parameters per transformer |
|---|---|---|---|
| K4 (vm-00085, not rerun) | mean hunger over the next 4 ticks | 2 | 969,506 |
| V90 (vm-00085, not rerun) | value, γ 0.9, n 4 | 2 | 969,506 |
| K4D8 | mean hunger over the next 4 ticks | 8 | 2,159,138 |
| V90D8 | value, γ 0.9, n 4 | 8 | 2,159,138 |

- **Depth is the only change:** `model.n_layer = 8`, as vm-00084's D8 (d_model 128, 4 heads, d_ff 512).
- **Everything else is the default:** four quadrant transformers, gain, log summaries, the ring, the egocentric touch world, `online.future_k = 4`, six seeds, 60k ticks.
- **The value is vm-00085's (phase 1):** the winner's prediction, stored while acting.
- **Where it runs:** the 12 new lives run on the RTX 4090 under MPS. vm-00085's lives ran on the 3090, and GPU lives are not bitwise reproducible, so arms are compared over seeds.
- **Frozen tests:** `eval_hunger.py chain` and `eval_hunger.py value`, as in vm-00085.
