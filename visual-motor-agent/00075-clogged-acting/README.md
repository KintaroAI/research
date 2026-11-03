# vm-00075: Clogged Acting

**Date:** 2026-11-02
**Status:** Running: started 2026-11-02 19:58. No readings yet
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00075/`
**Task plan:** [`../tasks/00086-clogged-acting.md`](../tasks/00086-clogged-acting.md)

## Goal

vm-00074 with the damage in real life: each transformer also acts on its own damaged view, while memory stays clean and training draws fresh damage. Design, arms and the readings set before the results are in the task plan.

## How the damage works here

The full mechanism is in [`ARCHITECTURE.md`, section 14](../ARCHITECTURE.md). What this experiment changes:

| | vm-00074 (C10, C30) | here (A10, A30) |
|---|---|---|
| Acting during a life | clean | **damaged, each transformer its own** |
| A transformer's context | clean | **damaged** |
| Memory | clean | clean |
| Training input | fresh damage | fresh damage |
| Training targets | clean | clean |

The agent lives through the damage rather than only practising on it. The world, food and hunger are as they were: a transformer that cannot see a food circle can still touch it.

## Method

- **Arms:** A10 and A30, K4 with log summaries, 64 x 64, `online.clog` .1 and .3, `online.clog_act = 1`, six seeds, 60k ticks, seeds 1–3 on the 4090 and 4–6 on the 3090, all 12 at once.
- **Comparison:** C10 and C30 of vm-00074 and the K4 of vm-00073, not rerun.
- **Readings:** set before the results, in the task plan.
- **Checks passed before the start:** 10 unit tests, the views of the transformers different on every one of 60 ticks on the GPU, a short life as fast as one without it, and a life refuses `clog_act` without `clog`.
