# vm-00117: Two Worlds, One Set of Weights

**Date:** 2026-11-25
**Status:** Running (queued 2026-11-25 on both cards)
**Source:** `dev/` (in progress)
**Outputs:** `~/data/research/visual-motor-agent/exp_00117/`
**Task plan:** [`../tasks/00130-two-worlds.md`](../tasks/00130-two-worlds.md)

## Goal

The first challenge of `ARCHITECTURE.md` section 13: "universal" has meant the same code in every world, each world with its own life. Here one set of weights meets a second world. Does a trained agent learn the other world faster or slower than a new one, how much of its first world is left afterwards, and how fast does it come back?

## Method

The default agent's two configs differ only in the world, so the weights of a life in one load in the other (`lifetime.py --init`). No code changes.

- **TB:** Breakout stage 1 from TT's (vm-00101) 60k weights, 60k ticks, against BD (vm-00103) from scratch, paired by seed.
- **BF:** the food world from BD's 60k weights, 60k ticks, against TT from scratch, paired by seed.
- **The old world, learning off,** from the second life's checkpoints at 10k, 30k and 60k: how much is left, against the first life's own frozen score and an untrained model's.
- **TBF, BFB:** a third life of 30k ticks back in the old world: how fast it comes back, against the first life's first 30k ticks.
- **Seeds 1–12;** seeds 1–6 on the RTX 3090 and 7–12 on the RTX 4090.

A life started from weights has an empty memory, so the old world is never replayed: learning one world after another without rehearsal, the harsh case.

The readings, set before the results, are in the task plan.

## Results

Not in yet.
