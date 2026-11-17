# vm-00096: The Instinct's Horizon in the World With Inertia

**Date:** 2026-11-17
**Status:** Running
**Source:** `dev/` (pending tag `exp/vm-00096`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00096/`
**Task plan:** [`../tasks/00107-instinct-horizon-inertia.md`](../tasks/00107-instinct-horizon-inertia.md)

## Goal

In vm-00095 the default agent learned the world with inertia poorly from birth: .41 of its food without inertia. Its instinct compares 16 candidate moves one push ahead, and a push changes the body's next position by at most 1 px per axis. Here each candidate is held for 4 ticks, the ticks after the first imagined by the model, so the instinct compares courses the body can actually take. Does that let the agent learn this world? The design and the readings set before the results are in the task plan.

## Method

| Arm | Ring | Instinct's horizon | Seeds |
|---|---|---|---|
| IB (vm-00095, not rerun) | yes | 1 | 1–12 |
| IR | no | 1 | 1–12 |
| IH | no | 4 (one Jacobi pass a step, pair decoding) | 1–12 |

- **All in the world with inertia from birth** (`env.inertia = 4`), with C8's settings otherwise.
- **The horizon** (`online.instinct_h = 4`) needs acting on full windows (`online.ring = 0`). IR is the same agent without the ring and with the instinct as before, so IH − IR isolates the horizon.
- **The lookahead for the quarter transformers** (task 00107): each piece predicts only its own block, so its imagined frames are the current frame with that block replaced.
- **Cards:** seeds 1–8 of both arms on the RTX 4090, 9–12 on the RTX 3090, under MPS.

## Results

Running.
