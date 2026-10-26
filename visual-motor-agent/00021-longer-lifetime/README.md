# vm-00021: A Longer Lifetime for the Universal Model

**Date:** 2026-10-25
**Status:** In Progress
**Source:** *tagged on completion as `exp/vm-00021`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00021/`
**Task plan:** [`../tasks/00025-longer-lifetime.md`](../tasks/00025-longer-lifetime.md)

## Goal

vm-00019/20's universal model learned what a click does in only one seed in three by 60k ticks, and the instinct helped
only there. Here the default universal model lives 200k ticks, six seeds.

## Hypothesis

1. Given time, every seed learns the three click outcomes (food −0.30, poison +0.50, nothing 0).
2. The instinct then lowers poison clicks in every seed.
3. The policy learns the instinct's choices: the policy alone improves too.
4. Exploration holds (the instinct keeps the model clicking).

## Method

`configs/U_universal.json` (gray, checkerboard poison +0.5, 8 moving dots, slow hunger, continual torus; the universal
model, 8×8 patches, 32 bins, gray; the instinct with 16 candidates; no babbling, no command; memory U), 200k ticks,
seeds 1–6 (1–3 on the RTX 4090, 4–6 on the RTX 3090). Measures in task 00025.

## Log

### 2026-10-25
- **Launched** the 6 lifetimes at 19:49: 77 ticks/s each on the RTX 4090, 47 on the RTX 3090.
