# Task 00128: The Move Token as a Policy-Gradient Actor (experiment vm-00115; option (c) of task 00123)

**Date:** 2026-11-25
**Status:** Running (runs started 2026-11-25). The owner, 2026-11-25, after vm-00114 closed negative: "I think we need to try (c)"; on the version: real windows (no dreaming).

## Context

- **Task 00123** asks whether the acting rule explains Dreamer's lead in Breakout (.99 against BD's .65 at 50–60k, vm-00108): our transformer and world model with a trained actor in place of the label-and-instinct choice.
- **Our acting rule now:** the move token predicts a distribution over move bins; the instinct samples 16 candidates from it (after the first at `instinct_temp` 2), scores each by its predicted label (`uni_candidates`, γ .95), makes the best one, and the move token learns by plain cross-entropy on the move made, so it imitates the instinct's picks. Across the 4 quarter pieces, `piece_rule = gain` makes the move of the piece whose pick scores best.
- **Dreamer's rule:** an actor trained by the gradient of a critic's value, through imagined steps (backprop through the model for continuous actions, REINFORCE for discrete).
- **What carries over:** our moves and frames are binned tokens, so there is no gradient through the model; vm-00114 found the model's own dream noisy by 3–4 imagined ticks. The version tested here is REINFORCE on real windows, with our label as the critic.

## The change: arm PG (`online.actor_pg = 1`)

- **Actor:** the move token's own distribution. The move made is one sample of it at temperature 1, no override.
- **Pieces:** `online.piece_rule = random` (the move of a random piece: one sample of the pieces' mixture; `gain` would be a 4-way choice by the critic, a small instinct).
- **Critic:** the label as now (`body_label`, `label = meal`, γ .95, stored and refreshed), unchanged.
- **Advantage at act time, per transformer k:** each samples 16 moves from its own move prediction (temperature 1) and scores them by its predicted label; the baseline b_k is their mean. With the label T (lower is better), A_k = b_k − T_k(move made), T_k(move made) being what `slabel` already computes. Stored with the transition (memory field `adv`, one per transformer).
- **Actor loss:** the move token's cross-entropy on the stored move, weighted per step by w = clip(A / s, −c, c), with s the running root mean square of A (EMA, `online.pg_decay` .999) and c = `online.pg_clip` (3); plus an entropy bonus on the move token's bins, `online.pg_ent` (.03 in the runs, chosen on the pilot) per value. w > 0 pulls the move token toward the move, w < 0 pushes it away. Only the newest windows (one per body) carry it (`online.pg_replay` 0, after pilot 1); replayed windows have no move loss.
- Nothing else changes: the same transformer, the same tokens and targets for every other token, the same replay.
- **One rule for every world;** nothing reads the world's state.

## Code

- **CUDA** (`src/vm_cuda.cu`, `vm_cuda.h`): `vmc_set_move_weight(h, S, w [S, ctx], ent)`: the move token's loss rows (the token before the move, `k = nt − 2`) get their two-hot cross-entropy gradient times w of their step, plus ent · ∂(Σ p log p)/∂logit over the used move values, at the same per-token scale; cleared by the next `set_batch`. `cuda_lib.CudaModel.set_move_weight`; `update_rollout(..., move_weight=)`.
- **Reference** (`reference/torch_reference.py` `uni_loss`): `batch["move_weight"]` [S, T] and `batch["move_ent"]`.
- **Acting** (`online.uni_act(..., pg=True)`): candidates at temperature 1, the move made is candidate 0, returns each body's baseline (mean label over the candidates).
- **lifetime.py:** `online.actor_pg`, `pg_ent`, `pg_clip`, `pg_decay`; the memory field `adv` [cap, S]; the weights built per transformer from the rows; refused without the GPU, without the stored meal label, with `instinct_h > 1`, with `state`; logs the advantage scale, the share of positive weights and the move token's entropy.
- **Tests** (`tests/test_policy_gradient.py`): CUDA against the reference (weights only, entropy only, both; plain and piece configs); w = 1 and ent = 0 is the old loss exactly; the next batch clears it; `uni_act(pg=True)` makes candidate 0 and returns the mean; a short life runs with `actor_pg`.

## Runs

- **Food world:** arm PG (`configs/U_agent.json` + the change), seeds 1–12, against TT (vm-00101). **Breakout stage 1:** arm PB (`configs/U_breakout.json` + the change), seeds 1–12, against BD (vm-00103). 60k ticks, paired by seed; then frozen at 30k and 60k.
- First a pilot: 1 seed per world for 10k ticks, to check the move token's entropy does not collapse or stay uniform and the cost per tick. If the entropy collapses or stays flat, `pg_ent` is tuned on the pilot only and stated.
- GPUs: 6 seeds per card at a time.
- **Cut to seeds 1–6, no frozen tests** (the owner, 2026-11-25 20:57: "lets cancel seeds 7–12 and the frozen tests - twelve seeds were meant to resolve small differences, and this one is the whole effect in every seed"). At that point PB was at 47k with hit share .11–.28 over the last 5k in every seed (random .235, BD .60 at 40–50k) and the move token's entropy at 0.00–0.06; PG at 26k ate 5–15 food per 1k in every seed (TT about 50). Readings 1 and 2 are read on 6 seeds; reading 3 (frozen) is dropped.

## Readings, set before the results (12 seeds, 40–60k)

1. **Breakout:** PB − BD hit share above zero; ticks to .5.
2. **Food world:** PG − TT poison per 1k and food per 1k (no direction predicted: vm-00112 found Dreamer's actor eats poison).
3. **Frozen** at 60k agrees in sign with 1 and 2.
4. **Descriptive:** the move token's entropy over the life; the share of positive weights.

## Pilot (food world, seed 1)

- **Pilot 1 (every window weighted, `pg_ent` .01 to 1, 3,000 ticks):** the move token's entropy fell from 6.3 to 1–2 within 1,000 ticks at `pg_ent` ≤ .3. At 1.0 it stayed near uniform (5.9 of 6.9), but in every variant the move cross-entropy on the replayed moves rose to 25–40: a move with a negative advantage stays in long-term memory and is pushed away again at every draw, so its bins go to zero without limit (REINFORCE assumes each sample used about once). Food stayed at chance.
- **Change:** `online.pg_replay` (default 0): only the newest windows (one per body) carry the policy-gradient weight; replayed windows have no move loss. Each step is then weighted at most ctx (8) times as the window slides.
- **Pilot 2:** `pg_replay` 0, `pg_ent` .01, .03, .1, .3, 6,000 ticks: the move token's entropy at 1k and 6k: .01 1.0 to 0.3; .03 3.6 to 1.8; .1 6.3 to 4.8; .3 6.8 to 6.6 (uniform 6.9). Food at chance in all four, as TT's is this early.
- **Pilot 3 (the choice of `pg_ent`):** .03 and .1, seeds 1–3, 30k ticks (`make pilot`). Food and poison per 1k at 20–30k (mean of seeds 1–3):

  | | food | poison | move entropy at 30k |
  |---|---|---|---|
  | TT (vm-00101) | 57.1 | 13.0 | |
  | PG `pg_ent` .03 | 10.3 | 27.1 | 0.2 |
  | PG `pg_ent` .1 | 9.4 | 31.7 | 0.9 |

  Neither learns to forage in 30k: food stays at its 0–5k level (about 10, chance) while TT's rises from 9.5 to 60.6; the move token's entropy falls to a near-fixed move. The share of positive advantages stays .43–.50. Checked for a bug before the runs: the sign (A = baseline − the move's label, lower label better), the step of each weight (the reference and the kernel agree) and that the newest windows are the weighted rows; none found. **`pg_ent` .03** by the stated rule (more food, less poison at 20–30k), a difference within noise. The 12-seed runs go ahead as planned: the pilot is 3 seeds in one world, and Breakout is the reading task 00123 asks about.

## Not here

- Dreamed rollouts for the actor (only once the dream holds at 4+ ticks; vm-00114).
- Recomputing replayed advantages with the current critic.
- PPO's ratio clipping (no old log-probabilities are stored).

## Implementation

- [x] CUDA `vmc_set_move_weight`, the loss kernel; `cuda_lib`; the reference
- [x] `uni_act(pg=True)`; `lifetime.py` and the memory field
- [x] Tests (`tests/test_policy_gradient.py`, 17)
- [x] Pilot
- [ ] `00115-policy-gradient-actor/` Makefile, README; runs
