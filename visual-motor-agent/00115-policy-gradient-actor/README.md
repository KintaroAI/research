# vm-00115: The Move Token as a Policy-Gradient Actor

**Date:** 2026-11-25
**Status:** Complete (negative; cut to seeds 1–6 with no frozen tests, the owner, 2026-11-25)
**Source:** `dev/` (pending tag `exp/vm-00115`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00115/`
**Task plan:** [`../tasks/00128-policy-gradient-actor.md`](../tasks/00128-policy-gradient-actor.md) (option (c) of task 00123, the acting rule)

## Goal

Task 00123 asks whether the acting rule explains Dreamer's lead in Breakout (.99 against the default agent's .65 at 50–60k, vm-00108). The default agent picks the best of 16 sampled moves by their predicted label, and its move token imitates the picks. Dreamer trains an actor by the gradient of its critic's value. Does our same transformer act better when its move token is trained as an actor by policy gradient, with our label as the critic and no instinct?

## Method

- **The change** (`online.actor_pg = 1`):
  - **The move made** is one sample of the move token's own distribution at temperature 1. There is no instinct. Of the four transformers, a random one's move is made (`online.piece_rule = random`).
  - **The critic** is the label, unchanged (`body_label`, `label = meal`, γ .95, refreshed).
  - **The advantage,** per transformer when it acts: the mean predicted label of 16 moves sampled from its move token, minus the predicted label of the move made (the label is a cost, so lower is better).
  - **The actor's loss:** the move token's cross-entropy on the move made, weighted by clip(A / s, ±3), with s the running root mean square of A, plus an entropy bonus (`online.pg_ent` .03). A positive weight pulls the move token toward the move, a negative one pushes it away.
  - **Only the newest window carries the weight** (`online.pg_replay` 0). Replayed windows have no move loss.
  - Everything else is the default agent's: the same transformers, tokens, targets, memory and one update per tick.
- **Arms:** PB in Breakout stage 1 (`configs/U_breakout.json` with the change) against BD (vm-00103); PG in the food world (`configs/U_agent.json` with the change) against TT (vm-00101). 60k ticks, paired by seed.
- **Pilots** (food world; details in the task plan):
  - With every window weighted, the move token's cross-entropy on replayed moves rose to 25–40 nats: a move with a negative advantage stays in memory and is pushed away again at every draw. Hence `pg_replay` 0.
  - `pg_ent` .01, .03, .1 and .3 over 6,000 ticks left the move token's entropy at .3, 1.8, 4.8 and 6.6 nats (uniform is 6.9).
  - `pg_ent` .03 and .1, three seeds, 30k ticks: neither foraged (10.3 and 9.4 food per 1k at 20–30k; TT 57.1). `.03` was taken by the rule stated beforehand (more food, less poison), a difference within noise.
- **Cut by the owner** (2026-11-25, with PB at 47k ticks and PG at 26k): seeds 1–6 only and no frozen tests. "Twelve seeds were meant to resolve small differences, and this one is the whole effect in every seed." PB ran on the RTX 4090 and PG on the RTX 3090, six lives per card.

## Readings (set before the results; 12 seeds planned, 6 run)

1. **Breakout:** PB − BD hit share at 40–60k above zero; ticks to .5.
2. **Food world:** PG − TT poison and food per 1k at 40–60k (no direction predicted: Dreamer's actor eats poison, vm-00112).
3. **Frozen** at 60k agrees in sign with 1 and 2. *Not run (cut).*
4. **Descriptive:** the move token's entropy over the life; the share of positive weights.

## Results

Seeds 1–6, paired by seed; ± is the 95% interval. The tables are `make table`.

### R1. Breakout: PB never leaves chance

| Ticks | BD | PB | PB − BD (95%) | Seeds |
|---|---|---|---|---|
| 0–10k | .229 | .221 | −.007 ± .043 | 3 of 6 above |
| 10–20k | .256 | .226 | −.030 ± .056 | 3 of 6 above |
| 20–30k | .325 | .228 | −.097 ± .041 | 0 of 6 above |
| 30–40k | .434 | .225 | −.209 ± .071 | 0 of 6 above |
| 40–50k | .572 | .220 | −.352 ± .106 | 0 of 6 above |
| 50–60k | .652 | .215 | −.436 ± .046 | 0 of 6 above |

- **Reading 1, at 40–60k: PB − BD = −.394 ± .075,** 0 of 6 seeds above.
- PB's hit share is .215–.228 in every 10k window, at or just under the random paddle's .235 (vm-00103).
- No PB seed reaches .5 in any 5k window. The best 5k window of each life is .28–.34. BD's six seeds reach .5 at 32–47k (median 40.5k).

### R2. The food world: PG does not forage

| Ticks | TT food | PG food | PG − TT food (95%) | Seeds | TT poison | PG poison | PG − TT poison (95%) | Seeds |
|---|---|---|---|---|---|---|---|---|
| 0–10k | 12.3 | 10.0 | −2.4 ± 2.3 | 0 of 6 above | 24.3 | 30.4 | +6.1 ± 2.4 | 6 of 6 above |
| 10–20k | 32.5 | 9.6 | −22.9 ± 8.6 | 0 of 6 above | 17.6 | 31.7 | +14.2 ± 3.5 | 6 of 6 above |
| 20–30k | 58.2 | 11.5 | −46.7 ± 6.2 | 0 of 6 above | 12.7 | 29.1 | +16.4 ± 5.4 | 6 of 6 above |
| 30–40k | 68.0 | 11.7 | −56.3 ± 2.8 | 0 of 6 above | 9.7 | 28.8 | +19.2 ± 5.2 | 6 of 6 above |
| 40–50k | 71.5 | 11.6 | −59.8 ± 3.2 | 0 of 6 above | 8.9 | 28.3 | +19.5 ± 3.9 | 6 of 6 above |
| 50–60k | 71.9 | 12.0 | −59.9 ± 3.8 | 0 of 6 above | 7.9 | 27.1 | +19.1 ± 6.8 | 6 of 6 above |

- **Reading 2, at 40–60k: food −59.9 ± 3.2 per 1k** (0 of 6 seeds above) **and poison +19.3 ± 5.3** (6 of 6 above).
- PG's food stays at its starting level for the whole life (10.0 at 0–10k, 12.0 at 50–60k) while TT's rises from 12.3 to 71.9. Its poison stays at 27–32 per 1k while TT's falls from 24.3 to 7.9.

### R3. The actor narrows to a near-fixed move within 20k ticks

Means of the logged rows and of the six seeds; the entropy's range across seeds in brackets. Uniform is 6.9 nats (the move's dx and dy, 32 bins each).

| Measure | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| PB: the move token's entropy, nats | 2.64 (1.58–3.43) | 0.67 (0.39–0.93) | 0.20 (0.10–0.36) | 0.09 (0.03–0.26) | 0.04 (0.01–0.08) | 0.03 (0.01–0.06) |
| PG: the move token's entropy, nats | 1.93 (1.66–2.42) | 0.59 (0.44–0.80) | 0.26 (0.08–0.32) | 0.13 (0.06–0.25) | 0.07 (0.03–0.12) | 0.04 (0.02–0.09) |
| PB: share of positive weights | .47 | .42 | .38 | .37 | .36 | .35 |
| PG: share of positive weights | .48 | .46 | .44 | .42 | .42 | .41 |
| PB: mean size of the advantage | .0009 | .0005 | .0005 | .0005 | .0006 | .0008 |
| PG: mean size of the advantage | .0044 | .0055 | .0053 | .0049 | .0048 | .0048 |
| PB: the move token's cross-entropy on the batch's moves | 25.8 | 37.9 | 43.0 | 41.1 | 41.9 | 42.2 |
| BD: the same | 0.72 | 0.55 | 0.53 | 0.53 | 0.56 | 0.60 |
| PG: the move token's cross-entropy on the batch's moves | 28.2 | 38.4 | 40.2 | 40.2 | 39.6 | 39.3 |
| TT: the same | 0.41 | 0.32 | 0.33 | 0.37 | 0.42 | 0.46 |
| PB: share of ticks at the hunger cap | .52 | .52 | .50 | .50 | .52 | .52 |
| BD: the same | .50 | .46 | .35 | .22 | .12 | .07 |
| PG: share of ticks at the hunger cap | .57 | .62 | .51 | .53 | .52 | .50 |
| TT: the same | .44 | .14 | .02 | .01 | .01 | .01 |

- **The entropy falls fast and early** (reading 4). In the rows at 1k, 2k, 5k and 10k ticks it is 4.75, 3.50, 2.54 and 1.30 nats in PB and 4.28, 3.00, 1.73 and 1.00 in PG. By 20–30k it is .2–.3, and by 50–60k .03–.04: each transformer proposes practically one move.
- **The share of positive weights falls below one half,** from .47–.48 to .35 (PB) and .41 (PG).
- **The advantage is small from the start.** In PB its mean size is .0005–.0009 through the whole life, in PG .0044–.0055. For scale, the label's spread across BD's 16 candidates is .030–.044 in a ball's last 8 ticks (vm-00116).
- **The moves in memory become improbable:** the move token's cross-entropy on the batch's moves (the newest window and the seven replayed, reported unweighted) is 26–43 nats, against .5–.7 for BD and .3–.5 for TT. The moves the agent made earlier in its life have almost no probability left under its move token.
- **Both arms starve:** half of all ticks are at the hunger cap through the whole life (.50–.62), where BD falls to .07 and TT to .01.
- **Cost:** PB ran at 9.81 ticks per second, six lives on the RTX 4090, as BD's seeds 7–12 did there (9.84). The 16 moves are still sampled and scored for the baseline.

## Analysis

1. **As built, the policy-gradient actor does not learn to act in either world.** Reading 1 has the opposite sign to the one hoped for (−.394 ± .075). Reading 2 is a loss on both counts (−59.9 food, +19.3 poison per 1k). On the same seeds, with the same model, label, memory and training, the instinct's choice learns both worlds (BD .65 hit share, TT 71.9 food per 1k at 50–60k).
2. **What fails is the actor, and early.** Its entropy is at 1.0–1.3 nats by 10k ticks, when the instinct's lives have barely begun to learn (BD .229 hit share and TT 12.3 food per 1k at 0–10k). So the move token narrows on the verdicts of a label that does not yet know the world.
3. **The weights are full size however little the label says.** The advantage is divided by its own running root mean square, so differences of .0005–.005 between moves, real or noise, push the move token as hard as large ones would. Once it has narrowed, each transformer's 16 baseline samples are practically one move, and the experience in memory holds little more than one move per transformer for the label to learn from.
4. **Why the instinct survives the same critic** (an interpretation, not tested here). vm-00116 found that BD's label orders two candidate moves correctly only .60–.62 of the time near a landing. The instinct takes the best of 16 fresh samples at every tick and commits to nothing: a wrong choice costs one tick, and its candidates keep their variety (14 distinct moves of 16 in BD, vm-00110). A policy gradient adds each tick's noisy verdict to the weights, where it stays.
5. **For task 00123:** this closes option (c) in the form tested: REINFORCE on real windows, one sample per tick, with the one-step label as the critic. It does not show that a trained actor cannot work on this model. Dreamer's actor learns from many imagined rollouts per update, which our model cannot yet supply (its dream is noise by 4 ticks, vm-00114).

## Not done

- Seeds 7–12 and the frozen tests at 30k and 60k (cut by the owner).
- A larger entropy bonus over a full life. In the pilots .1 did not forage in 30k ticks and .3 stayed near uniform at 6k.
- Advantages recomputed for replayed windows with the current critic; PPO's ratio clipping (no old log-probabilities are stored); dreamed rollouts for the actor.
- The instinct acting on PB's and PG's final weights, which would say whether their labels learned anything.

## Files

- `dev/src/vm_cuda.cu`, `dev/src/vm_cuda.h`: `vmc_set_move_weight`; the loss kernel's weighted move row and entropy bonus.
- `dev/cuda_lib.py`: `CudaModel.set_move_weight`, `update_rollout(..., move_weight=)`.
- `dev/reference/torch_reference.py`: `batch["move_weight"]`, `batch["move_ent"]`.
- `dev/online.py`: `uni_act(..., pg=True)`.
- `dev/lifetime.py`: `online.actor_pg`, `pg_ent`, `pg_clip`, `pg_decay`, `pg_replay`; the memory field `adv`; the metrics `pg_abs`, `pg_pos`, `pg_scale`, `move_ent`.
- `dev/tests/test_policy_gradient.py` (17 tests).
- `dev/analyze_lives.py pg`: the tables above.
