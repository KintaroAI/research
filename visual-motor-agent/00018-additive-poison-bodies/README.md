# vm-00018: Additive Poison, and One Model in Many Worlds

**Date:** 2026-10-25
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00018`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00018/`
**Task plan:** [`../tasks/00022-additive-poison-bodies.md`](../tasks/00022-additive-poison-bodies.md)

## Goal

Two follow-ups to vm-00017, both about whether the prediction works:

1. **Poison that adds** +0.5 to hunger instead of setting it to 1, so that hunger rarely sits at its cap, where a
   poison click leaves no trace.
2. **One model in many worlds:** 32 bodies early in life (vm-00016's best basic agent) give the model many more poison
   and food clicks to learn from. A one-body arm with the same batch size separates more data from bigger batches.

## Hypothesis

1. With additive poison and 100k ticks, the 8×8 patch model's drives head predicts the poison click (+0.5) as well as
   the food click, and stops predicting relief for a click on nothing.
2. Parallel bodies learn the click outcomes sooner (by 10k–20k ticks).
3. The gain comes from the extra experience, not the batch size: Q1B (one body, batch 39) learns no faster than Q1.

## Method

- **World:** vm-00017's, with `env.hazard_click_add=0.5` instead of `env.hazard_click=1` (task 00022).
- **Model:** vm-00017's PP (8×8 patch tokens, move loss detached, hidden policy layer, memory U), `lifetime.py`,
  100k ticks, seeds 1–3, on both GPUs.
- **Arms:** Q1 (one body, batch 8), Q32 (32 bodies for 10k ticks, halving every 5k, one from tick 30k; batch 39 → 8),
  Q1B (one body, batch 39 throughout).
- **Measures:** `analyze_hazards.py click` (world: click minus stay −0.301 food, +0.5 poison, 0 background; the eaten
  circle's redness) and the frozen tests at 0, 10k, 20k, 40k, 60k, 100k; online food hits and poison clicks (body 0, all
  bodies) and hunger at the cap.

## Log

### 2026-10-25
- **Implemented** `env.hazard_click_add` (test: adds after relief, clips at 1, off by default) and poison-click counts
  in `online.py` / `lifetime.py`. Smoke run (8 bodies, 1k ticks): about 10 poison clicks per 1,000 body-steps; the
  click test's truth for poison is 0.801 (0.3 + 0.5 + 0.001).- **Launched** at 14:33; all 9 finished at about 15:33 (100k ticks). The click tests ran at 15:33–15:44. The frozen
  tests of 8 runs failed (the GPU wrapper asked the installed library for a function only the new universal-model
  build has, task 00023); both wrappers now skip missing functions, and the tests were rerun at 16:10.

## Results

### R1. The click test: predicted next hunger, click minus stay (world: food −0.301, poison +0.500, nothing 0)

| Arm (seeds 1, 2, 3) | 10k | 40k | 100k |
|---|---|---|---|
| Q1 (one body, batch 8) | food −0.18, −0.01, −0.11; poison −0.17, −0.01, −0.10; nothing −0.17, −0.01, −0.09 | food −0.12, −0.22, −0.12; poison −0.06, −0.12, +0.09; nothing −0.10, −0.12, −0.11 | food −0.22, −0.25, −0.23; poison **+0.25, +0.23, +0.30**; nothing −0.08, −0.01, −0.00 |
| Q32 (32 bodies → 1) | food −0.17, −0.26, −0.16; poison −0.15, −0.22, −0.15; nothing −0.15, −0.20, −0.15 | food −0.27, −0.15, −0.15; poison **+0.07, +0.33, +0.21**; nothing −0.10, −0.00, −0.02 | food −0.20, −0.27, −0.23; poison +0.37, +0.19, +0.28; nothing −0.00, −0.11, −0.01 |
| Q1B (one body, batch 39) | food −0.03, −0.01, +0.02; poison −0.01, +0.05, +0.02; nothing ≈ 0 | food −0.01, −0.08, +0.05; poison +0.19, +0.15, +0.13; nothing ≈ 0 | food −0.17, −0.17, −0.08; poison +0.30, +0.29, +0.32; nothing −0.02, −0.01, −0.02 |

### R2. Behaviour: the frozen tests (episodic success / forage hits per 1k / forage poison clicks per 1k)

| Arm | birth | 20k | 100k |
|---|---|---|---|
| Q1 | 0.59–0.62 / 4.3–5.0 | 0.87, 0.79, 0.74 / 8.5, 7.9, 7.4 | 0.84, 0.84, 0.78 / 9.0, 8.8, 8.4 / —, 10.2, 10.5 |
| Q32 | 0.59–0.62 | 0.64, 0.59, 0.59 / 5.0, 4.6, 4.4 | **0.68, 0.67, 0.62** / 5.6, 5.0, 4.9 / 7.1, 8.0, 7.4 |
| Q1B | 0.59–0.62 | 0.87, 0.88, 0.86 / 10.2, 9.1, 8.2 | **0.92, 0.87, 0.87** / 12.0, 9.2, 8.7 / 16.6, 11.5, 12.8 |

Online, last 10k: food hits Q1 7.5–9.2, Q32 5.3–5.8, Q1B 8.1–11.0 per 1k; poison clicks 6–21 per 1k; hunger at the cap
0.21–0.48 of steps.

## Analysis

- **The prediction works by 100k in every arm** (hypothesis 1, mostly supported): food −0.08 to −0.27 of −0.30,
  poison +0.19 to +0.37 of +0.50, and a click on nothing back near 0 (0.00 to −0.11). Early on (10k–20k) every model
  first learns "a click lowers hunger" wherever it is; the location (food, poison, nothing) separates later.
- **Many bodies learn the click outcomes sooner** (hypothesis 2, supported for prediction): at 40k Q32 separates
  poison in all three seeds (+0.07, +0.33, +0.21) and a click on nothing is near 0 in two, where Q1 still predicts
  relief for poison in two seeds. Batch size alone (Q1B) separates poison early too, but food more slowly.
- **But the 32-body arm's behaviour never improves** (unexpected; the opposite of vm-00016): its test success stays
  near birth (0.62–0.68 at 100k against 0.59–0.62), while Q1 reaches 0.78–0.84 and Q1B 0.87–0.92. vm-00016's 32-body
  arm (ASB) was the best agent, but it also had the act token and the signed command. Here the policy reads the move
  token, which with the move loss detached is trained by nothing directly (vm-00010).
- **Why (checked after the runs; no code defect found):** Q32's policy never leaves its starting point: its click
  probability stays at 0.10 (the initial bias and the babbling rate) and its spread at 0.61 (the babbling σ) for all
  100k ticks, while Q1's and Q1B's click rates rise to 0.14–0.21; its gradient norm is 5× smaller than Q1B's at the
  same batch size (0.2–0.4 against 2.1–2.7). With 32 bodies, 32 of the 39 windows per update are the newest, made of
  the policy's own current samples, and the gradient of the log-likelihood of one's own samples is zero in
  expectation; the rest of the move targets are babbling, which pull it toward the babbling distribution. Only the
  hindsight command carries information the policy did not have when it acted. vm-00016's 32-body arm froze the same
  way for its first 5k ticks (click 0.10, gradient 0.19) and then learned (click 0.20 by 10k): it used the signed
  command (a label at every step), where vm-00018 uses the relief command (zero except shortly before food). So: the
  same code, a mechanism that only sometimes escapes; not tested directly.
- **Bigger batches help behaviour** (hypothesis 3, the opposite): Q1B, one body with 39 windows per update, is the best
  agent here (test 0.87–0.92, forage 8.7–12.0 per 1k). It also clicks poison most (11.5–16.6 per 1k): it clicks more.
- **The cap is not gone:** an additive poison of 0.5 still parks hunger at 1 on 21–48% of steps.

## Conclusions

1. **With additive poison and 100k ticks, the patch model's prediction gets all three click outcomes right**, after
   first over-generalizing "a click relieves".
2. **Many bodies speed up learning the predictions, not the behaviour**; bigger batches (one body) gave the best
   behaviour. With 32 bodies most of each batch is the policy's own fresh samples, whose imitation gradient is zero on
   average, so the policy stays where it started unless the command carries enough (§Analysis).
3. **0.5 is still too much poison** for slow hunger to stay off the cap.

## Next Steps

- Keep the share of fresh windows low with many bodies (e.g. 1 fresh in 8, as Q1: replay 7 windows per body), or
  test the signed command with 32 bodies in this world.
- Poison of 0.2–0.3, so that hunger rarely reaches the cap.
- The universal model (vm-00019) in this world.

