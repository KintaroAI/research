# vm-00035: The Touch World — Collect and Avoid by Moving Alone

**Date:** 2026-10-27
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00035`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00035/`
**Task plan:** [`../tasks/00042-touch-world.md`](../tasks/00042-touch-world.md)

## Goal

In the circle world food and poison act only on a click, and the agent clicks on ~23% of its steps wherever it is
(vm-00031): foraging there mixes reaching a circle with clicking it. Here contact alone counts (`env.touch`): does the
agent steer into food and around poison? Self-forcing (vm-00032) keeps the dream truer; does a truer dream let the
4-step lookahead steer better, and does `pair` decoding (task 00038) matter there?

## Method

- **World** (`env.touch`, `configs/U_touch.json`): `U_universal.json`'s world (grey torus, solid food, checkerboard
  poison adding 0.5 hunger, 8 drifting dots, slow hunger), but food acts on contact (relief, respawns elsewhere) and so
  does poison (+0.5, and that poison is replaced by a new one away from the cursor; always 2). The click does nothing.
  Food smaller: radius 8–11 (was 12–16); poison 5–8 as before.
- **Arms** (100k ticks, seeds 1–3, the instinct on: 16 candidates):
  - **TB:** no self-forcing, 1-step instinct (3090, 62 ticks/s).
  - **TSF:** self-forcing (8 windows, 1–3 steps, J = 2, pair; vm-00032's setting), 1-step instinct (4090, 22–30 ticks/s).
  - **TSF4:** self-forcing with the 4-step pair lookahead in life (`instinct_h = 4`, `instinct_decode = pair`; started
    because the frozen TB s1 test showed the pair lookahead ahead of the 1-step instinct; 11–13 ticks/s).
- **Frozen at 100k** (`online.py --eval-tests forage`, 16 fresh worlds × 1,000 steps): food and poison per 1k with the
  instinct off (the model's own move prediction), the 1-step instinct, the 4-step lookahead with mean and with pair
  decoding. The dream curve (`eval_dream.py`, 16 dreamed steps with the real moves).
- **Chance** (`forage_chance.py`): blind random walks in the same worlds.

## Log

### 2026-10-27
- TSF4 first ran three seeds on the 3090 next to the frozen tests at 1.2 ticks/s (23 h a life); stopped at 1k ticks.
  Alone on the 4090 it runs 20 ticks/s; rerun as seeds 1–2 on the 4090, seed 3 on the 3090 after the tests.
- TSF s1 ate nothing from 65k to 79k ticks (still touching poison, 17 of its 100 1k-windows without food), then
  recovered (9–22 per 1k to the end). No other life had more than one food-free window.

## Results

### R1. Chance: blind random walks (food and poison per 1k)

| walk | food | poison | food : poison |
|---|---|---|---|
| uniform moves in [-1, 1]² | 4.7 | 11.8 | 0.40 |
| full-speed diagonals, new each step | 12.7 | 18.7 | 0.68 |
| full-speed diagonals, held 5 steps | 24.7 | 31.5 | 0.78 |
| held 20 steps | 11.2 | 37.0 | 0.30 |
| held 100 steps | 2.9 | 39.3 | 0.07 |

Food alone is no evidence of seeking (a fast walk turning every 5 steps eats 24.7); the ratio is: no blind walk
exceeds 0.78.

### R2. Frozen at 100k: food / poison per 1k (food : poison)

| run | instinct off | 1-step | 4-step mean | **4-step pair** |
|---|---|---|---|---|
| TB s1 | 16.4 / 18.1 (0.91) | 18.4 / 14.3 (1.28) | 19.2 / 15.4 (1.25) | 24.3 / 12.9 (1.89) |
| TB s2 | 17.3 / 15.4 (1.12) | 16.4 / 10.7 (1.53) | 15.9 / 10.7 (1.49) | 19.6 / 7.4 (2.66) |
| TB s3 | 12.9 / 16.5 (0.78) | 16.0 / 16.2 (0.99) | 14.2 / 13.7 (1.04) | 15.9 / 11.7 (1.36) |
| TSF s1 | 10.8 / 14.2 (0.75) | 13.5 / 9.9 (1.36) | 12.6 / 10.2 (1.23) | 14.9 / 6.2 (2.41) |
| TSF s2 | 17.1 / 21.5 (0.79) | 18.2 / 14.2 (1.28) | 20.3 / 13.4 (1.52) | 26.5 / 8.5 (3.12) |
| TSF s3 | 14.6 / 17.5 (0.83) | 17.8 / 8.8 (2.03) | 18.3 / 9.2 (1.98) | 21.7 / 2.8 (7.71) |
| TSF4 s1 | 11.2 / 13.9 (0.81) | 13.6 / 6.6 (2.05) | 14.2 / 9.2 (1.54) | 15.6 / 3.3 (4.70) |
| TSF4 s2 | 15.4 / 16.5 (0.94) | 15.3 / 10.2 (1.49) | 16.9 / 10.2 (1.65) | 19.3 / 2.7 (7.19) |
| TSF4 s3 | 15.6 / 18.0 (0.86) | 17.6 / 7.2 (2.44) | 22.2 / 12.1 (1.84) | 24.1 / 2.9 (8.19) |

Means, 4-step pair: TB 19.9 food / 10.7 poison; TSF 21.0 / 5.8; TSF4 19.7 / 3.0. 1-step: TB 16.9 / 13.7, TSF 16.5 /
11.0, TSF4 15.5 / 8.0.

### R3. In life: food / poison per 1k, by quarter of the life

| run | 0–25k | 25–50k | 50–75k | 75–100k |
|---|---|---|---|---|
| TB s1 / s2 / s3 | 15.7/18.7 · 16.7/15.6 · 14.7/16.8 | 18.0/16.0 · 17.9/12.6 · 11.5/13.3 | 17.6/17.4 · 18.0/11.6 · 11.3/13.0 | 18.8/17.3 · 18.0/9.8 · 12.0/12.6 |
| TSF s1 / s2 / s3 | 17.8/17.4 · 16.9/22.0 · 14.2/15.8 | 15.9/14.6 · 19.1/18.4 · 14.3/10.0 | 8.5/17.1 · 18.9/18.3 · 14.5/8.5 | 11.7/16.8 · 18.8/16.8 · 14.4/7.0 |
| **TSF4 s1 / s2 / s3** | 11.0/11.3 · 15.3/13.7 · 15.4/12.2 | 10.6/7.3 · 15.7/6.0 · 20.8/8.2 | 11.9/4.1 · 15.0/4.5 · 20.2/8.0 | **12.1/3.6 · 16.9/3.8 · 21.0/5.2** |

### R4. The dream at 100k (after 16 dreamed steps)

| run | cursor distance (px) s1 / s2 / s3 | hunger error | pixels exactly right |
|---|---|---|---|
| TB | 12.3 / 14.6 / 6.3 | 0.130 / 0.128 / 0.178 | 0.912 / 0.900 / 0.903 |
| TSF | 2.5 / 3.1 / 3.6 | 0.049 / 0.074 / 0.120 | 0.922 / 0.923 / 0.917 |
| TSF4 | 3.6 / 2.8 / 3.2 | 0.034 / 0.022 / 0.053 | 0.932 / 0.919 / 0.904 |

## Analysis

- **The agent steers, without being rewarded for it.** Every run's own move prediction (instinct off) is near the
  best blind walk (0.75–1.12 against 0.78); every instinct raises the ratio, and the 4-step pair lookahead raises it
  most in all 9 runs (1.36–8.19). The instinct only reads the model's predicted hunger; collecting and avoiding come
  from the model predicting what contact will do.
- **Pair decoding is what makes the lookahead work.** With mean decoding the 4-step lookahead is no better than one
  step (TB 1.04–1.49, TSF 1.23–1.98); with pair it is better in every run. Likely because a mean-decoded imagined
  frame blurs the cursor and circles into grey levels the world never shows (task 00038), so imagined contacts are
  lost after a step or two.
- **Self-forcing makes the lookahead avoid poison.** Frozen with the pair lookahead, poison falls from 10.7 per 1k
  (TB) to 5.8 (TSF) and 3.0 (TSF4), while food stays ~20: ratios 1.4–2.7 against 2.4–7.7 and 4.7–8.2. Its dream stays
  truer (cursor 2.5–3.6 px off after 16 steps against 6.3–14.6; hunger error 0.02–0.12 against 0.13–0.18): an imagined
  path into a poison circle now looks like one 4 steps ahead.
- **In life, the lookahead is the behaviour.** With the 1-step instinct TB and TSF live alike (poison 7–17 per 1k in
  the last quarter). TSF4, living by the pair lookahead, cuts its poison over its life, from 11–14 per 1k in the first
  quarter to 3.6–5.2 in the last (TB 9.8–17.3), with food 12–21 (TB 12–19): it learns from the life its lookahead
  chose. A TB4 arm (the pair lookahead in life, no self-forcing) was not run, so the in-life gain is not split between
  the lookahead and self-forcing; the frozen tests split it (R2: self-forcing doubles the pair lookahead's ratio).
- **Food is collected about as often in every arm** (frozen pair 15–27 per 1k): the gain is avoidance. Seeking is
  limited by what the lookahead sees: 16 candidates held for 4 steps reach 16–23 px, and the food is often
  farther.
- **TSF s1's food-free stretch** (65–79k, still touching poison) recovered on its own; whether self-forcing makes such
  stretches likelier needs more seeds.

## Conclusions

1. In the touch world the universal model plus the hunger instinct collects food and avoids poison well above
   chance, with no reward: food : poison up to 8.2 against at most 0.78 for any blind walk.
2. A truer dream pays off in behaviour here: self-forcing with the pair lookahead cuts poison contacts ~3.5× (frozen:
   10.7 → 3.0 per 1k) at the same food. In the click world (vm-00032) it did not; there behaviour mixed reaching and
   clicking.
3. The pair decoding is necessary: the 4-step lookahead with mean decoding gains nothing over one step.

## Next Steps

- TB4 (the pair lookahead in life without self-forcing): the missing control for R3.
- Seeking: a longer horizon or more candidate paths (the lookahead sees 16–23 px), at its cost (TSF4 11–13 ticks/s).
- The touch world with moving food (`env.target_speed`) and a dream video of TSF4 avoiding poison.
