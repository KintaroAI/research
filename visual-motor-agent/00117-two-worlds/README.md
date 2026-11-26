# vm-00117: Two Worlds, One Set of Weights

**Date:** 2026-11-25 to 2026-11-26
**Status:** Complete
**Source:** `dev/` (pending tag `exp/vm-00117`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00117/`
**Task plan:** [`../tasks/00130-two-worlds.md`](../tasks/00130-two-worlds.md)

## Goal

The first challenge of `ARCHITECTURE.md` section 13: "universal" has meant the same code in every world, each world with its own life. Here one set of weights meets a second world. Does a trained agent learn the other world faster or slower than a new one, how much of its first world is left afterwards, and how fast does it come back?

## Method

The default agent's two configs differ only in the world, so the weights of a life in one load in the other (`lifetime.py --init`). No code changes.

- **TB:** Breakout stage 1 from TT's (vm-00101) 60k weights, 60k ticks, against BD (vm-00103) from scratch, paired by seed.
- **BF:** the food world from BD's 60k weights, 60k ticks, against TT from scratch, paired by seed.
- **The old world, learning off** (`ppo.lr = 0`, new fields and courts, seed 100 + s), from the second life's checkpoints at 10k, 30k and 60k:
  - TB's weights in the food world for 2,000 ticks; BF's in Breakout for 6,000 ticks.
  - **Before:** the first life's 60k weights in the same test (TT: vm-00114's `check/TT_ring`; BD: vm-00116's ring arm).
  - **Floor:** the untrained default model in the same test, the instinct choosing on an untrained label.
  - **Share kept** = (after − floor) ÷ (before − floor), on the means of the seeds.
- **TBF, BFB:** a third life of 30k ticks back in the old world, from the second life's 60k weights, against the first life's first 30k ticks.
- **Seeds 1–12;** seeds 1–6 on the RTX 3090 and 7–12 on the RTX 4090. A second or third life uses its first life's seed.

A life started from weights has an empty memory and a new optimizer, so the old world is never replayed: learning one world after another without rehearsal, the harsh case.

## Readings

Set in the task plan before the results.

1. **Into Breakout:** TB − BD hit share at 10–30k and at 40–60k; the first 5k window at .5. No direction predicted.
2. **Into the food world:** BF − TT food and poison per 1k at 10–30k and at 40–60k.
3. **Forgetting:** the share kept after 10k, 30k and 60k ticks of the new world. Below one half at 60k, the forgetting is large and the follow-up is rehearsal (one process, one memory across both worlds), then the flexible layer. One half or more, the weights hold two worlds without help.
4. **Savings:** the third life's first 30k ticks against the first life's.
5. **The cost of a past:** readings 1 and 2 at 40–60k. Does a model that already holds one world end lower in the next?

## Results

Paired by seed; ± is the 95% interval. The tables are `make table`. The RTX 4090's chain (seeds 7–12) ran from 22:12 on 2026-11-25 to 04:20, the RTX 3090's (seeds 1–6) from 22:38 to 09:16 on 2026-11-26. No run failed.

### R1. Into Breakout: a food-world past helps

Hit share, 12 seeds:

| Ticks | BD (12) | TB (12) | TB − BD (95%) | Seeds |
|---|---|---|---|---|
| 0–10k | .226 | .247 | +.021 ± .022 | 8 of 12 above |
| 10–20k | .251 | .351 | +.100 ± .038 | 12 of 12 above |
| 20–30k | .335 | .533 | +.198 ± .042 | 12 of 12 above |
| 30–40k | .442 | .660 | +.218 ± .058 | 12 of 12 above |
| 40–50k | .602 | .745 | +.143 ± .063 | 11 of 12 above |
| 50–60k | .678 | .791 | +.113 ± .053 | 10 of 12 above |

| First 5k window at or above | TB | BD | TB − BD, seeds reaching it in both |
|---|---|---|---|
| .500 | 21–33k (median 26k) | 32–55k (median 40.5k) | −14.8k ± 4.5k (12 earlier, 0 later of 12) |
| .700 | 29–47k (median 39.5k) | 44–58k (median 47.5k), 8 of 12 | −11.0k ± 6.9k (8 earlier, 0 later of 8) |

- **Reading 1: TB − BD = +.145 ± .031 at 10–30k** (12 of 12 seeds above) **and +.129 ± .054 at 40–60k** (11 of 12).
- **TB reaches .5 about 15k ticks earlier, in every seed** (−14.8k ± 4.5k). All 12 of its seeds reach .7; 8 of BD's do.
- **It does not start ahead.** At 0–10k the difference is +.021 ± .022 and both arms are at a random paddle's level (.235, vm-00103). The food world's weights do not play Breakout; they learn it faster.

### R2. Into the food world: a Breakout past hurts

Food and poison per 1k, 12 seeds:

| Ticks | TT food | BF food | BF − TT food (95%) | Seeds | TT poison | BF poison | BF − TT poison (95%) | Seeds |
|---|---|---|---|---|---|---|---|---|
| 0–10k | 12.6 | 10.7 | −2.0 ± 1.1 | 2 of 12 above | 24.5 | 25.8 | +1.4 ± 1.4 | 8 of 12 above |
| 10–20k | 32.4 | 13.8 | −18.6 ± 3.5 | 0 of 12 above | 17.4 | 21.6 | +4.2 ± 1.0 | 12 of 12 above |
| 20–30k | 58.5 | 18.9 | −39.6 ± 3.1 | 0 of 12 above | 12.5 | 20.1 | +7.6 ± 1.4 | 12 of 12 above |
| 30–40k | 68.0 | 29.8 | −38.3 ± 3.2 | 0 of 12 above | 9.7 | 16.1 | +6.4 ± 1.3 | 12 of 12 above |
| 40–50k | 71.4 | 44.2 | −27.2 ± 4.9 | 0 of 12 above | 8.5 | 13.0 | +4.5 ± 1.4 | 12 of 12 above |
| 50–60k | 71.8 | 55.9 | −15.9 ± 4.0 | 0 of 12 above | 8.0 | 11.0 | +3.0 ± 1.2 | 11 of 12 above |

- **Reading 2: BF − TT food = −29.1 ± 3.0 per 1k at 10–30k and −21.5 ± 4.3 at 40–60k,** 0 of 12 seeds above at either. **Poison is +5.9 ± 1.0 and +3.7 ± 1.1,** 12 of 12 above.
- **BF starts late.** It has 13.8 food per 1k at 10–20k and 18.9 at 20–30k, where TT has 32.4 and 58.5. Its level at 50–60k (55.9) is TT's at 20–30k (58.5): about 30k ticks behind.
- **It is still rising at 60k** (29.8, 44.2 and 55.9 in the last three windows) while TT is level (71.4, 71.8). The gap narrows from −39.6 ± 3.1 at 20–30k to −15.9 ± 4.0 at 50–60k and is not closed.

### R3. Forgetting: the old world's play is at an untrained model's level after 10k ticks

**TB's weights in the food world (2,000 ticks): food per 1k**

| Weights | Mean of 12 seeds | Share kept | After − floor (95%) | Seeds | After − halfway point (95%) | Seeds |
|---|---|---|---|---|---|---|
| before (the first life's 60k weights) | 69.8 | 1 |  |  |  |  |
| after 10k ticks of the new world | 16.8 | 0.12 | +7.2 ± 2.8 | 11 of 12 above | −22.9 ± 2.5 | 0 above, 12 below |
| after 30k ticks of the new world | 13.2 | 0.06 | +3.6 ± 2.5 | 10 of 12 above | −26.5 ± 2.0 | 0 above, 12 below |
| after 60k ticks of the new world | 10.9 | 0.02 | +1.3 ± 1.6 | 8 of 12 above | −28.8 ± 1.6 | 0 above, 12 below |
| floor (an untrained model) | 9.6 | 0 |  |  |  |  |

**TB's weights in the food world (2,000 ticks): poison per 1k**

| Weights | Mean of 12 seeds | Share kept | After − floor (95%) | Seeds | After − halfway point (95%) | Seeds |
|---|---|---|---|---|---|---|
| before (the first life's 60k weights) | 8.2 | 1 |  |  |  |  |
| after 10k ticks of the new world | 27.5 | −0.03 | +0.5 ± 3.6 | 8 of 12 above | +9.9 ± 3.2 | 12 above, 0 below |
| after 30k ticks of the new world | 28.0 | −0.05 | +1.0 ± 3.2 | 8 of 12 above | +10.4 ± 2.7 | 12 above, 0 below |
| after 60k ticks of the new world | 27.0 | 0.00 | +0.0 ± 3.3 | 6 of 12 above | +9.4 ± 2.7 | 12 above, 0 below |
| floor (an untrained model) | 27.0 | 0 |  |  |  |  |

**BF's weights in Breakout (6,000 ticks): hit share**

| Weights | Mean of 12 seeds | Share kept | After − floor (95%) | Seeds | After − halfway point (95%) | Seeds |
|---|---|---|---|---|---|---|
| before (the first life's 60k weights) | .701 | 1 |  |  |  |  |
| after 10k ticks of the new world | .237 | −0.02 | −.009 ± .032 | 5 of 12 above | −.237 ± .031 | 0 above, 12 below |
| after 30k ticks of the new world | .227 | −0.04 | −.018 ± .049 | 5 of 12 above | −.246 ± .028 | 0 above, 12 below |
| after 60k ticks of the new world | .217 | −0.06 | −.029 ± .049 | 3 of 12 above | −.257 ± .040 | 0 above, 12 below |
| floor (an untrained model) | .245 | 0 |  |  |  |  |

- **Reading 3, the share kept at 60k: .02 of the food, .00 of the poison avoided, −.06 of the hit share.** By the rule set beforehand (below one half) the forgetting is large.
- **It is complete, and at the first test.** After 10k ticks of the new world the shares kept are .12, −.03 and −.02. In all nine tests all 12 seeds are on the floor's side of the halfway point.
- **Only the food after 10k ticks is measurably above the floor** (+7.2 ± 2.8 per 1k, 11 of 12 seeds), and it fades: +3.6 ± 2.5 after 30k, +1.3 ± 1.6 after 60k. Poison and hit share are at the floor from the first test on.
- **The tests' sizes:** in 2,000 ticks of the food world a seed eats 126–147 food before and 13–25 at the floor; in 6,000 ticks of Breakout a seed meets 72–100 balls. Before − floor is +60.2 ± 2.1 food, −18.8 ± 2.0 poison and +.456 ± .084 hit share, 12 of 12 seeds each.

**The old world's prediction in the same tests** (cross-entropy in nats, the mean over a test's rows and the 12 seeds; not among the readings set beforehand). An untrained model is at 3.5, the uniform level of 32 bins.

| Old world | Weights | Patches | Body | Move |
|---|---|---|---|---|
| food world, TB's weights | before | 0.79 | 1.53 | 3.15 |
| food world, TB's weights | after 10k | 1.78 | 1.57 | 5.11 |
| food world, TB's weights | after 30k | 2.54 | 1.96 | 5.27 |
| food world, TB's weights | after 60k | 2.94 | 2.03 | 5.30 |
| food world, TB's weights | floor | 3.49 | 3.48 | 3.49 |
| Breakout, BF's weights | before | 0.66 | 0.70 | 3.03 |
| Breakout, BF's weights | after 10k | 1.21 | 0.93 | 4.16 |
| Breakout, BF's weights | after 30k | 1.16 | 0.95 | 4.05 |
| Breakout, BF's weights | after 60k | 1.15 | 1.08 | 3.86 |
| Breakout, BF's weights | floor | 3.49 | 3.48 | 3.49 |

- **The frame's prediction is lost more slowly than the play, and unevenly.** Breakout's patches go from .66 to 1.21 after 10k ticks of the food world and stay there (1.15 after 60k). The food world's go from .79 to 1.78, 2.54 and 2.94, most of the way to an untrained model's 3.49.
- **The body's tokens** keep more: 1.53 to 2.03 in the food world, .70 to 1.08 in Breakout.
- **The move token is worse than untrained** in the old world: 5.1–5.3 and 3.9–4.2 nats against 3.49. It names the new world's moves with confidence.

### R4. Savings: back in the old world

A third life of 30k ticks from the second life's 60k weights, 12 seeds.

**Against the first life's first 30k ticks** (the reading set beforehand)

| Ticks | TT food | TBF food | TBF − TT food (95%) | Seeds | TT poison | TBF poison | TBF − TT poison (95%) | Seeds |
|---|---|---|---|---|---|---|---|---|
| 0–10k | 12.6 | 14.4 | +1.8 ± 1.3 | 10 of 12 above | 24.5 | 26.7 | +2.2 ± 1.7 | 9 of 12 above |
| 10–20k | 32.4 | 29.8 | −2.7 ± 5.1 | 5 of 12 above | 17.4 | 20.7 | +3.3 ± 1.0 | 12 of 12 above |
| 20–30k | 58.5 | 48.4 | −10.1 ± 5.1 | 0 of 12 above | 12.5 | 16.0 | +3.4 ± 1.4 | 12 of 12 above |
| 10–30k | 45.5 | 39.1 | −6.4 ± 4.9 | 4 of 12 above | 15.0 | 18.3 | +3.4 ± 1.0 | 12 of 12 above |

| Ticks | BD | BFB | BFB − BD (95%) | Seeds |
|---|---|---|---|---|
| 0–10k | .226 | .258 | +.032 ± .044 | 10 of 12 above |
| 10–20k | .251 | .366 | +.115 ± .038 | 12 of 12 above |
| 20–30k | .335 | .504 | +.168 ± .061 | 12 of 12 above |
| 10–30k | .292 | .432 | +.140 ± .044 | 12 of 12 above |

| Measure: third life, first life | First life at 50–60k | Third life: first 5k window at half of it | First life: the same | Third − first | Third life at 20–30k | Third at 20–30k − first at 50–60k (95%) | Seeds |
|---|---|---|---|---|---|---|---|
| food per 1k: TBF, TT | 71.8 | 16–29k (median 22k) | 17–23k (median 18.5k) | +2.1k ± 2.7k (4 earlier, 6 later of 12) | 48.4 | −23.4 ± 4.8 | 0 of 12 above |
| hit share: BFB, BD | .678 | 6–19k (median 12.5k) | 7–28k (median 25k) | −9.7k ± 4.1k (11 earlier, 1 later of 12) | .504 | −.174 ± .068 | 1 of 12 above |

**Against the second life of the other order** (both enter the world from 60k ticks of the other one; only the third life has lived in it before; not among the readings set beforehand)

| Ticks | BF food | TBF food | TBF − BF food (95%) | Seeds | BF poison | TBF poison | TBF − BF poison (95%) | Seeds |
|---|---|---|---|---|---|---|---|---|
| 0–10k | 10.7 | 14.4 | +3.8 ± 1.4 | 11 of 12 above | 25.8 | 26.7 | +0.8 ± 1.1 | 8 of 12 above |
| 10–20k | 13.8 | 29.8 | +15.9 ± 3.3 | 12 of 12 above | 21.6 | 20.7 | −0.9 ± 0.8 | 3 of 12 above |
| 20–30k | 18.9 | 48.4 | +29.5 ± 3.8 | 12 of 12 above | 20.1 | 16.0 | −4.1 ± 1.8 | 1 of 12 above |
| 10–30k | 16.4 | 39.1 | +22.7 ± 3.4 | 12 of 12 above | 20.8 | 18.3 | −2.5 ± 1.1 | 0 of 12 above |

| Ticks | TB | BFB | BFB − TB (95%) | Seeds |
|---|---|---|---|---|
| 0–10k | .247 | .258 | +.011 ± .032 | 8 of 12 above |
| 10–20k | .351 | .366 | +.015 ± .027 | 9 of 12 above |
| 20–30k | .533 | .504 | −.029 ± .055 | 3 of 12 above |
| 10–30k | .437 | .432 | −.006 ± .029 | 4 of 12 above |

- **Reading 4, the food world: no savings against a new model.** TBF − TT food is +1.8 ± 1.3 per 1k at 0–10k, −2.7 ± 5.1 at 10–20k and −10.1 ± 5.1 at 20–30k (0 of 12 seeds above), with 3.4 ± 1.0 more poison at 10–30k. It reaches half of TT's final level at 16–29k (median 22k), TT at 17–23k (median 18.5k).
- **Reading 4, Breakout: BFB − BD = +.140 ± .044 at 10–30k** (12 of 12 seeds above). It reaches half of BD's final level 9.7k ± 4.1k ticks earlier.
- **In Breakout that is the food world's gift, not a memory of Breakout:** BFB − TB = −.006 ± .029 at 10–30k. The agent that had played Breakout before learns it as fast as the one that had only lived in the food world.
- **In the food world the earlier life does count:** TBF − BF = +22.7 ± 3.4 food per 1k at 10–30k (12 of 12 seeds above), with 2.5 ± 1.1 less poison. Both come from 60k ticks of Breakout. The one that had lived in the food world before learns it nearly as fast as a new model; the other is 30k ticks behind (R2).
- **Neither is back at its first life's level after 30k ticks:** 48.4 food per 1k at 20–30k against 71.8 at the end of the first life (−23.4 ± 4.8), and .504 hit share against .678 (−.174 ± .068).

### R5. How the lives move

Logged in life, 12 seeds; not among the readings set beforehand. Smoothness is the mean cosine between one tick's move and the next; capped is the share of ticks at the hunger cap.

**The food world**

| Arm | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| TT, smoothness | 0.18 | 0.37 | 0.47 | 0.50 | 0.51 | 0.52 |
| BF, smoothness | 0.10 | 0.11 | 0.19 | 0.32 | 0.45 | 0.53 |
| TBF, smoothness | 0.18 | 0.29 | 0.40 |  |  |  |
| TT, capped | 0.44 | 0.15 | 0.02 | 0.01 | 0.01 | 0.01 |
| BF, capped | 0.50 | 0.37 | 0.26 | 0.12 | 0.05 | 0.02 |
| TBF, capped | 0.42 | 0.17 | 0.06 |  |  |  |

**Breakout**

| Arm | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| BD, smoothness | 0.23 | 0.21 | 0.23 | 0.23 | 0.23 | 0.20 |
| TB, smoothness | 0.49 | 0.44 | 0.37 | 0.32 | 0.30 | 0.28 |
| BFB, smoothness | 0.38 | 0.34 | 0.33 |  |  |  |
| BD, capped | 0.52 | 0.47 | 0.33 | 0.22 | 0.10 | 0.06 |
| TB, capped | 0.47 | 0.31 | 0.14 | 0.08 | 0.04 | 0.03 |
| BFB, capped | 0.46 | 0.31 | 0.17 |  |  |  |

- **The way of moving comes along, in both directions.** TB enters Breakout with the food world's smooth moves (.49 at 0–10k; TT ended at .52, BD is at .20–.23 all its life) and keeps part of them to the end (.28 at 50–60k).
- **BF enters the food world moving less smoothly than a new model** (.10–.11 for 20k ticks; TT .18, then .37) and reaches TT's smoothness only at 50–60k (.53).
- **BF is hungry for longer:** .37 of ticks at the hunger cap at 10–20k and .26 at 20–30k, where TT has .15 and .02.
- **The third lives:** TBF starts as smooth as a new model (.18 at 0–10k, not BF's .10) and is as seldom at the cap (.17 at 10–20k against TT's .15), though it smooths more slowly (.29 and .40 against TT's .37 and .47). BFB plays Breakout with much of the food world's smoothness (.38 at 0–10k; TB .49, BD .23).

## Analysis

The start in a world, by the agent's past (12 seeds, ticks 10–30k; from R1, R2 and R4):

| Enters | A new model | From the other world only | From this world, then the other |
|---|---|---|---|
| Breakout, hit share | .292 (BD) | .437 (TB) | .432 (BFB) |
| The food world, food per 1k | 45.5 (TT) | 16.4 (BF) | 39.1 (TBF) |

- **Transfer is one-sided.** A food-world past makes Breakout faster to learn (+.145 ± .031 at 10–30k, .5 reached about 15k ticks earlier in every seed) and higher at 60k (+.113 ± .053 at 50–60k). A Breakout past makes the food world slower to learn by about 30k ticks.
- **What comes along looks like a way of moving** (R5; an observation, not a test). TT's moves grow smooth over its life (.18 to .52), BD's never do (.20–.23). TB enters Breakout smooth and learns it faster. BF enters the food world moving less smoothly than a new model and sits at the hunger cap for 20k ticks longer; it starts to eat as its moves smooth out. The food world's body has inertia, so moves that change every tick take it nowhere.
- **The cost of a past (reading 5).** None in Breakout: TB ends above BD (+.129 ± .054 at 40–60k). In the food world BF is 21.5 ± 4.3 food per 1k below TT at 40–60k, but it is still rising by 12 to 14 per 10k ticks while TT is level, so 60k ticks do not say where it ends.
- **Forgetting is fast and complete in play.** After 10k ticks of the new world the old world's play is at an untrained model's level but for a trace of food (share kept .12), and by 60k that is gone too. By the rule set beforehand the weights do not hold two worlds without help.
- **The play goes before the seeing.** After 60k ticks of the food world Breakout's frame is still predicted at 1.15 nats (.66 before, 3.49 untrained), yet its play is at the floor from the first test on. The move token, the source of the candidates, is worse than untrained in the old world.
- **Coming back depends on the world** (the table above).
  - Into Breakout, a food-world past gives the same start whether or not the agent had played Breakout before (.432 and .437, against a new model's .292). No trace of the earlier Breakout life shows.
  - Into the food world, the agent that had lived there before is far ahead of the one that had not (39.1 against 16.4), though both come from 60k ticks of Breakout, and a little behind a new model (45.5).
  - So the food-world life leaves something that 60k ticks of Breakout do not erase and that the frozen test does not show: its play was at the floor (R3). Whether that is the way of moving or what the model sees is not separated here.
  - **A caveat for Breakout:** BF's food-world life ended lower than TT's (55.9 against 71.8 food per 1k at 50–60k). A smaller gift from the food world topped up by a remnant of Breakout would give the same numbers.
- **What the test of forgetting misses.** Play with learning off says the old world is gone after 10k ticks. Relearning says that, of the food world, it is not.

## Conclusions

1. **A food-world past helps in Breakout:** TB − BD = +.145 ± .031 hit share at 10–30k (12 of 12 seeds above) and +.129 ± .054 at 40–60k; .5 is reached 14.8k ± 4.5k ticks earlier.
2. **A Breakout past hurts in the food world:** BF − TT = −29.1 ± 3.0 food per 1k at 10–30k and −21.5 ± 4.3 at 40–60k (0 of 12 seeds above), with +5.9 ± 1.0 and +3.7 ± 1.1 poison. BF is about 30k ticks behind and still rising at 60k.
3. **The old world's play is forgotten within 10k ticks of the new one:** the share kept at 60k is .02 of the food, .00 of the poison avoided and −.06 of the hit share, and already .12, −.03 and −.02 after 10k. By the rule set beforehand the follow-up is rehearsal.
4. **Back in the food world the agent relearns no faster than a new model learns** (TBF − TT = −6.4 ± 4.9 food per 1k at 10–30k) **but much faster than an agent with a Breakout past only** (TBF − BF = +22.7 ± 3.4, 12 of 12 seeds above).
5. **Back in Breakout it learns faster than a new model** (BFB − BD = +.140 ± .044) **and as fast as an agent that never played Breakout but lived in the food world** (BFB − TB = −.006 ± .029).
6. **The way of moving comes along in both directions,** and the frame's prediction outlasts the play.

## Next steps

To decide with the owner.

- **Rehearsal,** the follow-up named beforehand: one process and one memory across both worlds, so the old world's ticks are replayed while the new one is lived. The measures are ready: the share kept with learning off, and the start in a world by the agent's past.
- **Then the flexible layer** (`ARCHITECTURE.md` section 13, "adaptation that does not erase").
- **What these runs leave open:**
  - Whether BF ever reaches TT's level: a longer second life.
  - What of the food world survives Breakout: the same third life from TB's weights at 10k and 30k, and the old world's label read with learning off.
  - Whether the gift into Breakout grows with the food-world skill: TB from TT's weights at 30k.

## Limits

- Two worlds, one order of each, 60k ticks a life; the default agent only.
- Each life starts with an empty memory and a new optimizer: the harsh case, by design.
- The readings of how the lives move, of the old world's prediction and of the third life against the other order's second life were not set beforehand.

## Files

- `00117-two-worlds/Makefile`: `runs`, `frozen`, `floor`, `back`, `table`.
- `dev/analyze_lives.py two-worlds`: the tables above; `dev/tests/test_lives.py` (7 tests of its helpers).
- The tables as printed: `exp_00117/tables.md`.
- No change to the model, the worlds or `lifetime.py`.
