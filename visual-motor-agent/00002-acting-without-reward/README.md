# vm-00002: Acting Without Reward: Planning With a Self-Taught World Model

**Date:** 2026-10-23
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00002`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00002/`
**Task plan:** [`../tasks/00002-latent-dynamics-and-planning.md`](../tasks/00002-latent-dynamics-and-planning.md)

## Goal

vm-00001 showed that a reward-free babbler learns what its actions do: moves shift the cursor, a
click inside the circle removes it and relieves hunger. Can the agent *use* that knowledge? This
experiment tests whether it can find and click circles by planning with a world model it learned
without reward, minimizing its own predicted hunger. Nothing is trained on reward: no value
function and no policy training (plan §11, phase 5).

## Hypothesis

1. A compact latent dynamics model learned during babbling supports multi-step imagination.
   Pixel-space imagination did not (cursor error 0.8 → 8 → 27 px over three steps in vm-00001).
2. Babbling with fewer clicks (probability 0.1 instead of 0.5) teaches the model how the cursor
   moves over the circle, which vm-00001's babbler never learned. A simulation shows 11.6× more
   such transitions.
3. Model-predictive control on that model (cross-entropy method, horizon 12, cost = predicted
   hunger) hits the circle far more often and faster than the random policy, with zero reward.

## Method

### Model additions (`model.dyn_hidden`)
- Latent: the encoder's visual feature `s_t` (128-d), kept informative by inverse dynamics (cursor)
  and a coarse 16×16 decoder (circle).
- Transition: `s' = s + MLP([s, a])`.
- Hunger head: `h' = sigmoid(MLP([s, h, a]))`.
- Training: during babbling, chains of up to `dyn_horizon` imagined steps start at every token.
  They are trained against the encoder's latents of the real future frames (with stopped
  gradients), the real next hunger and the real next frames (coarse).

### Planner (`dev/plan.py`)
At every step, the planner encodes the current frame and samples 256 action sequences of length 12.
It imagines their outcomes with the transition and hunger heads, refits the sampling distribution to
the 10% with the lowest summed predicted hunger (4 iterations), and executes the first action.

### Evaluation
The 128-episode test set of the PPO runs (radius 12–16):
- success within 200 steps and median steps to a hit;
- random policy, untrained model and PPO (vm-00001) as references.

## Log

### 2026-10-23
- Task plan written. A simulation of the babbling policy fixed the click probability at 0.1:

  | Click probability | Cursor moving over a live circle | Hunger reliefs per 1k steps |
  |---|---:|---:|
  | 0.5 | 0.24% of steps | 4.85 |
  | 0.3 | 0.76% | 4.45 |
  | 0.2 | 1.36% | 4.14 |
  | 0.1 | 2.78% | 3.43 |
  | 0.05 | 4.73% | 2.60 |
- Latent dynamics heads implemented in C++ (`model.dyn_hidden`). Every gradient matches the float64
  PyTorch reference to about 1e-7 relative, including chains that cross episode ends. `vm_imagine`
  matches an unrolled reference rollout.
- **First attempt collapsed the representation** (runs 001–002, stopped at 130k steps).
  - Setup: the imagined chains started from the encoder's feature with gradients flowing back into
    the encoder.
  - What happened: the latent-matching loss pulled the features toward constants that are trivially
    predictable. At 65k steps, 32 of 128 feature dimensions were active, against 102 for the
    vm-00001 babbler at the same point. The across-frame spread was 0.025 vs 0.145, and the
    inverse-dynamics R² stayed at 0 where vm-00001 reached 0.91 by 80k steps.
  - This is the collapse the plan (§7) warns about; the decoder on real frames did not prevent it.
  - **First fix:** the chains start from a *detached* feature, and only the decoder on real frames
    still trains the encoder. Relaunched as runs 003 and 004 (stopped at about 100k steps).
- **The decoder alone still shrank the features** (runs 003–004).
  - At 65k steps: mean |v| 0.027 vs 0.127 in vm-00001, 61 of 128 dims active, and the
    inverse-dynamics loss stuck at the predict-zero level (0.30 vs 0.07 for vm-00001 at the same
    point).
  - Probably its large gradients kill ReLU units in the encoder's feature layer.
  - **Fix: full decoupling.** The dynamics heads (transition, hunger, decoder) are trained on top
    of the encoder and never change it. The encoder trains exactly as vm-00001's babbler (run 012:
    cursor decodable at R² 0.96). Runs 005 (`P_dyn`, click 0.1) and 006 (`P_dyn_p50`, click 0.5).
- Runs 005 and 006 babbled for 2.1M steps each. Their inverse dynamics learned on schedule:
  R² 0.79 (005) and 0.90 (006) after 98k steps. The planner and the imagination analysis were run
  on every milestone checkpoint.
- **The linear probe misreads imagined latents.** Read through a ridge probe, imagined cursors looked
  worse than no motion: 5.4 px after one step, against 2.8 px for assuming the cursor stays put. The
  probe has large weights in directions where real latents barely vary, so small off-manifold errors
  in imagined latents blow up. `analyze_imagination.py` now also reads the cursor by retrieval: the
  nearest latent among frames of the same circle with the cursor on a 2-px grid. That gives 1.5 px.
  It also gained four tests:
  - an approach test: a scripted straight path to the circle, then one click;
  - an edge test: one imagined click at a signed distance g from the circle's edge;
  - straight paths: imagined vs true displacement over 12 identical moves;
  - over the circle: one imagined move with the cursor on the background vs over the circle.
- Planner sweep on 005 at 2.1M: horizon 12/20, white or colored sampling noise (β = 2, 3), σ0 0.6/1.0.
  None beat the default (`plan.py --beta`, `--sigma0`).
- Failure analysis of the planner (below, `analyze_planner.py`). It leads to vm-00004: the model keeps
  learning from what the planner does.

## Results

All planner numbers are on the 128 test episodes of the PPO evaluation (seed 777000, radius 12–16,
cursor starts outside the circle, 200-step limit, hunger observed). "Median steps" counts a miss as
200 steps.

### R1. The planner on the babbler's world model

| Babbling steps | 005 (click 0.1): success | median steps | 006 (click 0.5): success | median steps |
|---|---:|---:|---:|---:|
| 0 (untrained) | 0.16 | 200 | 0.16 | 200 |
| 65k | 0.39 | 200 | 0.28 | 200 |
| 131k | **0.87** | 11.5 | 0.65 | 24.5 |
| 262k | 0.91 | 81.5 | 0.45 | 200 |
| 393k | 0.88 | 8 | 0.41 | 200 |
| 524k | 0.91 | 124 | 0.89 | 12 |
| 786k | 0.83 | 13 | 0.77 | 13 |
| 1.05M | 0.85 | 11 | 0.91 | 16 |
| 1.57M | 0.88 | 13 | 0.91 | 17 |
| 2.10M | **0.89** | **11** | 0.88 | 8 |

References on the same episodes:

| Policy | Success | Median steps |
|---|---:|---:|
| Random actions, click 0.5 | 0.54 | 170 |
| Random actions, click 0.1 | 0.46 | 200 |
| **Same CEM planner, random costs** (control: same sampling, refitting and warm start) | **0.23** | 200 |
| CEM planner on the untrained model | 0.16 | 200 |
| PPO with reward (vm-00001, end of training) | 1.00 | 5–7 |

The controls do worse than random actions. The CEM's smoothing averages random elites to near-zero
moves, so without information in the costs the cursor barely moves. Everything above 0.23 comes from
the learned model.

### R2. What the latent imagination knows (005)

Cursor read by retrieval, from imagined latents along random action sequences (256 episodes, mean
error in px):

| Babbling steps | step 1 | step 6 | step 12 |
|---|---:|---:|---:|
| 0 | 24.2 | 31.5 | 29.6 |
| 131k | 2.6 | 5.1 | 6.4 |
| 262k | 1.8 | 3.3 | 4.4 |
| 2.10M | **1.5** | **2.8** | **4.3** |
| *no-motion baseline* | *2.8* | *6.5* | *8.5* |
| *real latent (readout floor)* | *1.1* | *1.1* | *1.0* |

- **Systematic shrinkage.** On straight paths (2.1M), imagined displacement is 75–90% of the true one
  at full speed and 53–66% at half speed. On a straight approach to the circle, the imagined cursor
  is 7–13 px off by the time of the click.
- **Movement over the circle (H2).** After one imagined move (2.1M), the cursor error by retrieval is:

  | Cursor | 005 (click 0.1) | 006 (click 0.5) | No-motion baseline |
  |---|---:|---:|---:|
  | on the background | 1.68 px (corr .78/.85) | 1.92 px (.85/.70) | 3.04 |
  | over the circle | **2.56 px (.49/.70)** | 3.32 px (.37/.25) | 3.09 |

  The pixel prediction head (vm-00001's test) shows the same pattern. Over-circle retrieval is .198
  for 005 against .158 for 006 and .112 for vm-00001's babbler. The background reaches .936.
- **Hunger relief (edge test).** Imagined relief after one click with the cursor at a signed
  distance g from the circle's edge (1 = hunger gone, 0 = no change; truth 1 inside and 0 outside):

  | 005 at | g −8 | −3 | −1 | +1 | +2 | +3 | +5 | +12 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 0 | −.03 | −.03 | −.03 | −.03 | −.03 | −.03 | −.03 | −.03 |
  | 262k | .15 | .14 | .12 | .12 | .11 | .10 | .09 | .07 |
  | 524k | .17 | .15 | .12 | .12 | .11 | .10 | .08 | .06 |
  | 786k | .74 | .50 | .32 | .24 | .18 | .11 | .04 | −.02 |
  | 1.05M | .97 | .81 | .62 | **.39** | **.25** | .15 | .01 | −.03 |
  | 2.10M | .99 | .89 | .64 | **.36** | **.20** | .09 | .01 | .00 |

  006 is similar but a step later: a sharp relief only from 1.05M on, .94 / .70 / .44 / .22 / .14 at
  g −8 / −3 / −1 / +1 / +2 at 2.1M.

### R3. Why the planner fails (005 at 2.1M)

Success by the starting gap between the cursor and the circle's edge:

| Gap (px) | 0–8 | 8–16 | 16–24 | 24–32 | 32–48 | > 48 |
|---|---:|---:|---:|---:|---:|---:|
| Planner | 19/19 | 42/43 | 26/32 | 20/24 | 7/8 | 0/2 |
| Random actions (click 0.5) | 16/19 | 24/43 | 18/32 | 9/24 | 2/8 | 0/2 |

The failed episodes all look the same (`analyze_planner.py`; this trace has 16 failures, see below):
- the cursor gets within 0.1–5.5 px of the edge but never enters the circle;
- the planner clicks outside 91–192 times. In successful episodes the median is 5;
- its best plan expects the hunger to vanish: a relative imagined drop of 0.98–1.00, the same as in
  successful episodes.

The planner exploits the blurred edge of the hunger head (R2: .36 relief at 1 px outside). Every
click outside fails, but the model does not learn from that during planning. The same state yields
the same plan, so the agent clicks again.

Re-runs of the planner can differ by a couple of episodes (0.875 vs 0.891 here) when the thread count
changes. Small floating-point differences change which samples become elites.

The sweep confirms this is not a search problem:

| Planner setting (005 at 2.1M) | Success | Median steps |
|---|---:|---:|
| default: horizon 12, white noise, σ0 0.6 | 0.89 | 11 |
| colored noise β 2 | 0.89 | 10 |
| colored noise β 2, σ0 1.0 | 0.88 | 9 |
| colored noise β 3, σ0 1.0 | 0.89 | 9 |
| white noise, σ0 1.0 | 0.82 | 11 |
| horizon 20 | 0.81 | 11 |
| horizon 20, colored noise β 2, σ0 1.0 | 0.88 | 9 |

## Analysis

1. **A world model learned without reward supports goal-directed action.** The agent babbled with a
   frozen random policy and was trained only to predict its next latent, frame and hunger. Planning
   to lower its own predicted hunger, it then hits 89% of test episodes in a median of 11 steps.
   Random actions hit 46–54% in a median of 170–200 steps, and the same planner with uninformative
   costs hits 23%. No reward, value function or trained policy is involved. PPO, with 1M+ steps of
   reward, is still better: 100% in 5–7 steps.
2. **The model's knowledge arrives in stages, and planning follows it.**
   - **Cursor dynamics first (131–262k):** imagined cursors beat the no-motion baseline at every
     horizon.
   - **A soft proximity gradient next (262–524k):** clicks near the circle are imagined to relieve
     a little hunger (.12–.17 inside against .06–.09 at 5–12 px outside). That is enough to hit 91%, but
     slowly (median 82–124 steps): the planner follows the gradient and clicks along the way.
   - **A sharp inside/outside contingency last (786k–1M):** with it the planner becomes fast
     (median 8–13).
3. **Lower click probability (H2) helps, but does not solve it.**
   - The click-0.1 babbler supports planning earlier: 0.87 vs 0.65 success at 131k, 0.91 vs 0.45
     at 262k, 0.88 vs 0.41 at 393k. Its sharp hunger relief appears one checkpoint earlier.
   - It predicts movement over the circle better: latent error 2.56 vs 3.32 px, pixel-head
     retrieval .20 vs .16.
   - Over the circle it is still far from the background (1.7 px). This is one seed per arm.
4. **The failures are model exploitation, not search.**
   - The hunger head's inside/outside boundary is blurred over about ±3 px, and the imagined
     cursor carries 1.5–4 px of error.
   - A planner that minimizes imagined hunger therefore finds points just outside the edge that
     "work" in imagination, and gets stuck there.
   - Longer horizons and different sampling noise do not change this (0.81–0.89).
   - The agent does observe its failed clicks. An agent that learns from its own actions should
     correct this. That is vm-00004.
5. **Imagination is biased toward smaller moves.** Imagined displacement is 53–90% of the truth,
   most of all for moderate moves. That is the usual regression to the mean of an MSE-trained
   model on random babbling. The planner partly compensates by choosing larger actions.

## Conclusions

- Without any reward, motor babbling plus prediction teaches a world model that an agent can use
  to act: 89% hits against 23% for the same planner with uninformative costs.
- The latent imagination tracks the cursor over 12 steps: 4.3 px error against 8.5 for no motion.
  It learns, in order: movement, a proximity gradient for hunger, then the sharp click-inside
  contingency.
- The remaining failures come from the planner exploiting the blurred inside/outside boundary of
  the hunger head. The agent repeats a click that its own experience shows does not work.
- Babbling with fewer clicks speeds up the first useful model, and improves but does not fix
  movement prediction over the circle (single seed).

## Next Steps

- vm-00004: train the world model on the planner's own experience, with the same reward-free
  prediction losses (`train.actor=planner`). The control is babbling for as long.
- Seeds 2 and 3 for 005/006.
- An ensemble of hunger heads with pessimistic planning, as a model-side alternative to learning
  from failures.
- Babbling with temporally correlated actions, against the shrinkage of long moves.
