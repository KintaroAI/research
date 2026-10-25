# vm-00012: The Planner Agent in the Continual Torus World, With Hazards

**Date:** 2026-10-25
**Status:** Complete
**Source:** `exp/vm-00012` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00012/`
**Task plan:** [`../tasks/00016-planner-continual-world.md`](../tasks/00016-planner-continual-world.md)

## Goal

vm-00011 put the basic agent in a continual world: on a torus it forages, with an egocentric view it solves the task,
and it never avoids hazards, because its learning signal (the hindsight hunger command) marks only the moves before
relief. The planner agent (vm-00005's L) chooses each move by imagining 12 steps of hunger with its learned latent
dynamics and picking the plan with the least. If its hunger model learns that hazards raise hunger, it should steer
around them. Here it lives in the same worlds, on the GPU (task 00015).

## Hypothesis

1. The planner agent forages better than the basic agent in every world.
2. It avoids hazards: fewer steps inside one than chance and than the basic agent.
3. Moving circles and the egocentric view are no harder for its latent dynamics than the plain torus.

## Method

- **Agent:** L (`configs/L_gated.json`: one-frame model, inverse and latent dynamics, CEM planner over 12 steps, 5%
  babbled moves and babbling when it imagines no gain, memory U), `online.device=cuda` with the CUDA multi-process
  service, 40k steps, seeds 1–3.
- **Worlds** (vm-00011's options): LT (continual torus), LTH (+ 2 hazards, 0.02 hunger per step inside), LTM (+ moving
  circles, 0.5 px per step), LTE (+ egocentric view). Follow-up: LTH10 (hazards 0.1 per step), with vm-00011's DMI in
  the same world (DTH10, 60k steps).
- **Evaluation:** online hits per 1,000 steps and the share of steps inside a hazard. `analyze_hazards.py hazard` (new): the
  hunger rise the model imagines over 12 steps of standing still with the cursor inside a hazard and outside every
  hazard, against the world's (256 placed states; the true positions place the states, the model sees only frames
  and drives).

## Log

### 2026-10-25
- **Launched** the 12 lifetimes at 01:08: 42 steps/s each on the shared GPU (the CPU planner agent: 3.5 under load).
  Finished at 01:24.
- **Results** (online hits per 1,000 steps; seeds 1, 2, 3):

  | Arm | steps 0–5k | 10k–20k | 30k–40k | in a hazard, 30k–40k |
  |---|---|---|---|---|
  | LT | 23.2, 26.2, 32.2 | 24.5, 25.1, 25.7 | 28.1, 32.8, 30.9 | – |
  | LTH | 30.4, 34.0, 35.2 | 25.0, 21.3, 20.9 | 25.1, 24.6, 27.8 | 0.070, 0.058, 0.068 |
  | LTM | 33.4, 33.8, 28.8 | 23.6, 21.6, 23.2 | 32.0, 26.1, 32.0 | – |
  | LTE | 106.0, 126.8, 113.0 | 136.2, 144.2, 144.2 | **141.9, 134.8, 141.6** | – |

  The basic agent in the same worlds (vm-00011, 50k–60k, 20% babbled): T 8.7–9.8, TH 6.8–9.5 (in a hazard 0.06–0.09),
  TM 6.3–8.3, TE 17.3–19.3.
- **The planner's hunger model does not see the hazards** (`analyze_hazards.py`, LTH; imagined hunger rise over 12
  still steps):

  | Seed | at 20k: inside, outside (extra) | at 40k: inside, outside (extra) | true: inside, outside (extra) |
  |---|---|---|---|
  | 1 | +0.453, +0.437 (+0.016) | +0.030, +0.052 (−0.021) | +0.245, +0.062 (+0.183) |
  | 2 | +0.144, +0.171 (−0.027) | +0.050, +0.063 (−0.013) | |
  | 3 | +0.046, +0.071 (−0.025) | +0.109, +0.115 (−0.006) | |

  By 40k the imagined rise outside a hazard is close to the truth (0.03–0.12 against 0.06), but inside a hazard the
  model imagines no extra rise at all. A hazard adds 0.02 hunger per step, on 6–7% of the steps; a hit removes up to
  1.0. In squared error the hazard's effect is negligible next to a hit's, so the hunger head has no reason to learn
  it. The planner therefore has nothing to steer around.
- **Launched the follow-up** with hazards five times as costly (0.1 per step inside): LTH10 (planner) and DTH10 (DMI),
  seeds 1–3, runs 013–018.
- **Strong hazards (0.1 per step) do not change it** (runs 013–018, 40k for the planner, 60k for DMI; last 10k steps):

  | Arm | hits per 1,000 steps | in a hazard (0–5k → last 10k) |
  |---|---|---|
  | LTH10 (planner) | 23.1, 23.0, 22.6 | 0.071 → 0.058, 0.060 → 0.055, 0.076 → 0.067 |
  | DTH10 (DMI) | 6.7, 8.2, 7.8 | 0.064 → 0.067, 0.063 → 0.055, 0.081 → 0.071 |

  The planner's imagined extra rise inside a hazard at 40k: −0.009, +0.013, +0.008, against the world's +0.630.
- **Why: nothing in the network represents hazard contact** (LTH10 at 40k; 4,096 placed states, half with the cursor
  inside a hazard, `analyze_hazards.py contact`):
  - a linear probe of "the cursor is inside a hazard": R² 0.07, 0.00, −0.05 from v and 0.10, 0.04, −0.03 from z;
  - the drives head, whose gradient does train the encoder, predicts the same next hunger inside and outside
    (0.440 vs 0.440, 0.495 vs 0.512, 0.507 vs 0.494; the world: +0.105 vs +0.005).
  Contact is a conjunction of two positions, like a hit. But a hit changes hunger by up to 1.0, while a hazard step
  changes it by 0.1: in squared error a hazard event weighs about a hundredth of a hit. The latent heads cannot
  shape v (v is detached by design, vm-00002), and the drives loss is too weak to make the encoder represent contact.
  This is vm-00001's problem in the drives: plain pixel MSE ignored the 1%-of-pixels cursor until changed pixels were
  weighted.

## Results

| World | planner agent L, 30k–40k | basic agent DMI (vm-00011), 50k–60k |
|---|---|---|
| continual torus | 28.1–32.8 hits per 1,000 steps | 8.7–9.8 |
| + egocentric view | **134.8–141.9** | 17.3–19.3 |
| + moving circles | 26.1–32.0 | 6.3–8.3 |
| + hazards 0.02 / step | 24.6–27.8; in a hazard 0.058–0.070 | 6.8–9.5; 0.06–0.09 |
| + hazards 0.1 / step | 22.6–23.1; 0.055–0.067 | 6.7–8.2; 0.055–0.071 |

The hazards cover about 6.5% of the screen.

## Analysis

- **The planner agent forages three to seven times better than the basic agent** in every world (hypothesis 1), from
  the first 5k steps on (23–35 hits per 1,000 steps).
- **The egocentric view lifts it to one hit every seven steps**, and moving circles cost it nothing (hypothesis 3).
  With the cursor at the center, every move shifts the whole scene the same way, so its one-frame latent dynamics
  have a single, translation-like effect to learn.
- **Neither agent avoids hazards, at either cost** (hypothesis 2, not supported). For the planner the reason is
  measurable: its hunger model imagines no extra hunger inside a hazard, because no feature and no head represents
  contact. A hazard step is too small an error to be learned beside the hits.

## Caveats

- 3 seeds, 40k steps; online measures include the babbled moves (5% for L, 20% for DMI); no frozen planner test in
  the continual worlds.
- Hazard costs of 0.02 and 0.1 per step only; an extra weight on unusual drive changes is untested.

## Conclusions

1. **The planner agent, with a learned world model and a search over imagined hunger, forages far better than the
   basic agent** in a continual world, and with an egocentric view it hits every seventh step.
2. **Hazards are invisible to both**: hazard contact is never represented, because its effect on hunger is too small
   to matter in the losses. Avoiding harm needs the representation first.

## Next Steps

- **Weight unusual drive changes** (a `drive_changed_weight`, like `pred_changed_weight` for pixels): steps where hunger
  changes other than by its steady rise count more, in the drives loss and in the latent hunger loss. Then repeat
  LTH and DTH.
- **A two-sided hindsight command** for the basic agent: the achieved change of hunger over the next 12 steps, not only
  relief.
