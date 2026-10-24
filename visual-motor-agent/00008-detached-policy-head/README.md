# vm-00008: Detached Policy Head: Keep the Move Loss Out of the Shared Features

**Date:** 2026-10-24
**Status:** Complete
**Source:** `exp/vm-00008` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00008/`
**Task plan:** [`../tasks/00008-detached-policy-head.md`](../tasks/00008-detached-policy-head.md)

## Goal

The basic agent (vm-00006, vm-00007) never learned to aim, and its features lost the cursor during
training. The planner agent's features kept it. One difference: the basic agent's move loss (it
learns to predict its own sampled moves) also trains the shared encoder and Transformer.

Here the move loss trains only the policy head. The features are then shaped by prediction alone,
as in the planner agent. The questions:
- Is the move loss what erodes the features?
- With features that hold the cursor and the circle, does the agent aim, and does the hindsight
  hunger command steer it?

## Hypothesis

1. **The move loss erodes the features.** With its gradient stopped at the policy head's input z:
   - with inverse dynamics, the visual features v keep the cursor (R² ≥ 0.9, as in the planner
     agent) and z keeps the direction to the circle;
   - without inverse dynamics, nothing teaches v the cursor, but the circle is no longer erased
     (R² near the random encoder's 0.94, against 0.28–0.39 in vm-00007).
2. **With such features, the agent aims.** Its aim cosine rises clearly above vm-00007's −0.05 to
   +0.04, and its test success beats the random initial model (0.44–0.48).
3. **A linear head cannot use the command.** It can aim, but it cannot make aiming depend on the
   command unless z holds their product. A head with a hidden layer (stage 2) can.

## Method

- **Detached policy head** (`ppo.pi_detach=1`): the move loss updates the head (its weights and the
  move std); no gradient from it reaches z, the Transformer or the encoder.
- **Agent:** vm-00007's basic agent (`configs/AHR_gated.json`):
  - a window of 8 real states; predicts the next move, frame and drives;
  - hindsight hunger input; 20% babbled moves; batch 8;
  - 50k steps, 4 threads, seeds 1–2.
- **Arms**, each under memory U (store everything) and R (relief chunks plus 10% of the rest):

  | Arm | Policy head | Inverse dynamics | Controls (vm-00007) |
  |---|---|---|---|
  | D | linear, detached | no | U, R (runs 001–004) |
  | DI | linear, detached | yes | I+U, I+R (runs 017–020) |
  | DMI (stage 2) | hidden layer of 128, detached | yes | DI |

  With the defaults the code is unchanged, so vm-00007's runs are exact controls.
- **Evaluation:**
  - own moves on vm-00002's 128 test episodes every 10k steps (`online.py --eval`);
  - linear probes of the cursor, the circle and the direction from v and z at 10k, 30k and 50k
    (`analyze_self_actor.py probe`);
  - aim cosine, steps at a wall, and the effect of asking for no relief at 30k and 50k
    (`analyze_self_actor.py aim`);
  - online hits and memory composition from the logs.

## Log

### 2026-10-24
- **Implemented `ppo.pi_detach`** (C++ `PPOConfig`, ctypes, PyTorch reference): the policy head's
  backward gets a null input gradient. Both update paths share that code.
- **Verification:**
  - gradient tests against the reference for the detached move loss, alone (with inverse dynamics)
    and with PPO, at 4 and 16 threads: 32 new cases pass;
  - negative control: the previous library fails exactly these 32 cases;
  - with only the move loss and `pi_detach=1`, every gradient outside the head is exactly zero
    (16 cases), and the previous library fails all of them;
  - with the defaults, 300-step runs of the basic agent (with and without inverse dynamics) give
    byte-identical checkpoints before and after the change;
  - `make test_all` passes (338 Python tests).
- Also fixed `bench_vm_agent.cpp`, which left `PPOConfig` (and `VMConfig`) fields uninitialized.
- **Launched D and DI** (`make detached`, runs 001–008) at 11:15, next to the vm-00007 follow-up
  (8 planner runs).
- **Implemented `model.pi_hidden`** (stage 2): π = W2 relu(W1 z + b1) + b2, computed by one function
  for training, `vm_forward` and acting. The new tensors `pi_h_w`, `pi_h_b` have size 0 when it is
  off, so the layout, the initialization and old checkpoints are unchanged (header slot 37).
- **Verification of stage 2:**
  - a new test configuration with the hidden layer runs every reference test: 26 gradient cases
    (including the detached ones), acting equals training, causality, checkpoint round trip and the
    head-only test;
  - with the defaults, 300-step runs give byte-identical checkpoints with and without the change, for
    the basic agent, the planner agent (latent dynamics) and the running DI configuration;
  - old checkpoints load (vm-00007 runs 009 and 017);
  - `make test_all` passes (370 Python tests).
- **Launched DMI** (`make mlp`, runs 009–012) at 11:22.
- **Feature probes at 10k** (linear R², seeds 1, 2; the controls are vm-00007's runs at the same
  step):

  | Arm | z: cursor | z: circle | z: direction | v: cursor | v: circle | v: direction |
  |---|---|---|---|---|---|---|
  | U (move loss into the features) | 0.51, 0.46 | 0.10, 0.22 | 0.38, 0.38 | 0.62, 0.58 | 0.20, 0.34 | 0.46, 0.51 |
  | D, memory U | 0.36, 0.32 | 0.08, 0.13 | 0.32, 0.26 | 0.43, 0.37 | 0.10, 0.17 | 0.37, 0.31 |
  | R | 0.54, 0.44 | 0.08, 0.26 | 0.41, 0.35 | 0.65, 0.56 | 0.17, 0.37 | 0.51, 0.48 |
  | D, memory R | 0.50, 0.41 | 0.07, 0.32 | 0.45, 0.34 | 0.54, 0.48 | 0.07, 0.33 | 0.47, 0.39 |
  | I+U | 0.60, 0.49 | 0.25, 0.18 | 0.48, 0.38 | 0.67, 0.58 | 0.34, 0.29 | 0.54, 0.48 |
  | DI, memory U | 0.86, 0.83 | 0.29, 0.45 | 0.67, 0.69 | 0.88, 0.84 | 0.35, 0.48 | 0.70, 0.70 |
  | I+R | 0.65, 0.49 | 0.17, 0.18 | 0.50, 0.39 | 0.67, 0.63 | 0.25, 0.32 | 0.54, 0.53 |
  | DI, memory R | 0.76, 0.74 | 0.22, 0.44 | 0.60, 0.62 | 0.79, 0.74 | 0.24, 0.48 | 0.63, 0.62 |
  | planner agent (vm-00005 L s1) | 0.95 | 0.56 | 0.79 | 0.97 | 0.62 | 0.80 |

  - **With inverse dynamics, detaching helps a lot:** the cursor from v rises from 0.58–0.67 to
    0.74–0.88, the direction from 0.48–0.54 to 0.62–0.70.
  - **Without it, detaching hurts:** the cursor from v falls from 0.56–0.65 to 0.37–0.54. Early on,
    the move loss also teaches the cursor (predicting one's own moves needs it, e.g. at a wall). It
    gets in the way only of a better teacher.
  - **The circle is weak in every arm** (0.07–0.48 against 0.94 for the random encoder), detached or
    not. Erasing it does not need the move loss.
  - The test is whether this holds or decays by 50k (vm-00007: I+U's cursor fell from 0.67 to 0.52).
- **No arm aims at 10k** (`analyze_self_actor.py aim`, the test episodes):
  - the aim cosine is −0.06 to +0.04 in D, and −0.28 to −0.09 in DI: the arm whose features hold the
    direction best moves *away* from the circle;
  - asking for no relief instead changes it by at most 0.015;
  - the cursor sits at a wall in 29–71% of the steps (23% at birth).
  - **A likely reason:** the head imitates all of the agent's own moves, and the memory over-represents
    moves that keep episodes going. Moves toward the circle end episodes within steps; moves away from
    it, or into a wall, fill 200-step misses. With features that hold the direction, the head can
    reproduce "away" as well as "toward". A linear head cannot use the command to tell the moves before a
    hit from the rest; DMI's hidden layer can.
- **DMI at 10k** (seeds 1, 2):
  - **features** as in DI: cursor from v 0.86–0.92 (memory U) and 0.72–0.86 (R), direction 0.59–0.75;
  - **aim cosine** −0.07 to +0.06. The command now matters: asking for no relief changes it by up to
    0.11 (DI: at most 0.015), but not consistently toward the circle;
  - **test success** of its own moves: U 0.59 and 0.44, R 0.48 and 0.45, against 0.45 and 0.45 at
    birth (the hidden layer changes the initial draws). U seed 1 (median 114 steps, 21 clicks per
    episode, at a wall in 33% of the steps) is the first basic agent above its birth level this early.
    One seed at one checkpoint.
- **Feature probes at 30k** (seeds 1, 2; change from 10k in brackets):

  | Arm | v: cursor | v: circle | z: direction |
  |---|---|---|---|
  | U | 0.53, 0.47 (−0.09, −0.11) | 0.28, 0.39 | 0.28, 0.29 (−0.10, −0.09) |
  | D, memory U | 0.50, 0.38 (+0.07, +0.01) | 0.00, 0.83 | 0.35, 0.37 (+0.03, +0.11) |
  | R | 0.57, 0.55 (−0.08, −0.01) | 0.19, 0.36 | 0.34, 0.32 (−0.07, −0.03) |
  | D, memory R | 0.66, 0.63 (+0.12, +0.15) | 0.09, 0.71 | 0.45, 0.49 (0.00, +0.15) |
  | I+U | 0.57, 0.44 (−0.10, −0.14) | 0.37, 0.35 | 0.35, 0.21 (−0.13, −0.17) |
  | DI, memory U | 0.89, 0.91 (+0.01, +0.07) | 0.43, 0.47 | 0.70, 0.73 (+0.03, +0.04) |
  | I+R | 0.57, 0.48 (−0.10, −0.15) | 0.26, 0.27 | 0.33, 0.24 (−0.17, −0.15) |
  | DI, memory R | 0.81, 0.82 (+0.02, +0.08) | 0.34, 0.57 | 0.59, 0.66 (−0.01, +0.04) |

  - **With the move loss in the features, they decay** from 10k to 30k in all 8 runs: the cursor
    by 0.01–0.15, the direction by 0.03–0.17.
  - **Detached, they hold or improve** in all 8 runs. DI keeps the cursor at 0.81–0.91 and the
    direction at 0.59–0.73 (the planner agent at 100k: 0.96 and 0.84).
  - Without inverse dynamics, the detached features catch up: by 30k, D-R's cursor (0.63–0.66) is
    above every control's.
  - The circle varies by seed: detached seed 2 recovers it (0.71–0.83), seed 1 does not
    (0.00–0.09).
- **Feature probes at 50k** (seeds 1, 2; vm-00007's controls at the same step):

  | Arm | v: cursor | v: circle | z: direction |
  |---|---|---|---|
  | U | 0.46, 0.42 | 0.28, 0.39 | 0.21, 0.21 |
  | D, memory U | 0.53, 0.53 | 0.52, 0.74 | 0.41, 0.49 |
  | R | 0.50, 0.46 | 0.16, 0.39 | 0.21, 0.26 |
  | D, memory R | 0.63, 0.62 | 0.25, 0.74 | 0.43, 0.50 |
  | I+U | 0.52, 0.42 | 0.40, 0.33 | 0.30, 0.16 |
  | DI, memory U | 0.89, 0.90 | 0.37, 0.40 | 0.69, 0.72 |
  | I+R | 0.52, 0.45 | 0.27, 0.32 | 0.24, 0.22 |
  | DI, memory R | 0.79, 0.87 | 0.40, 0.51 | 0.56, 0.66 |

  - The controls keep decaying (cursor 0.42–0.52, direction 0.16–0.30). The detached features hold, and
    without inverse dynamics the circle comes back in 3 of 4 runs (0.52–0.74).
- **Aim at 50k:** D −0.01, +0.12 (memory U; +0.29 within 24 px of the circle), −0.02, +0.03 (R).
  DI −0.19, −0.20 (U), −0.06, +0.05 (R). At a wall in 33–63% of the steps.
- **Test success of the agents' own moves** (seeds 1, 2; birth 0.48, 0.44):

  | Arm | 10k | 20k | 30k | 40k | 50k | mean, 10k–50k |
  |---|---|---|---|---|---|---|
  | D, memory U | 0.43, 0.27 | 0.35, 0.39 | 0.37, 0.48 | 0.33, 0.54 | 0.39, 0.55 | 0.37, 0.45 |
  | D, memory R | 0.38, 0.37 | 0.26, 0.50 | 0.30, 0.45 | 0.38, 0.47 | 0.38, 0.48 | 0.34, 0.45 |
  | DI, memory U | 0.27, 0.18 | 0.43, 0.23 | 0.41, 0.22 | 0.25, 0.29 | 0.25, 0.25 | 0.32, 0.23 |
  | DI, memory R | 0.27, 0.30 | 0.23, 0.34 | 0.24, 0.44 | 0.25, 0.46 | 0.29, 0.41 | 0.25, 0.39 |

  The vm-00007 controls' means (from their stored evaluations): U 0.23, 0.16; R 0.13, 0.22; I+U 0.33,
  0.17; I+R 0.17, 0.22.
- **DMI, all checkpoints (14:05): the basic agent aims for the first time.**
  - **Test success** (seeds 1, 2; birth 0.45, 0.45): memory U 0.59/0.44 (10k), 0.69/0.52, 0.79/0.71,
    0.81/0.59, 0.82/0.81 (50k), means 0.74, 0.61; memory R 0.48/0.45, 0.47/0.42, 0.76/0.52, 0.84/0.60,
    0.81/0.63, means 0.67, 0.53. At 50k: median 39–102 steps (R seed 1: 31 at 40k), 10–24 clicks per
    episode.
  - **Aim cosine at 50k:** +0.28, +0.13 (memory U), +0.17, +0.04 (R); at 30k +0.20, +0.13, +0.13,
    +0.01. At a wall in 15–28% of the steps (birth 23%).
  - **The command steers it:** asking for no relief instead lowers the cosine by 0.04–0.12 at 50k
    (+0.28 → +0.16, +0.13 → +0.01, +0.17 → +0.10, +0.04 → −0.01).
  - **Features:** cursor from v 0.88–0.91, circle 0.45–0.56, direction 0.71–0.77, at 30k and 50k.
- **D and DI aim at 30k:** still near zero. D: −0.02, +0.07 (memory U), +0.04, +0.08 (R), and +0.06 to +0.13
  within 24 px of the circle. DI: −0.03, −0.11, −0.12, −0.04. At a wall in 30–73% of the steps.
  Better features alone do not make this agent aim.
- **Online hits, 5k–10k** (seeds 1, 2; 20% babbled moves; random babbling hits 0.47–0.48):
  - with the move loss in the features (vm-00007): U 0.20, 0.11; R 0.00, 0.31; I+U 0.17, 0.08;
    I+R 0.11, 0.18;
  - detached: D-U 0.31, 0.44; D-R 0.34, 0.39; DI-U 0.27, 0.07; DI-R 0.21, 0.14; DMI-U 0.70, 0.27;
    DMI-R 0.45, 0.38.

## Results

### R1. Test success of the agent's own moves (vm-00002's 128 test episodes; seeds 1, 2)

| Arm | birth | 10k | 20k | 30k | 40k | 50k | mean, 10k–50k |
|---|---|---|---|---|---|---|---|
| U (vm-00007) | 0.48, 0.44 | 0.15, 0.11 | 0.22, 0.16 | 0.26, 0.20 | 0.22, 0.17 | 0.29, 0.16 | 0.23, 0.16 |
| D, memory U | 0.48, 0.44 | 0.43, 0.27 | 0.35, 0.39 | 0.37, 0.48 | 0.33, 0.54 | 0.39, 0.55 | 0.37, 0.45 |
| R (vm-00007) | 0.48, 0.44 | 0.05, 0.23 | 0.14, 0.20 | 0.13, 0.19 | 0.17, 0.25 | 0.13, 0.25 | 0.13, 0.22 |
| D, memory R | 0.48, 0.44 | 0.38, 0.37 | 0.26, 0.50 | 0.30, 0.45 | 0.38, 0.47 | 0.38, 0.48 | 0.34, 0.45 |
| I+U (vm-00007) | 0.48, 0.44 | 0.12, 0.12 | 0.36, 0.21 | 0.34, 0.15 | 0.42, 0.21 | 0.41, 0.16 | 0.33, 0.17 |
| DI, memory U | 0.48, 0.44 | 0.27, 0.18 | 0.43, 0.23 | 0.41, 0.22 | 0.25, 0.29 | 0.25, 0.25 | 0.32, 0.23 |
| I+R (vm-00007) | 0.48, 0.44 | 0.20, 0.15 | 0.11, 0.09 | 0.18, 0.27 | 0.23, 0.34 | 0.16, 0.25 | 0.17, 0.22 |
| DI, memory R | 0.48, 0.44 | 0.27, 0.30 | 0.23, 0.34 | 0.24, 0.44 | 0.25, 0.46 | 0.29, 0.41 | 0.25, 0.39 |
| **DMI, memory U** | 0.45, 0.45 | 0.59, 0.44 | 0.69, 0.52 | 0.79, 0.71 | 0.81, 0.59 | **0.82, 0.81** | **0.74, 0.61** |
| **DMI, memory R** | 0.45, 0.45 | 0.48, 0.45 | 0.47, 0.42 | 0.76, 0.52 | 0.84, 0.60 | **0.81, 0.63** | **0.67, 0.53** |

- With 128 episodes, one success rate has a standard error of about 0.04.
- The hidden layer changes the initial random draws, so DMI's birth differs.
- DMI at 50k: median 39–102 steps per episode (memory R, seed 1: 31 at 40k, 39 at 50k), 10–24 clicks per
  episode.

### R2. Online hits in steps 40k–50k (seeds 1, 2; 20% of the moves are babbled)

| Arm | memory U | memory R |
|---|---|---|
| move loss in the features (vm-00007) | 0.40, 0.26 | 0.31, 0.36 |
| … with inverse dynamics (vm-00007) | 0.41, 0.27 | 0.23, 0.34 |
| D | 0.54, 0.46 | 0.39, 0.56 |
| DI | 0.35, 0.31 | 0.23, 0.37 |
| **DMI** | **0.78, 0.74** | **0.70, 0.52** |

Random babbling hits in 0.47–0.48 of its episodes (vm-00005 B). Over its whole life, DMI with memory U,
seed 1, hit in 371 of 499 episodes.

### R3. Features (linear R² on 1,024 held-out random states; seeds 1, 2)

| Arm | v: cursor 10k / 30k / 50k | z: direction 10k / 30k / 50k |
|---|---|---|
| move loss in the features (U, R; vm-00007) | 0.56–0.65 / 0.47–0.57 / 0.42–0.50 | 0.35–0.41 / 0.28–0.34 / 0.21–0.26 |
| … with inverse dynamics (I+U, I+R) | 0.58–0.67 / 0.44–0.57 / 0.42–0.52 | 0.38–0.50 / 0.21–0.35 / 0.16–0.30 |
| D | 0.37–0.54 / 0.38–0.66 / 0.53–0.63 | 0.26–0.45 / 0.35–0.49 / 0.41–0.50 |
| DI | 0.74–0.88 / 0.81–0.91 / 0.79–0.90 | 0.60–0.69 / 0.59–0.73 / 0.56–0.72 |
| DMI | 0.72–0.92 / 0.87–0.91 / 0.88–0.91 | 0.55–0.72 / 0.68–0.76 / 0.68–0.74 |
| planner agent (vm-00005 L s1, 100k) | 0.96 | 0.84 |

The circle from v at 50k: controls 0.16–0.40; D 0.25–0.74; DI 0.37–0.51; DMI 0.45–0.56 (random encoder:
0.94).

### R4. Aim at 50k (cosine between the mean move and the direction to the circle, away from the walls)

| Arm | asking for relief | asking for none | within 24 px | at a wall |
|---|---|---|---|---|
| D | −0.01, +0.12, −0.02, +0.03 | −0.01, +0.08, +0.01, +0.03 | −0.04 to +0.29 | 33–49% |
| DI | −0.19, −0.20, −0.06, +0.05 | −0.26, −0.20, −0.09, +0.04 | −0.23 to +0.13 | 49–63% |
| **DMI** | **+0.28, +0.13, +0.17, +0.04** | +0.16, +0.01, +0.10, −0.01 | +0.03 to +0.24 | 17–28% |

Order: memory U seeds 1, 2, then memory R seeds 1, 2. vm-00007's agents: −0.05 to +0.04, and the command
changed it by at most 0.03. Birth: +0.02, −0.01, with 23% of the steps at a wall.

## Analysis

### The move loss erodes the features (hypothesis 1, confirmed for the cursor, not for the circle)

- **With the move loss in the features, they decay** in all 8 control runs from 10k to 50k: the cursor
  from v by 0.10–0.18, the direction from z by 0.09–0.26.
- **Detached, they hold (within 0.04) or improve** in all 12 runs. With inverse dynamics they come close
  to the planner agent's: the cursor 0.79–0.91 against 0.96, the direction 0.56–0.74 against 0.84.
- **Early on, the move loss also teaches.** Without inverse dynamics, the detached features start below
  the controls' at 10k (cursor 0.37–0.54 against 0.56–0.65) and overtake them by 50k (0.53–0.63
  against 0.42–0.50). Predicting one's
  own moves needs the cursor (at a wall, for example), but it wears the features down over time.
- **The circle is not the move loss's doing.** It falls from the random encoder's 0.94 to 0.07–0.48 by 10k
  in every arm, detached or not, and comes back only partly (0.25–0.74 at 50k). Prediction with
  residual frames and U-Net skips does not need it in v: the circle is static, and the decoder sees the
  encoder's maps directly.

### Features alone do not make it aim (hypothesis 2, not with a linear head)

- **Detaching alone roughly doubles test success without inverse dynamics** (D: means 0.34–0.45, against
  0.13–0.23), but the agent barely aims (cosine −0.02 to +0.12) and sits at a wall in a third or more of the
  steps.
- **With the best linear-head features, it aims away** (DI: −0.19 and −0.20 with memory U at 50k). Test
  success is no better than the controls'.
- **A likely reading:** a linear head imitates all of the agent's moves, and the memory over-represents
  moves that keep episodes going. Moves toward the circle end the episode within steps; moves away from it
  or into a wall fill 200-step misses. Once z holds the direction, a linear head can reproduce "away" as
  readily as "toward", and it cannot use the command to separate the moves before a hit from the rest.

### A head with a hidden layer lets the command steer (hypothesis 3, confirmed)

- **DMI aims** in 3 of 4 runs (+0.13 to +0.28) and **asking for relief aims more** than asking for none
  in every run, by 0.04–0.12. Among the linear heads, the command changes the aim by at most 0.03 in 6 of 8
  runs (0.04 and 0.07 in the other two), and in vm-00007 by at most 0.03.
- **It is the first basic agent that clearly beats its random initial model:** 0.63–0.82 at 50k against
  0.45, rising through life in every run, and 0.52–0.78 of its online episodes hit in its last 10k steps,
  above random babbling (0.47–0.48).
- **What it learned:** to imitate its own moves *given* the relief that followed them (the hindsight
  command), and then to ask for relief. A linear readout of z cannot form "aim, if relief is asked for";
  one hidden layer can.
- **No reward and no planner.** The only signal is the agent's own hunger, used as a label for its own past
  moves.

### Memory

- Under DMI, storing everything (U) did as well as storing relief (R) or better: means 0.74 and 0.61
  against 0.67 and 0.53 (2 seeds). As in vm-00007, the relief gate does not help the basic agent.

### Caveats

- 2 seeds per arm; seeds differ by up to 0.2 in test success.
- **The detached head still shares the gradient-norm clip** with the rest of the network (found in
  vm-00009). Its move-loss gradient raises the global norm, so every update of the world model is scaled
  down more, and by a varying amount. `ppo.pi_own_clip` (added afterwards, default off) removes this
  coupling; these runs were made without it.
- DMI differs from DI in the hidden layer and in its initial weights.

## Conclusions

1. **The basic agent failed in vm-00006/7 for two reasons:** its move loss eroded the features, and its
   linear policy head could not use the hindsight command.
2. **With both fixed** (a detached head with one hidden layer, features from frame, drive and inverse-
   dynamics prediction), it learns goal-directed moves from its own experience, without reward and without
   a planner: test success 0.63–0.82 at 50k against 0.45 at birth, aiming at the circle, steered by the
   command.
3. **Detaching alone** keeps the features but gives at most weak aiming. Better features with a linear head
   can even make the habit aim away.
4. **The circle's position fades from the features in every arm.** The move loss is not the cause.

## Next Steps

- **Longer and wider:** DMI for 200k steps and more seeds. Does it approach the planner agent (vm-00005 L)?
- **Isolate the head fully:** DMI with `ppo.pi_own_clip=1`.
- **Which part matters:** the hidden layer without inverse dynamics (no DM arm yet).
- **With distractors and patch tokens:** vm-00010 runs DMI with moving dots (CMDI) and a patch-token agent
  with the same head (PMD).
- **Compare with the planner as teacher** (vm-00009): a habit learned from the planner's moves, against
  this one learned from the agent's own moves.
