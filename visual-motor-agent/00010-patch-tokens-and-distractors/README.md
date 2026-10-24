# vm-00010: Patch Tokens: One Token per Image Patch and per Input, in a World With Moving Dots

**Date:** 2026-10-24
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00010`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00010/`
**Task plan:** [`../tasks/00010-patch-tokens-and-distractors.md`](../tasks/00010-patch-tokens-and-distractors.md)

## Goal

So far the CNN squeezes each frame into one 128-number vector, and the drives, the previous move and the
command are fused with it into a single token per step. In the basic agent that vector loses the circle
and often the cursor (vm-00006 to vm-00008), and small inputs can drown in it.

Here each 16×16 patch of the frame is its own token, with a position embedding for its place on the
screen, and the drives, the previous move and the command get one token each. Each token predicts its
own part of the next state. Small grey dots now move around the world on their own, so the patches hold
motion that the agent does not cause, and the agent has to work out which moving thing obeys it.

## Hypothesis

1. **The cursor and the circle stay readable,** and the direction to the circle is readable from the
   move token's output, better than from vm-00008's CNN vector.
2. **Patch tokens couple where things change.** When the cursor or a dot moves from patch A into
   patch B, B's token attends to A's far more than to other patches.
3. **Self versus other.** Reversing the executed move changes the predicted next frame around the
   cursor, not around the dots.
4. Open: whether the basic agent then aims.

## Method

- **World:** `env.distractors=8`: grey 2×2-px dots (brightness 60–180) moving at 0.5–2 px per step in
  random directions, bouncing off the walls, drawn under the circle and the cursor. They never affect the
  task, and their random streams are separate, so the episodes are the same with or without them.
- **Model** (`model.patch=16`), per step 19 tokens:
  - 16 patch tokens: a shared linear embedding of the raw 16×16×3 pixels, plus a learned position vector
    for its cell that starts as a smooth 2D sine-cosine code;
  - one token each for the drives, the previous move and the command;
  - a type embedding per kind of token, and the time embedding of the step.
  - Attention is block-causal over the 8-step window: a token sees every token of its own step and of the
    earlier steps.
  - Each patch token predicts the change of its 8×8 cell of the next 32×32 frame, and the drives token the
    next drives, both from [token output, executed move]. The move token's output is the policy.
- **Agent:** vm-00008's arm D: the basic agent (`configs/AHR_gated.json`: window of 8 real states, hindsight
  hunger command, 20% babbled moves, batch 8, memory U), policy head linear and detached
  (`ppo.pi_detach=1`), no inverse dynamics. 50k steps, seeds 1–2.
- **Arms:**

  | Arm | Model | Dots | Threads |
  |---|---|---|---|
  | P0 | patch tokens | none | 8 |
  | PD | patch tokens | 8 | 8 |
  | CD | CNN | 8 | 4 |

  Control without dots: vm-00008's D-U (exp_00008/001–002), the same agent with the CNN model.
- **Evaluation:**
  - own moves on the 128 test episodes every 10k steps (`online.py --eval`);
  - probes of the cursor, the circle and the direction from the move token's output (`analyze_self_actor.py
    probe`), and aim (`aim`);
  - coupling: when the cursor or a dot crosses from patch A to B, B's attention on A relative to the other
    patches (`analyze_patch.py coupling`);
  - action sensitivity: where the predicted next frame changes when the move is reversed
    (`analyze_patch.py sensitivity`).

## Log

### 2026-10-24
- **Implemented in a copy of `dev/`** while vm-00008 and the vm-00007 follow-up were running:
  - C++: `model.patch` (header slot 38): token building, block-causal attention (an end index per token;
    the old causal path is unchanged), the per-token heads, the backward pass, and `vm_attention` (the
    attention weights of one sequence). The Transformer layers moved into shared forward and backward
    functions. `vm_act` and `vm_encode` refuse patch mode: the basic agent acts through `vm_forward` on its
    window, so acting equals training by construction.
  - PyTorch reference: patch mode (`forward_patch`, `predict_patch`); the Transformer layers as one
    function.
  - World: `env.distractors` (and `dot_size`, `dot_brightness_*`, `dot_speed_*`); `info["gt_dots"]`.
  - `analyze_patch.py` (coupling, sensitivity); `analyze_self_actor.py probe` reads only z in patch mode.
- **Verification:**
  - 60 new tests (`tests/test_patch_tokens.py`): loss and every gradient against the reference (4 loss
    configurations, 4 model configurations, 4 and 16 threads), forward outputs, block causality, the
    attention weights (rows sum to 1, zero exactly outside the allowed tokens), the detached head,
    checkpoints, invalid configurations.
  - Negative controls: a reference with causal attention inside a step, or with a prediction that ignores
    the move, fails the gradient test (errors of 13–320%); the intact one matches to 2e-7.
  - 6 world tests (`tests/test_distractors.py`): rewards, hits, drives and positions are identical with and
    without dots for the same seeds and actions; dots change only background pixels, in grey within the
    brightness range; they bounce, keep their speed, are redrawn at resets and survive a state round trip.
  - With the defaults, 300-step runs give byte-identical checkpoints before and after the change: the
    basic agent with a detached hidden-layer head, the planner agent with latent dynamics, the basic agent
    with inverse dynamics.
  - All 436 Python tests and the C++ tests pass.
- **Smoke runs** (1,000 steps): patch mode runs at about 3.1 steps/s on 4 threads under load (the CNN
  basic agent: about 6.5). No coupling yet (ratio 1.00 at 300–1,000 steps).
- **Merged into `dev/`** at 13:26, after vm-00009 had launched; `make test_all` passes there.
- **Launched** (`make runs`, runs 001–006) at 13:49, when vm-00008 had finished. The configs are as
  intended (patch 16 or none, 8 dots or none, `pi_detach` 1, memory U; 8 threads for patch arms, 4 for CD).
- **Added two arms at 14:08** (`make hidden`, runs 007–010), after vm-00008's DMI became the first basic agent
  to aim (a detached head with a hidden layer; test success 0.63–0.82 at 50k, against 0.45 at birth). The
  linear head of P0, PD and CD barely aims in vm-00008, so:
  - PMD: patch tokens, dots, the hidden-layer head (8 threads);
  - CMDI: vm-00008's DMI with dots (CNN, the hidden-layer head, inverse dynamics; 4 threads). Do the dots
    disrupt the agent that works?
- **Diagnostics at 10k** (seeds 1, 2):
  - **Probes** (z: the move token's output, or the CNN's Transformer output): the patch model keeps the
    circle (P0 0.77, 0.68; PD 0.49, 0.56; CD 0.07, 0.19) but holds the cursor (P0 0.23, 0.21; PD 0.13, 0.16;
    CD 0.27, 0.22) and the direction (P0 0.18, 0.13; PD 0.08, 0.09; CD 0.26, 0.23) weakly. Without dots, the
    CNN (vm-00008 D-U) had cursor 0.36, 0.32 and direction 0.32, 0.26 at 10k. (At 30k, the birth values
    showed that the patch model's circle is not learned: see below.)
  - **Action sensitivity:** reversing the move concentrates the change of the patch model's prediction
    near the cursor: 23–26% (P0) and 17–20% (PD) of it within 6 px of the cursor, on 4% of the area. Near the
    dots it is proportional to their area (22% on 21%). The CNN with dots shows no concentration yet
    (4–5% on 4%), and its prediction changes half as much overall.
  - **Coupling: none yet.** When the cursor crosses from patch A to B, B attends to A slightly more than to
    other patches (median ratio 1.02–1.24 by layer), but an unrelated patch attends to A as much or more
    (1.02–1.32): every patch looks a little at the cursor's patch. B beats the control in only 22–45% of the
    crossings. For dots, both ratios are 1.01–1.03. (The control was added at this point; without it the
    first reading looked like coupling.)
- **Diagnostics at 30k** (15:27; seeds 1, 2). The probes at birth (checkpoint 0) change how the circle reads:

  | R² from z | cursor: birth → 10k → 30k | circle | direction |
  |---|---|---|---|
  | P0 (patch) | 0.08, 0.08 → 0.23, 0.21 → 0.30, 0.26 | 0.77, 0.74 → 0.77, 0.68 → 0.67, 0.72 | 0.10, 0.09 → 0.18, 0.13 → 0.21, 0.18 |
  | PD (patch, dots) | 0.05, 0.05 → 0.13, 0.16 → 0.19, 0.19 | 0.66, 0.66 → 0.49, 0.56 → 0.43, 0.56 | 0.07, 0.07 → 0.08, 0.09 → 0.11, 0.13 |
  | CD (CNN, dots) | 0.04, 0.03 → 0.27, 0.22 → 0.38, 0.38 | 0.94, 0.92 → 0.07, 0.19 → 0.15, 0.34 | 0.23, 0.21 → 0.26, 0.23 → 0.34, 0.35 |
  | D-U (CNN, vm-00008) | 0.06, 0.03 → 0.36, 0.32 → 0.44, 0.32 | 0.95, 0.94 → 0.08, 0.13 → 0.01, 0.80 | 0.24, 0.22 → 0.32, 0.26 → 0.35, 0.37 |

  - **Random features read the circle**, which is large: at birth, the CNN 0.92–0.95, the patch model's move
    token 0.62–0.77. The CNN loses it by 10k, as its prediction loss weights the changed pixels (mostly the
    cursor's) ten times, and gets it back in some runs. The patch model's move token stays near its birth
    level. So the patch model does not keep the circle by learning it; its random attention averages the
    patches.
  - **Both learn the cursor and the direction; the CNN more.**
  - **The dots cost the patch model**: cursor 0.19 against 0.26–0.30 and direction 0.11–0.13 against 0.18–0.21.
    The CNN shows no such cost.
  - **Coupling: still none.** B's attention on A, relative to other patches: 1.17–1.51 in layer 1, 1.04–1.31 in
    layer 2. An unrelated patch C on A: 1.22–1.47 and 1.02–1.36. B is above C in 31–59% of the crossings. For dots,
    all ratios are 0.97–1.04.
  - **Action sensitivity, relative to the background** (the change per unit of area outside the cursor and dot
    regions; 10k → 30k):

    | | near the cursor | near the dots | background change per pixel |
    |---|---|---|---|
    | P0 | 7.3, 8.7 → 12.8, 9.3× | – | 0.037, 0.030 → 0.025, 0.035 |
    | PD | 5.4, 6.9 → 9.3, 11.1× | 1.33, 1.44 → 1.34, 1.45× | 0.041, 0.032 → 0.033, 0.028 |
    | CD | 1.3, 0.9 → 17.3, 12.9× | 0.83, 0.79 → 1.07, 1.02× | 0.022, 0.024 → 0.016, 0.016 |

    The CNN started later but passed the patch model by 30k. Reversing the move changes its prediction near
    the cursor 13–17 times more than in the background, and near the dots no more than in the background. In
    the patch model the move changes the predicted dots a third more than the empty background, and the
    background itself twice as much as in the CNN. Every patch token's prediction takes the executed move as
    an input, so every cell can respond to it.
  - **Aim: none** (cos(mu, to circle)): P0 −0.03, −0.16; PD −0.09, −0.06; CD −0.06, +0.02 (vm-00008 D-U at 30k:
    −0.02, +0.07). As expected for a linear head; arms PMD and CMDI test the hidden layer.
- **Where the information sits** (16:00). With the detached head, no loss trains the move token's final output:
  the policy reads a by-product of layers trained for the other tokens. Added `analyze_patch.py tokens`: ridge
  probes from each kind of token output of a one-step window, with 16,384 probe states. The outputs come from
  the PyTorch reference; its move-token output matches the C++ z to 8e-6. Ranges over the six patch runs (P0,
  PD, PMD; seeds 1, 2), at 30k and (at birth):

  | Token outputs | cursor | circle | direction |
  |---|---|---|---|
  | move token (z, the policy's input) | 0.17–0.30 (0.07–0.11) | 0.46–0.72 (0.64–0.77) | 0.10–0.20 (0.08–0.10) |
  | drives token | 0.07–0.11 (0.02–0.04) | 0.12–0.28 (0.18–0.31) | 0.03–0.05 (0.03–0.04) |
  | command token | 0.13–0.18 (0.03–0.05) | 0.27–0.47 (0.25–0.41) | 0.06–0.08 (0.04–0.06) |
  | patch tokens, mean | 0.19–0.35 (0.08–0.11) | 0.44–0.60 (0.66–0.78) | 0.14–0.22 (0.08–0.11) |
  | patch tokens side by side (16 × 128) | 0.91–0.98 (0.81–0.93) | 0.97–0.98 (0.98–0.99) | 0.83–0.90 (0.72–0.81) |

  - The patch tokens hold nearly everything, and did so at birth: each keeps a 128-number linear projection
    of its pixels. Learning adds a little (cursor +0.06–0.12, direction +0.10–0.14). With dots, slightly
    less (cursor 0.91–0.94 against 0.98).
  - No single token gathers it. The move token reads about as much as the patches' mean. So the policy, linear
    or with a hidden layer, has almost no direction to act on.
- **Own moves on the test episodes, 0–30k** (16:14; birth 0.44–0.48):

  | Arm | seed 1: 0 / 10k / 20k / 30k | seed 2 |
  |---|---|---|
  | P0 | 0.45 / 0.36 / 0.47 / 0.49 | 0.45 / 0.43 / 0.25 / 0.39 |
  | PD | 0.45 / 0.43 / 0.41 / 0.37 | 0.45 / 0.45 / 0.41 / 0.41 |
  | CD | 0.48 / 0.37 / 0.41 / 0.46 | 0.44 / 0.27 / 0.34 / 0.32 |
  | PMD | 0.45 / 0.40 / 0.36 / 0.41 | 0.46 / 0.40 / 0.45 / 0.39 |
  | CMDI | 0.45 / 0.16 / 0.27 / 0.38 | 0.45 / 0.25 / 0.18 / 0.27 |
  | D-U (vm-00008, no dots) | 0.48 / 0.43 / 0.35 / 0.37 | 0.44 / 0.27 / 0.39 / 0.48 |
  | DMI-U (vm-00008, no dots) | 0.45 / 0.59 / 0.69 / 0.79 | 0.45 / 0.44 / 0.52 / 0.71 |

  - **No arm rises above birth.** The patch model does not aim, with the linear head or the hidden layer.
  - **The dots break DMI**: CMDI, the same agent and seeds with dots, falls to 0.16–0.25 at 10k and stays below
    birth, while DMI climbs to 0.71–0.79.
- **Why the dots break DMI** (16:25; probes from z, aim at 30k):

  | | cursor 10k / 30k | circle 10k / 30k | direction 10k / 30k | aim at 30k |
  |---|---|---|---|---|
  | DMI-U (no dots) | 0.86, 0.90 / 0.89, 0.89 | 0.31, 0.30 / 0.49, 0.50 | 0.68, 0.72 / 0.73, 0.76 | +0.20, +0.13 |
  | CMDI (dots) | 0.59, 0.66 / 0.75, 0.79 | 0.16, 0.18 / 0.07, 0.11 | 0.48, 0.55 / 0.60, 0.61 | +0.00, +0.02 |

  - With dots the CNN loses the circle (0.07–0.11 against 0.49–0.50) and holds the cursor and the direction less
    well. Its head does not aim. The prediction loss weights changed pixels ten times; the dots change pixels
    at every step, the circle almost never. A likely reading: the features spend their capacity on the dots and
    drop the circle.
  - The patch arm PMD at 30k is as P0 and PD: z cursor 0.17, 0.18, direction 0.11, 0.12, aim −0.04, −0.05; no
    coupling (B above C in 31–43% of the cursor crossings); sensitivity near the cursor 25–28% on 4% of the area,
    near the dots 19% on 21%.
  - CMDI's prediction separates self from other the most of all arms: 49–52% of the change near the cursor,
    11–12% near the dots on 21% of the area. Knowing which moving thing obeys the move is not enough to aim.
- **At 50k** (16:50; all runs finished by 16:46):
  - **CMDI recovers late, so the dots delay DMI rather than break it.** Test success at 40k and 50k: seed 1 0.68, 0.37;
    seed 2 0.87, 0.90 (DMI-U 0.81, 0.82 and 0.59, 0.81). Online hit rate in 40k–50k: 0.62, 0.76 (DMI-U 0.78, 0.74).
    Its z still holds no circle (0.06, 0.13) but the direction better (0.64, 0.67, against 0.48–0.55 at 10k). The
    absolute circle position is not needed once the direction is there. The 30k reading above ("the dots break
    DMI") holds only up to 30k: the circle stays lost, but the head does not need it.
  - **Coupling appears without dots.** In P0, B attends to A more than an unrelated patch does in 60% and 70%
    (seed 1, layers 1 and 2) and 55% and 81% (seed 2) of the 150 cursor crossings, up from 22–45% at 10k. With dots
    it does not (PD 35–55%, PMD 21–52%). The effect is small: medians of 1.06–1.49 against 0.99–1.43.
  - Everything else continues the 30k picture: no patch arm rises above birth, none aims (−0.14 to +0.01), and the
    move token's output still holds little (cursor 0.22–0.33, direction 0.13–0.23).

## Results

### R1. Own moves on the 128 test episodes (seeds 1, 2)

| Arm | Model | Dots | Head (detached) | birth | 10k | 20k | 30k | 40k | 50k | mean 10k–50k |
|---|---|---|---|---|---|---|---|---|---|---|
| P0 | patch tokens | – | linear | 0.45, 0.45 | 0.36, 0.43 | 0.47, 0.25 | 0.49, 0.39 | 0.48, 0.49 | 0.50, 0.45 | 0.46, 0.40 |
| PD | patch tokens | 8 | linear | 0.45, 0.45 | 0.43, 0.45 | 0.41, 0.41 | 0.37, 0.41 | 0.41, 0.48 | 0.44, 0.36 | 0.41, 0.42 |
| PMD | patch tokens | 8 | hidden layer | 0.45, 0.46 | 0.40, 0.40 | 0.36, 0.45 | 0.41, 0.39 | 0.48, 0.42 | 0.41, 0.43 | 0.41, 0.42 |
| CD | CNN | 8 | linear | 0.48, 0.44 | 0.37, 0.27 | 0.41, 0.34 | 0.46, 0.32 | 0.48, 0.47 | 0.44, 0.32 | 0.43, 0.34 |
| CMDI | CNN + inverse dynamics | 8 | hidden layer | 0.45, 0.45 | 0.16, 0.25 | 0.27, 0.18 | 0.38, 0.27 | 0.68, 0.87 | 0.37, 0.90 | 0.37, 0.49 |
| D-U (vm-00008) | CNN | – | linear | 0.48, 0.44 | 0.43, 0.27 | 0.35, 0.39 | 0.37, 0.48 | 0.33, 0.54 | 0.39, 0.55 | 0.37, 0.45 |
| DMI-U (vm-00008) | CNN + inverse dynamics | – | hidden layer | 0.45, 0.45 | 0.59, 0.44 | 0.69, 0.52 | 0.79, 0.71 | 0.81, 0.59 | 0.82, 0.81 | 0.74, 0.61 |

Online hit rate in 40k–50k (20% of the moves babbled; random babbling 0.47–0.48): P0 0.49, 0.47; PD 0.47, 0.48;
PMD 0.53, 0.47; CD 0.51, 0.44; CMDI 0.62, 0.76; D-U 0.54, 0.46; DMI-U 0.78, 0.74.

### R2. Features: linear R² from z (birth → 10k → 30k → 50k; 1,024 held-out random states)

| Arm | cursor | circle | direction |
|---|---|---|---|
| P0 | 0.08, 0.08 → 0.23, 0.21 → 0.30, 0.26 → 0.28, 0.33 | 0.77, 0.74 → 0.77, 0.68 → 0.67, 0.72 → 0.61, 0.68 | 0.10, 0.09 → 0.18, 0.13 → 0.21, 0.18 → 0.20, 0.24 |
| PD | 0.05, 0.05 → 0.13, 0.16 → 0.19, 0.19 → 0.21, 0.22 | 0.66, 0.66 → 0.49, 0.56 → 0.43, 0.56 → 0.46, 0.47 | 0.07, 0.07 → 0.08, 0.09 → 0.11, 0.13 → 0.13, 0.15 |
| PMD | 0.07, 0.05 → 0.12, 0.12 → 0.17, 0.18 → 0.24, 0.21 | 0.65, 0.62 → 0.44, 0.44 → 0.54, 0.52 → 0.56, 0.49 | 0.08, 0.08 → 0.06, 0.08 → 0.11, 0.12 → 0.19, 0.13 |
| CD | 0.04, 0.03 → 0.27, 0.22 → 0.38, 0.38 → 0.41, 0.48 | 0.94, 0.92 → 0.07, 0.19 → 0.15, 0.34 → 0.34, 0.51 | 0.23, 0.21 → 0.26, 0.23 → 0.34, 0.35 → 0.36, 0.45 |
| CMDI | 0.04, 0.03 → 0.59, 0.66 → 0.75, 0.79 → 0.79, 0.87 | 0.94, 0.92 → 0.16, 0.18 → 0.07, 0.11 → 0.06, 0.13 | 0.23, 0.21 → 0.48, 0.55 → 0.60, 0.61 → 0.64, 0.67 |
| D-U | 0.06, 0.03 → 0.36, 0.32 → 0.44, 0.32 → 0.50, 0.49 | 0.95, 0.94 → 0.08, 0.13 → 0.01, 0.80 → 0.48, 0.69 | 0.24, 0.22 → 0.32, 0.26 → 0.35, 0.37 → 0.41, 0.49 |
| DMI-U | 0.06, 0.03 → 0.86, 0.90 → 0.89, 0.89 → 0.89, 0.87 | 0.95, 0.94 → 0.31, 0.30 → 0.49, 0.50 → 0.49, 0.46 | 0.24, 0.22 → 0.68, 0.72 → 0.73, 0.76 → 0.72, 0.74 |

z is the move token's output in the patch arms and the Transformer output of the single token per step in the
CNN arms. Arms that differ only in the dots start from the same weights (P0 and PD, CD and D-U, CMDI and DMI-U);
their birth values differ because the dot arms' probe states contain dots.

### R3. Patch arms: where the information sits (`analyze_patch.py tokens`, 16,384 probe states; 50k, birth in brackets)

| Arm | move token (z): cursor / circle / direction | all patch tokens side by side |
|---|---|---|
| P0 | 0.29, 0.33 / 0.62, 0.69 / 0.20, 0.23 (0.10, 0.11 / 0.77, 0.76 / 0.10, 0.10) | 0.98, 0.98 / 0.99, 0.98 / 0.91, 0.91 (0.92, 0.93 / 0.99, 0.99 / 0.80, 0.81) |
| PD | 0.22, 0.24 / 0.48, 0.47 / 0.13, 0.16 (0.07, 0.08 / 0.68, 0.67 / 0.08, 0.08) | 0.94, 0.96 / 0.98, 0.98 / 0.87, 0.88 (0.81, 0.82 / 0.98, 0.98 / 0.72, 0.73) |
| PMD | 0.25, 0.24 / 0.55, 0.48 / 0.20, 0.14 (0.08, 0.07 / 0.64, 0.64 / 0.08, 0.09) | 0.96, 0.96 / 0.97, 0.97 / 0.87, 0.88 (0.83, 0.83 / 0.98, 0.98 / 0.73, 0.73) |

The drives and command tokens and the patches' mean hold no more than the move token (Log, 16:00).

### R4. Coupling: cursor crossings from patch A to B (150 per run; share in which B attends to A more than an unrelated patch C does; layer 1, layer 2)

| Arm | 10k | 30k | 50k |
|---|---|---|---|
| P0 | 0.29, 0.35; 0.22, 0.45 | 0.50, 0.59; 0.45, 0.55 | **0.60, 0.70; 0.55, 0.81** |
| PD | 0.32, 0.31; 0.27, 0.41 | 0.39, 0.31; 0.45, 0.51 | 0.48, 0.35; 0.55, 0.46 |
| PMD | – | 0.43, 0.37; 0.37, 0.31 | 0.52, 0.47; 0.41, 0.21 |

Seeds separated by a semicolon. With 150 crossings, a share outside 0.42–0.58 differs from 0.5 by more than two
standard errors. Dot crossings (611 per run) at 30k and 50k: 0.43–0.62, medians 0.95–1.06 for B and for C.

### R5. Action sensitivity: reversing the executed move (change per unit of area relative to the background; 10k / 30k / 50k)

| Arm | near the cursor | near the dots | background change per pixel |
|---|---|---|---|
| P0 | 7.3, 8.7 / 12.8, 9.3 / 14.4, 12.2× | – | 0.037, 0.030 / 0.025, 0.035 / 0.023, 0.028 |
| PD | 5.4, 6.9 / 9.3, 11.1 / 9.3, 12.0× | 1.33, 1.44 / 1.34, 1.45 / 1.24, 1.27× | 0.041, 0.032 / 0.033, 0.028 / 0.034, 0.027 |
| PMD | – / 9.9, 8.6 / 12.5, 14.3× | – / 1.30, 1.27 / 1.23, 1.30× | – / 0.033, 0.035 / 0.027, 0.023 |
| CD | 1.3, 0.9 / 17.3, 12.9 / 31.2, 22.6× | 0.83, 0.79 / 1.07, 1.02 / 1.07, 1.06× | 0.022, 0.024 / 0.016, 0.016 / 0.011, 0.016 |
| CMDI | – / 24.3, 27.2 / 34.1, 28.6× | – / 1.06, 1.16 / 1.09, 1.06× | – / 0.013, 0.013 / 0.010, 0.012 |

The region near the cursor is 4% of the frame, near the dots 21%.

### R6. Aim (cosine between the mean move and the direction to the circle, away from the walls; 30k / 50k)

P0 −0.03, −0.16 / +0.01, −0.12; PD −0.09, −0.06 / −0.12, −0.14; PMD −0.04, −0.05 / −0.07, −0.05; CD −0.06, +0.02 /
−0.11, +0.00; CMDI +0.00, +0.02 / −0.04, +0.12; D-U −0.02, +0.07 / −0.01, +0.12; DMI-U +0.20, +0.13 / +0.28, +0.13.

## Analysis

### The move token, not the patches, is the bottleneck (hypothesis 1, not supported)

- **The policy's input holds less than the CNN's.** At 50k the move token gives cursor 0.21–0.33 and direction
  0.13–0.24, against the CNN's 0.41–0.50 and 0.36–0.49 (CD, D-U) and 0.79–0.89 and 0.64–0.74 with inverse dynamics
  (CMDI, DMI-U).
- **The patch tokens hold nearly everything, and did so at birth**: side by side, cursor 0.94–0.98 and direction
  0.87–0.91 at 50k (birth 0.81–0.93 and 0.72–0.81). The shared linear embedding keeps a 128-number projection of
  each patch's pixels. Learning adds 0.05–0.15.
- **Nothing makes one token gather it.** With the detached head, no loss reaches the move token's output. Each
  patch token predicts only its own cell, and the drives token only the drives. The move token's output is a
  by-product of layers trained for the other tokens, and reads about as much as the patches' mean.
- **The circle "kept" by the patch model is not learned**: its move token's circle R² (0.46–0.68 at 50k) is
  at or below its birth level (0.62–0.77). Random features read the large circle. The CNN starts higher (0.92–
  0.95) and loses it within 10k, because its prediction does not need it (vm-00008).

### Coupling: weak, late, and only without dots (hypothesis 2, partly)

- **Without dots, the patch the cursor enters starts to attend to the patch it leaves**: in P0, more than an
  unrelated patch does in 55–81% of the crossings at 50k (22–45% at 10k). The effect is small.
- **With dots, not for the cursor** (21–55%; in one run's second layer B attends to A less than C does). For the
  dots' own crossings, at most a trace: 46–62% at 50k, four of eight values above 56% (two standard errors with
  611 crossings), with medians within 0.05 of the control's.
- **Every patch attends a little more to the cursor's patch** (medians up to 1.8 for the unrelated patch too). The
  cursor's patch is salient to all, which a test without the control would have read as coupling.
- The prediction does not need coupling. Each patch token gets the executed move directly, and the cursor's
  own patch can predict most of the change.

### Self versus other: the CNN separates them better (hypothesis 3, partly)

- **Both models learn where the move acts.** Reversing it changes the prediction near the cursor 9–14 times more
  than the background in the patch model and 23–34 times more in the CNN at 50k. The patch model started first
  (5–9× at 10k, when the CNN was at 1×), and the CNN passed it by 30k.
- **The CNN's dots do not respond to the move** (1.06–1.09× the background at 50k). **The patch model's do a
  little** (1.23–1.30×), and its background responds twice as much as the CNN's (0.023–0.034 against 0.010–0.016
  per pixel). Every patch token's prediction takes the move as an input, so every cell can respond to it; the
  CNN injects it once, at the bottleneck.

### No patch agent aims (hypothesis 4)

- **P0, PD and PMD stay at their birth level** (means 0.40–0.46 against 0.45; online 0.47–0.53, as random
  babbling) and none aims (−0.14 to +0.01). The hidden layer, which made the CNN agent aim in vm-00008, does not
  help: its input has no direction to work with.
- **The CNN with dots and a linear head (CD) is as vm-00008's D-U** without dots: it does not aim either.

### The dots delay the agent that works

- **DMI with dots (CMDI) starts below birth** (0.16–0.25 at 10k, against DMI-U's 0.44–0.59) and catches up by 40k–50k:
  0.68 and 0.87 at 40k, online 0.62 and 0.76 in 40k–50k (DMI-U 0.78, 0.74). Seed 1's test success then falls to
  0.37 at 50k, while its online rate keeps rising (0.15 → 0.62 over its life). Frozen tests of these agents
  swing; vm-00008's DMI-U seed 2 also went 0.71 → 0.59 → 0.81.
- **The direction tracks the delay**: CMDI's z direction is 0.48–0.55 at 10k and 0.64–0.67 at 50k, against DMI-U's
  0.68–0.76 from 10k on. Its circle stays lost (0.06–0.13), even in the run at 0.90. The absolute circle position
  is not what the head needs.
- **A likely reading:** the prediction loss weights changed pixels ten times, and the dots change pixels at every
  step. The features spend capacity on the dots, so the cursor and the direction form more slowly.

### Caveats

- 2 seeds per arm; frozen test success swings by up to 0.3 between checkpoints (CMDI seed 1: 0.68 at 40k, 0.37 at
  50k).
- One patch size (16), one set of dots (8, 2×2 px, grey). Dots that look like the cursor would test self versus
  other harder.
- The patch arms run at 8 threads, the CNN arms at 4. Both learn on every step, so their schedules match.
- The coupling and sensitivity tests use worlds driven by random moves, not the agents' own.

## Conclusions

1. **Patch tokens hold the whole scene, but the policy reads one token that nothing trains.** Each patch token
   predicting its own cell does not make any token gather the cursor and the direction. The move token's output
   holds less than the CNN's vector, and no patch agent aims, with a linear head or a hidden layer.
2. **Patches couple only weakly, late, and only without dots**: at 50k, the patch the cursor enters attends to
   the one it leaves more than an unrelated patch does in 55–81% of the crossings. Every patch attends a little to
   the cursor's patch.
3. **Both models learn where their own move acts; the CNN separates self from other more cleanly.** Its predicted
   dots do not respond to the move, while the patch model's respond a little.
4. **Moving dots delay the CNN agent that aims** (DMI) by 20k–30k steps, without stopping it. Its features drop
   the circle, and the direction grows more slowly.

## Next Steps

- **A summary token** (planned in the task under Later), trained to predict the whole next state. The policy
  reads it. That gives the patch model one token that has to gather the scene.
- **Or the move token predicts**: give the move token's output a prediction target of its own (the next drives,
  or the next cursor cell) without letting the move loss in.
- **Less weight on changed pixels** (`ppo.pred_changed_weight` below 10), so that the dots do not dominate the
  frame loss. Does CMDI then keep pace with DMI?
- **Dots that look like the cursor**, for a harder test of self versus other.
