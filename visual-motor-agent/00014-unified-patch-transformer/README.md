# vm-00014: One Transformer for Everything: Fine Patch Tokens, Nothing Detached, in the Hazard World

**Date:** 2026-10-25
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00014`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00014/`
**Task plan:** [`../tasks/00018-unified-patch-transformer.md`](../tasks/00018-unified-patch-transformer.md)

## Goal

Test the design of `../../visual_motor_learning_transformer.md`: one GPT-style Transformer on continuous tokens
(8×8 pixel patches plus a token each for the drives, the previous move and the command), no CNN, every part of the
next state predicted by the same network. vm-00013 found that the planner cannot learn hazard contact because its
latent state is trained by frame prediction only. In one shared Transformer nothing is detached: the drives token's
hunger loss reaches every patch token through attention.

Changes from the document: RGB, not grayscale (the red target and the blue hazard have the same standard grayscale,
93.8 and 94.0); one embedding per kind of token, not one shared projection over zero-padded vectors; no reward
anywhere; acting by the basic agent's hindsight command.

## Hypothesis

1. In a shared Transformer, hazard contact gets learned: the drives head predicts the extra hunger inside a hazard
   (the world: +0.105 against +0.005), which vm-00013's models did not.
2. When the move loss trains the Transformer, the move token learns to aim (detached it did not, vm-00010), without
   eroding what the patch tokens hold.
3. 8×8 patches hold the cursor and contact better than 16×16 patches or the whole frame as one token.
4. The basic agent still does not avoid hazards: its hindsight command marks only relief.

## Method

- **Model:** vm-00010's patch tokens (`model.patch`): each step is one token per patch (a shared linear embedding of
  its raw RGB pixels, plus a learned patch-position embedding), plus a drives, a previous-move and a command token, each
  with its own embedding; a type and a step embedding; block-causal attention over 8 steps. Each patch token predicts
  the change of its cell of the next 32×32 frame, the drives token the next drives, the move token the next move.
  - P8: 64 patches of 8×8, 67 tokens per step, 536 per window; P16: 16 patches, 19 per step; P64: the whole frame as one
    token (12,288 values), 4 per step.
- **GPU attention for long windows** (task 00018): the thread-per-row attention kernel kept 256 scores per thread and
  did about 10⁹ atomic adds per P8 update. Blocks longer than 32 tokens now use batched cuBLAS products (scores
  Q·Kᵀ and P·V per head, batched over the windows) with a masked softmax kernel; the backward likewise. The P8
  update takes 6.7 ms instead of 38, the forward 0.5 ms instead of 3.9; a lifetime runs at 105 steps/s alone and 35 with
  12 sharing the GPU. Verified against the C++ model (P8 and P64 added to `tests/test_cuda_model.py`: 219 GPU tests);
  a wrong key gradient in the new path fails exactly the three long-window models. The C++ model was checked against
  the PyTorch reference for both new patch sizes (`tests/test_patch_tokens.py`).
- **World:** vm-00013's continual torus with 2 hazards at 0.1 hunger per step.
- **Agent:** the basic agent (`configs/AHR_gated.json`, memory U, policy hidden layer 128, hindsight command), 60k
  steps on the GPU, seeds 1–3. Arms: P8D (move loss detached), P8A, P16A, P64A (move loss trains the Transformer).
  Controls: vm-00012's DTH10 and vm-00013's W10D (DMI, the CNN model).
- **Measures:** online hits per 1,000 steps and the share of steps in a hazard; `analyze_hazards.py contact` (for patch
  models: probes of "inside a hazard" from the move token, the drives token, the token of the patch under the cursor
  and the mean of the patch tokens; the drives head inside against outside) at 0, 20k, 40k, 60k; the episodic and
  forage tests; `analyze_patch.py tokens` and `analyze_self_actor.py aim` at 60k.

## Log

### 2026-10-25
- **Implemented** (task 00018): GPU attention without the 256-key limit and with batched GEMMs for long blocks;
  contact probes for patch models; tests for 8×8 and whole-frame patches. 709 Python tests and the C++ tests pass.
- **At birth (P8, step 0):** a probe already reads contact from the token of the patch under the cursor (R² 0.57: its
  raw blue pixels are in it), but the drives head predicts the same hunger inside and outside (0.390, 0.390).
- **Launched** the 12 lifetimes at 09:31.
- **Contact at 20k** (`analyze_hazards.py contact`): no arm's drives head predicts more hunger inside a hazard
  (inside − outside: P8D +0.013, −0.001, 0.000; P8A −0.004, −0.006, −0.006; P16A −0.009, −0.024, −0.025; P64A +0.001,
  +0.022, +0.007; the world +0.100). The cursor's patch token holds contact (R² P8D 0.62–0.64, P8A 0.41–0.48, P16A
  0.33–0.35, P64A 0.00–0.03), the move and drives tokens hardly (≤ 0.17).
- **A confound in every hazard experiment so far: hunger is capped at 1** (`circle_world.py`: clip to [0, 1]). At the
  cap a hazard step changes nothing, so the drives carry no trace of it. This agent hits 2–15 times per 1,000 steps;
  its hunger is at 1 on about half of the steps. At birth (P64 frozen, forage test): 5.9% of steps in a hazard, only
  1.7% where it raised hunger, hunger at the cap on 55% of steps. New measures for this (scoring only): online
  `hazard_felt` and `hunger_capped` in `metrics.jsonl` (from now on; the 12 running lifetimes started before), and
  `forage_hazard_felt`, `forage_capped` in the forage test.
- **Contact at 40k (all arms) and 60k (P16A, P64A)**, the drives head inside − outside (the world +0.100): P8D +0.004,
  −0.002, +0.004; P8A −0.005, +0.001, −0.009; P16A +0.014, −0.005, −0.038 (60k: +0.018, −0.015, −0.030); P64A +0.011,
  +0.005, +0.005 (60k: −0.002, +0.009, −0.001). R² of contact from the cursor's patch token: P8D 0.67–0.69 (birth 0.57),
  P8A 0.37–0.44, P16A 0.31–0.36, P64A 0.00–0.03; from the drives token at most 0.29 (P8D), else ≤ 0.06.
- **Online, last 10k steps** (P8 arms at 52k): hits per 1,000 steps P8D 5.0, 6.0, 6.5; P8A 4.0, 6.0, 7.5; P16A 4.7,
  3.5, 5.9; P64A 6.6, 4.9, 8.4 (random agents on the torus 4.4–5.0, vm-00011; DMI 7.6–10). In a hazard 0.053–0.110
  (0–5k: 0.049–0.092).
- **All 12 finished at about 10:00.** The aim analysis never ended: it ran the test episodes for the world's
  `max_steps`, 10⁹ in a continual world. Fixed (`analyze_self_actor.py` now uses the episodic test episodes, as
  `online.py --eval`) and rerun on the GPU (`--device cuda`, 11 s per run instead of 330).

## Results

### R1. Online and the frozen tests (seeds 1, 2, 3)

| Arm | hits per 1,000 steps, last 10k | in a hazard, last 10k | episodic test 0 → 20k → 60k | forage test at 60k (hits per 1k) |
|---|---|---|---|---|
| P8D (8×8, detached) | 6.1, 5.6, 6.9 | 0.065, 0.072, 0.082 | 0.60 → 0.77 → 0.74; 0.62 → 0.65 → 0.73; 0.59 → 0.74 → 0.79 | 7.1, 6.3, 7.7 |
| P8A (8×8, through) | 4.5, 4.9, 7.7 | 0.065, 0.050, 0.103 | 0.60 → 0.60 → 0.73; 0.62 → 0.71 → 0.61; 0.59 → 0.91 → 0.84 | 5.5, 5.9, 8.9 |
| P16A (16×16, through) | 4.7, 3.5, 5.9 | 0.110, 0.057, 0.081 | 0.59 → 0.55 → 0.60; 0.61 → 0.56 → 0.65; 0.60 → 0.81 → 0.80 | 4.5, 3.5, 8.7 |
| P64A (whole frame, through) | 6.6, 4.9, 8.4 | 0.080, 0.053, 0.088 | 0.59 → 0.91 → 0.78; 0.62 → 0.64 → 0.61; 0.61 → 0.91 → 0.93 | 6.4, 5.7, 10.4 |

Random agents on the torus: 4.4–5.0 hits per 1,000 steps (vm-00011); chance share in a hazard about 0.065; DMI (CNN) in
the same world (vm-00012 DTH10): 6.7–8.2.

### R2. Hazard contact (`analyze_hazards.py contact`, 60k)

| Arm | drives head, inside − outside (world +0.100) | R² of contact: cursor's patch token | drives token | move token (z) |
|---|---|---|---|---|
| P8D | +0.011, +0.011, +0.025 | 0.72, 0.70, 0.69 | 0.11, 0.45, 0.35 | 0.16, 0.46, 0.32 |
| P8A | −0.007, +0.006, −0.010 | 0.32, 0.42, 0.42 | −0.01 to 0.06 | 0.00 to 0.05 |
| P16A | +0.018, −0.015, −0.030 | 0.33, 0.31, 0.36 | 0.01–0.02 | −0.01 to 0.01 |
| P64A | −0.002, +0.010, −0.001 | 0.03, 0.01, 0.02 | ≤ 0 | ≤ 0 |

### R3. What the tokens hold at 60k (`analyze_patch.py tokens`: R² of cursor / circle / direction to the circle)

| Arm | all patch tokens side by side | move token (the policy's input) |
|---|---|---|
| P8D | 0.87–0.93 / 0.97 / 0.75–0.79 | 0.04–0.11 / 0.65–0.74 / 0.11–0.19 |
| P8A | **−0.34 to −0.06** / 0.94–0.96 / −0.22 to 0.00 | 0.00–0.01 / 0.15–0.39 / 0.03–0.08 |
| P16A | 0.21–0.25 / 0.97–0.98 / 0.28–0.31 | 0.00 / 0.08–0.12 / 0.01–0.02 |
| P64A | 0.02–0.03 / 0.94–0.95 / 0.19–0.20 | 0.01 / 0.81–0.82 / 0.16–0.17 |

vm-00010 (16×16, detached, episodic world): all patch tokens cursor 0.94–0.98.

### R4. Aim at 60k (`analyze_self_actor.py aim`: cos(μ, direction to the circle), away from the walls)

P8D +0.142, +0.032, +0.157; P8A +0.041, +0.003, +0.030; P16A +0.014, +0.011, +0.025; P64A +0.057, +0.000, +0.065.

## Analysis

### No contact learned (hypothesis 1, not supported, and confounded)

No arm's drives head predicts more hunger inside a hazard (at most +0.025 of the world's +0.100, mostly about 0). In
P8D the information is there: the token under the cursor reads contact at R² 0.69–0.72 (0.57 at birth, from its raw blue
pixels), and in two seeds the drives and move tokens pick it up too (0.35–0.46). But at these rates hunger sat at its
cap of 1 on about half of the steps, where a hazard step leaves no trace in the drives (found during this
experiment). vm-00015 repeats the question without the cap.

### The move loss through the Transformer erases the cursor (hypothesis 2: the opposite)

The clearest case yet of the effect vm-00007/8 found in the CNN. Letting the move loss train the Transformer (P8A)
does not make the move token aim: it holds less (circle 0.15–0.39, against 0.65–0.74 detached), and it erases the
cursor from the patch tokens themselves (R² −0.34 to −0.06, against 0.87–0.93 detached). Imitating its own moves
pushes the representation toward what predicts the agent's next move, and the cursor's position does not. The aim
falls with it (+0.003 to +0.041, against up to +0.157 detached). Detaching remains necessary.

### Patch size (hypothesis 3, not supported)

All three sizes were run with the move loss attached, so they are compared in its presence. Contact under the cursor:
8×8 0.32–0.42, 16×16 0.31–0.36, the whole frame 0.01–0.03. The cursor from all patch tokens: 16×16 keeps 0.21–0.25
while 8×8 loses it entirely (below 0), the whole frame 0.02–0.03. So no size is better on every measure, and the move
loss dominates the comparison. The whole frame as one token never locates the cursor but finds the large red circle
(0.94). Its episodic test in seed 3 (0.93 at 60k) comes without aim (+0.065) or a cursor: the hindsight command's
click timing and drift, not steering. Detached, 8×8 patches hold the cursor as well as vm-00010's detached 16×16 did
(0.87–0.93 against 0.94–0.98, different worlds).

### The design's promise and its limit

The document's unified Transformer keeps the scene in its patch tokens (P8D) as vm-00010 did. Nothing in it makes the
token the policy reads gather the scene: detached, the move token is trained by nothing; attached, its loss destroys
the scene. That is the gap the act token (vm-00016) tries to close.

### Caveats

- The hunger cap (above); hazard results here are superseded by vm-00015.
- The frozen tests vary widely between seeds (0.61–0.93 at 60k in P64A); three seeds per arm.
- P8A, P16A and P64A share the attached move loss; P8D is the only detached arm, so patch size is compared attached.

## Conclusions

1. **A shared Transformer on 8×8 RGB patches keeps the scene, including contact, in its patch tokens**, but did not
   learn that contact raises hunger (confounded by the hunger cap; see vm-00015).
2. **Letting the move loss train the Transformer erases the cursor from the patch tokens** (R² 0.87–0.93 → below 0)
   and the agent stops aiming. The move loss must stay detached.
3. **Patch size matters less than the move loss.** With it attached, 8×8 and 16×16 patches hold contact about
   equally and 16×16 keeps more of the cursor; the whole frame as one token never locates the cursor.
4. **No arm forages much above chance** (3.5–8.4 hits per 1,000 steps against 4.4–5.0) and none avoids hazards.

## Next Steps

- Slow hunger, so that hazard contact shows in the drives (vm-00015).
- A token the policy reads that a prediction loss trains (the act token, vm-00016).

