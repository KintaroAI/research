# Task 00018: One Transformer for Everything: Fine Patch Tokens, Nothing Detached, in the Hazard World (experiment vm-00014)

**Date:** 2026-10-25
**Status:** In Progress

## Context

- `../../visual_motor_learning_transformer.md` proposes one GPT-style Transformer on continuous tokens: 8×8 pixel
  patches plus a token each for the move, the drives and the reward, learned position and role embeddings, no CNN,
  and every output predicted by the same network.
- vm-00010's patch-token model is most of that already: each step is one token per 16×16×3 patch plus a drives, a
  previous-move and a command token; each kind has its own linear embedding (an LLM's embedding table generalized to
  continuous inputs), plus a type, a patch-position and a step embedding; each token predicts its own part of the
  next state. vm-00010 found that the patch tokens hold the scene (cursor R² 0.94–0.98 from all of them), but the
  move token, which no loss trains when the move loss is detached, does not aim.
- vm-00012/13: neither the planner nor the basic agent learns hazard contact. The planner's hunger model reads the
  detached CNN features v, which only frame prediction trains, so contact never enters its latent state. In a single
  shared Transformer nothing is detached: the drives token's hunger loss reaches every patch token through attention.
- Changes from the document, from what we measured:
  - **RGB, not grayscale:** the standard grayscale of the red target (220, 40, 40) is 93.8, of the blue hazard
    (50, 90, 230) 94.0: food and hazard would look the same.
  - **One embedding per kind of token, not one shared projection over zero-padded vectors** (as vm-00010): a shared
    projection puts hunger on the same column as pixel 0 of every patch.
  - **No reward** in the inputs or the losses (the project's question); hunger is the internal signal. Acting is the
    basic agent's: the hindsight command, no planner.

## Goal

Test the document's design in the continual torus world with hazards: 8×8 patches (64 patch tokens per step, 67
tokens per step, 536 per 8-step window), with the move loss detached or training the whole Transformer, and the whole
frame as one token as the document's control.

## Questions

1. **Does a shared Transformer learn hazard contact** where vm-00013's models did not? The drives head's next hunger
   inside a hazard against outside (the world: +0.105 against +0.005), and probes of "inside a hazard" from the token
   outputs.
2. **Does the move token aim when the move loss trains the Transformer** (vm-00010: detached, it did not)? And does
   that erode what the patch tokens hold, as it eroded the CNN features (vm-00007/8)?
3. **Do finer patches help?** 8×8 against vm-00010's 16×16 (the cursor sprite spans several 16×16 patches' worth of
   pixels in one patch).
4. **The whole frame as one token** (`model.patch=64`): one patch token, one embedding of all 12,288 values.
   Patch sizes are compared with the move loss training the Transformer (P8A, P16A, P64A), in the same world.

## Implementation

- **GPU attention without the 256-key limit.** `k_attn_bwd` keeps the scores of one row in a 256-float local array;
  recompute them in its second loop instead, and drop the `ctx * ntok <= 256` check (`make_dims`). The CPU model has no
  such limit.
- **Throughput:** measure the P8 update and forward on the GPU (536 tokens per window, about 2.2M attention scores per
  update); optimize the attention kernels only if a 60k lifetime takes more than about 2 hours.
- **`analyze_hazards.py contact` for patch models** (it calls `encode`, which patch models refuse): probes from the
  move token (z), the drives token, the mean and the concatenation of the patch tokens (token outputs from the PyTorch
  reference, as `analyze_patch.py tokens`), and the drives head inside against outside.
- **Tests:** CUDA against C++ for 8×8 patches with more than 256 tokens per window, and for one whole-frame patch
  (`tests/test_cuda_model.py`); C++ against the PyTorch reference for one whole-frame patch (`tests/test_patch_tokens.py`).

## Experiment vm-00014

World: vm-00013's continual torus with 2 hazards at 0.1 hunger per step (`env.respawn`, `env.wrap`, `env.hazards=2`,
`env.hazard_cost=0.1`). Agent: the basic agent (`configs/AHR_gated.json`, memory U, `model.pi_hidden=128`, hindsight
command), `model.patch` set, `online.device=cuda`, 60k steps, seeds 1–3.

| Arm | Tokens | Move loss |
|---|---|---|
| P8D | 64 patches of 8×8 + 3 | detached (`ppo.pi_detach=1`) |
| P8A | 64 patches of 8×8 + 3 | trains the Transformer (`ppo.pi_detach=0`) |
| P16A | 16 patches of 16×16 + 3 (vm-00010's tokens) | trains the Transformer |
| P64A | the whole frame as one token + 3 | trains the Transformer |

Controls: vm-00012's DTH10 (DMI, the CNN model, same world), vm-00010 (16×16 patches, episodic world).

## Measures

- Online: hits per 1,000 steps, the share of steps in a hazard (chance about 0.065).
- `analyze_hazards.py contact` at 20k, 40k and 60k.
- `analyze_patch.py tokens` (cursor, circle, direction from each kind of token) and `analyze_self_actor.py aim` at 60k.
- The episodic test (`online.py --eval --eval-device cuda`) every 20k.

## Later

- The signed hindsight command (the achieved hunger change, so moves into hazards are marked).
- A global summary token that the prediction losses train, read by the policy.
