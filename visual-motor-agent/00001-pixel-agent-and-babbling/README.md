# vm-00001: Pixel Agent and Reward-Free Babbling

**Date:** 2026-10-23
**Status:** Complete
**Source:** `exp/vm-00001` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00001/` (runs `NNN_<name>_s<seed>`)

## Goal

1. Build plan phases 0, 1 and 3:
   - the world, with a pixel-identical browser twin;
   - a C++ agent: CNN, causal Transformer, Gaussian+Bernoulli policy, value, and action-conditioned
     next-frame prediction;
   - PPO.
2. Answer the owner's question directly: **can the model learn meaningful dynamics of its own
   actions without being reinforced for it?** Train only a next-frame / next-drive predictor while
   the policy stays at its random initialization (motor babbling). Then test its predictions
   counterfactually.
3. Measure sparse-reward PPO (plan run A) and the prediction-auxiliary arm (run C) on the same
   architecture.

## Hypothesis

- **H1 (reward-free dynamics).** With no reward, prediction alone teaches:
  - movement semantics: dx/dy shift the cursor;
  - the click contingency: a click with the cursor over the circle removes the circle, and nowhere
    else;
  - the interoceptive consequence: hunger drops after that click.
- **H2 (auxiliary prediction).** Adding the prediction loss to PPO speeds up acquisition relative
  to reward-only PPO.

## Method

### World (`dev/world/`)

- Viewport: 64×64.
- Background: dark (20,20,30).
- Target: one red (220,40,40) circle, radius U[12,16]. "Small" runs use U[4,8].
- Cursor: a 7-px white "+" crosshair with a black outline, drawn by the world. It starts outside
  the circle.
- Actions: dx, dy clipped to [−1,1]×4 px; click if the third value > 0.5.
- Success = click with the cursor inside the circle. The circle then vanishes in the terminal frame
  and the episode ends.
- Episode limit: 200 steps.
- Reward: +10 success − 0.001/step.
  - Optional bounded movement/empty-click bonuses (≤ 0.374 per episode, analytic bound).
  - Optional drives: hunger +0.005/step, reset by success; restlessness grows when still. Exposed as
    two inputs; optional hunger cost.
- The HTML/JS twin (`world/browser_demo/`) is pixel-identical to the simulator on 312 fixtures
  (3,390 images, 0 mismatched pixels) and step-identical on 1,540 steps.

### Agent (`dev/src/vm_agent.cpp`)

The agent is about 1.28–1.30M parameters of C++17/OpenMP code on CPU (this VM has no GPU).

- **Encoder:** conv k4s2 (3→16→32→32) → FC 128.
- **Token:** [visual 128, drives 2, previous executed action 3] → 128-d embedding + learned
  position.
- **Transformer:** 2 pre-LN blocks, 4 heads, FF 512, context 32.
- **Policy:** tanh-mean Gaussian (dx, dy) with state-independent log σ (init −0.5), plus a
  Bernoulli click.
- **Value:** linear head.
- **Prediction head:** conditioned on (z_t, a_t) → next 32×32 frame + next drives. Options were
  added during this experiment (see Log):
  - `coord`: CoordConv input + spatial-broadcast decoder;
  - `pred_residual`: predict the change;
  - `pred_skip`: U-Net skips of the encoder's 8×8 and 16×16 maps of the *current* frame, with the
    action tiled in, so action and local features meet before a ReLU;
  - `pred_changed_weight`: weight on pixels that differ between consecutive frames.
- **Inverse-dynamics head (optional, `inv_hidden`):** MLP([v_t, v_{t+1}]) → own executed (dx, dy).
  It reads the visual features of two consecutive frames of one episode, taken before the token
  adds the previous action, so the answer cannot leak. Its target is the agent's own action, not
  world state.

Correctness checks:
- Every gradient was checked against a float64 PyTorch reference: 60 tests over 6 architecture
  variants, including all the options above; relative error about 1e-7 per tensor.
- KV-cached acting equals batched training bitwise, so the PPO ratio is exactly 1 before the first
  update.
- Causality tests pass.

### Reward-free babbling protocol

- `ppo.use_ppo = 0`: no policy, value or entropy losses, so the policy stays at init (σ ≈ 0.61,
  click p ≈ 0.5).
- Only `pred_coef · L_frame + drive_coef · L_drives` trains.
- Drives are on (observed), so hunger relief is learnable.
- Bonuses are irrelevant here (no RL).

### Counterfactual evaluation (`dev/analyze_dynamics.py`)

On held-out states, the world is restored with `set_state` and steps each candidate action. The
model predicts each outcome from the same single-frame context. Ground truth is used only for
scoring.

| Metric | Meaning | Chance / truth |
|---|---|---|
| `outcome_explained_acc` | Is the prediction conditioned on the true move the best explanation of the actual next frame, among 9 moves? | 1/9 |
| `action_retrieval_acc` | Is each prediction closest to its own action's true outcome? (strict) | 1/9 |
| `mean_pred_disp_per_move` | Cursor displacement read off the predicted frames, per commanded move | truth ±4 px |
| `vanish_inside` / `vanish_outside` | Fraction of red removed in the predicted frame by a click, cursor inside vs outside | 1 / 0 |
| `hunger_drop_inside` | Predicted hunger without minus with an inside click | ≈ 0.25 |
| `probe_*_r2_*` | Ridge R² of cursor/target position from z, conv-3 map (a3) or visual feature (vf) | random-init control |

## Log

### 2026-10-23: build and verification

- World (69 tests), JS twin (pixel-exact), C++ agent and PyTorch-reference tests all pass.
- Throughput on 64 threads: act 3.7 ms per 128 envs; world step 1.4 ms; PPO update about 5.5 s per
  4 epochs × 16k transitions; about 2k env-steps/s per run.

### Random-policy baselines (2,000 episodes, `random_baseline.py`)

| Policy | r 12–16 success | median steps | r 4–8 success | median steps |
|---|---:|---:|---:|---:|
| U[−1,1], click p=0.5 | 62.8% | 55 | 32.1% | 67 |
| N(0,1) clipped, p=0.5 (untrained Gaussian σ=1) | 73.5% | 52 | 38.9% | 76 |
| N(0,0.5²), p=0.5 (≈ our init σ=0.61) | 51.9% | 50 | 26.2% | 78 |
| no movement, p=0.5 | 0% | – | 0% | – |

### Sparse-reward PPO (run A), radius 12–16

`002_A_baseline_s1` (plain architecture; stopped after convergence at 0.56M steps):

| env steps | eval success (stoch.) | eval success (det.) | median steps | clicks / success |
|---:|---:|---:|---:|---:|
| 16k | 0.645 | 0.172 | 52 | 96 |
| 180k | 0.957 | 0.641 | 29 | 27.5 |
| 344k | **1.000** | **1.000** | 7 | 5.3 |
| 508k | 1.000 | 1.000 | 5 | 4.6 |

The trained agent moves almost straight to the circle and clicks: recorded episodes take 2–12
steps, with path length close to the start distance. With only a sparse reward, PPO solves the
large-circle task in about 0.35M steps.

### Reward-free babbling: architecture iterations

| Run | Decoder | Result at ≥ 0.4M steps |
|---|---|---|
| 001 `P_babble` | MLP → deconv, full-frame sigmoid, plain MSE | Reconstructs the circle (blurry) and **drops the cursor entirely**. Retrieval at chance at 836k steps; cursor probe R² no better than random features. |
| 003 `P_babble_w10` | + changed-pixel weight ×11 | Same: no cursor at 426k |
| 004/005 | + more updates, CoordConv + spatial broadcast | Same: no cursor at 262k |
| 006 `P_babble_res` | + residual (predict the change) | Collapses to "predict no change" |
| 007 `P_babble_skip` | + U-Net skip (conv-2 map) and tiled action at the **last, linear** deconv | Reduces changed-pixel error but cannot use the action: a linear layer adds an action-dependent constant, it cannot shift a sprite |
| 009 `P_babble_skip2` | skips at both deconvs; action + conv-3 map before a ReLU | Directional hints at 524k, not stable by 1M. Context 32 means only ~32 updates per 16k steps. |
| 011 `P_babble_skip_ctx8` | as 009, context 8, 128 updates per 16k steps, lr 1e-3 | **Learns everything by 2M** (slowly; table below) |
| 012 `P_babble_inv` | as 011 **+ inverse dynamics** (predict own dx, dy from consecutive visual features) | **Learns everything by 1M**; movement accurate by 0.5M |

Small-object dynamics are the crux. The cursor is about 1% of the pixels, and copying the last
frame already gets about 98–99% of pixels right. A decoder that renders from a global latent
learns the big circle and treats the cursor as noise; this is the CoordConv failure, and SimPLe and
UNREAL report the same issue. Three things together made it learnable:
- predicting the change;
- local (skip) access to the current frame's features;
- a nonlinearity between the action and those features.

Run 009 at 524k steps (about 1,000 updates), mean predicted cursor displacement (px) per commanded
move (truth: ±4 per axis):

| move | right | left | down | up | down-right | up-right | down-left | up-left |
|---|---|---|---|---|---|---|---|---|
| predicted (dx, dy) | (+0.22, +0.11) | (+0.05, −0.01) | (+0.12, +0.35) | (+0.09, −0.14) | (+0.14, +0.33) | (+0.23, −0.05) | (+0.08, +0.32) | (+0.09, −0.35) |

All four vertical moves and all four diagonals have the correct y sign, and right-ward moves are
predicted further right than left-ward ones. The magnitude is still about 10% of the truth. This
version did not stabilize by 1M (see Results). The update count was the limit, which runs 011/012
fix.

## Results

### R1. Reward-free babbling learns what the agent's actions do (H1: supported)

The policy is frozen at its random initialization and there is no reward of any kind. Only
next-frame and next-drive prediction train, plus, in 012, inverse dynamics. All runs use radius
12–16 and observed drives. Metrics come from 256 held-out counterfactual states (`analyze_dynamics.py`).

| Run | steps | action retrieval (chance .111) | outcome explained (chance .111) | corr(pred, true disp) x / y | vanish: inside / outside click (truth 1 / 0) | predicted hunger relief (truth ≈ .27) | cursor-x probe R² (visual feature) |
|---|---:|---:|---:|---:|---:|---:|---:|
| untrained (skip arch) | 0 | .118 | .111 | .11 / .13 | .15 / .14 | .009 | .38 |
| 001 plain decoder | 836k | .116 | .113 | n/a (cursor never drawn) | −.01 / −.00 | .002 | .34 |
| 009 skip, ctx 32 | 1M | .121 | .130 | .08 / −.04 | .02 / .02 | .003 | .20 |
| 011 skip, ctx 8 | 524k | .125 | .147 | −.04 / .00 | .12 / .02 | .010 | .32 |
| 011 skip, ctx 8 | 1M | .269 | .304 | .88 / .29 | **.97 / .04** | **.351** | .05 |
| 011 skip, ctx 8 | 2M | **.778** | **.722** | **.91 / .94** | **.98 / .03** | **.313** | .41 |
| 012 skip, ctx 8 + inverse | 131k | .131 | .152 | .22 / .26 | .02 / .02 | .004 | .97 |
| 012 skip, ctx 8 + inverse | 262k | .549 | .435 | .82 / .90 | .02 / .02 | .006 | .97 |
| 012 skip, ctx 8 + inverse | 524k | **.808** | **.707** | **.93 / .93** | .02 / .02 | .006 | .96 |
| 012 skip, ctx 8 + inverse | 1M | **.914** | **.865** | **.94 / .96** | **.97 / .04** | **.287** | .94 |
| 012 skip, ctx 8 + inverse | 2M | **.926** | **.913** | **.95 / .98** | **.98 / .02** | .172* | .95 |

\* At 2M, the predicted hunger after an inside click is 0.022 (truth 0.0), and after an outside
click 0.267 (truth 0.273). There is no false relief. The smaller "drop" comes from underestimating
the no-click hunger of inside states (0.19 vs 0.26 true).

Summary figure: `exp_00001/babbling_comparison.png` (all babbling runs, five metrics over
training). Per-run detail: `<run>/analysis/dynamics_over_training.png`.

Mean predicted cursor displacement (px) per commanded move, run 012 at 1M (truth ±4 per axis,
0 for stay). The displacement error is 1.2 px, against 4.1 px for a predictor that assumes no
motion.

| right | left | down | up | down-right | up-right | down-left | up-left | stay |
|---|---|---|---|---|---|---|---|---|
| (+3.2, −0.0) | (−3.3, +0.0) | (−0.1, +3.6) | (−0.1, −3.4) | (+2.3, +3.1) | (+2.8, −3.0) | (−2.3, +3.3) | (−2.6, −3.3) | (−0.1, +0.0) |

Without any reinforcement, the model:
- predicts where each of its moves takes the cursor, in direction and roughly in magnitude;
- predicts that a click removes the circle only when the cursor is inside it (0.97 vs 0.04);
- predicts that this click relieves its hunger (0.29 predicted vs about 0.27 true).

This is the interoceptive consequence that a reward would otherwise have to teach.
Figures: `012_P_babble_inv_s1/analysis/ckpt_01048576/counterfactual.png` (predicted next frames;
columns 9–12: cursor inside, prediction without a click, with a click, truth) and
`analysis/delta_524k.png` (predicted vs true change images: removal at the cursor, appearance in
the commanded direction).

Necessary ingredients, from the failed variants:
1. **Decoder inductive bias.** A decoder rendering from a global latent never draws the 1%-of-pixels
   cursor (001, 003–006). Change prediction plus U-Net skips plus a nonlinearity between the action
   and the local features (009/011/012) can.
2. **Update count.** The same model with context 32 (≈ 32 updates per 16k steps) shows nothing by
   1M (009). With context 8 (128 updates per 16k steps) it learns everything by 2M (011).
3. **Inverse dynamics** is not required but speeds learning about 2–4×. Movement is accurate by 524k
   instead of about 2M. It also makes the controllable cursor linearly decodable from the shared
   visual feature (R² 0.94–0.98, vs about 0.4 at init). The click/hunger contingencies appear at
   about 1M in both 011 and 012.

### R2. What each training signal puts into the representation

Ridge probes of cursor and target position on held-out random-policy frames. The table shows R²,
mean of x and y.

| Model | z: cursor | z: target | visual feature: cursor | visual feature: target |
|---|---:|---:|---:|---:|
| random init | .36 | .96 | .36 | .96 |
| reward-only PPO, 1M (010) | **.92** | **.91** | .76 | .94 |
| forward-prediction babbling, 2M (011) | .30 | .22 | .25 | .14 |
| forward + inverse babbling, 1M (012) | **.96** | .51 | **.96** | .54 |

Reward shapes the latent to encode both the cursor and the target, since the policy needs their
relation. Reward-free learning encodes the controllable entity (with inverse dynamics) and lets
the uncontrollable target decay. That is the known selectivity of inverse models (ICM, Pathak
et al. 2017). Forward prediction through skips computes the effects locally and builds no linear
position code in z.

### R3. Sparse-reward PPO with and without the prediction auxiliary (H2: weakly supported, 3 seeds)

Radius 12–16, 1M steps. "Skip arch" means `coord=1, pred_residual=1, pred_skip=1`, context 32,
default PPO schedule (4 epochs × 4 minibatches). Arm A has `pred_coef=0`, so its prediction head is
idle. Arm C has `pred_coef=1, pred_changed_weight=10` and is otherwise identical. Evaluation runs
every 5–10 iterations on 256 fixed-seed episodes with bonuses off (`summarize_runs.py`).

| run | steps to succ ≥ .95 | steps to det ≥ .9 | succ / det @180k | succ / det @344k | succ / det @508k | final median steps |
|---|---|---|---|---|---|---|
| A plain arch, s1 | 180k | 344k | 0.96 / 0.64 | 1.00 / 1.00 | 1.00 / 1.00 | 5 |
| A skip arch, s1 | 508k | 508k | 0.73 / 0.30 | 0.945 / 0.40 | 1.00 / 0.96 | 6 |
| A skip arch, s2 | 590k | 672k | 0.51 / 0.21 | 0.76 / 0.27 | 0.84 / 0.40 | 7 |
| A skip arch, s3 | 590k | 590k | 0.73 / 0.24 | 0.80 / 0.27 | 0.949 / 0.55 | 7 |
| **C** skip arch, s1 | **344k** | **426k** | 0.84 / 0.29 | 1.00 / 0.898 | 1.00 / 0.97 | 6 |
| **C** skip arch, s2 | **508k** | **508k** | 0.62 / 0.20 | 0.83 / 0.33 | 1.00 / 0.92 | 6 |
| **C** skip arch, s3 | **426k** | **508k** | 0.72 / 0.23 | 0.79 / 0.30 | 1.00 / 0.95 | 6 |

With the same architecture and seed, the prediction auxiliary reaches both thresholds earlier in
all three seeds, with 1.2–1.5× fewer steps:

| | A (mean of 3 seeds) | C (mean of 3 seeds) |
|---|---|---|
| steps to 95% success | 563k | 426k |
| steps to det ≥ 0.9 | 590k | 481k |

Each gap is one or two evaluation intervals (82k steps). Three seeds in the same direction give
p = 0.125 (one-sided sign test), so the effect is consistent but small. It is clearest at a fixed
budget of 508k steps: C has a reliable deterministic policy in all three seeds (det 0.92–0.97), A
in only one (0.40–0.96).

This is despite C's prediction head not learning movement at PPO's update rate (outcome-explained
at chance at 1M, R1 protocol). The benefit must therefore come from shaping the shared encoder,
not from a usable forward model.

The plain architecture (one seed) was fastest of all: 180k steps to 95%, against 508–590k for the
three skip-arch A seeds. That gap is larger than the spread among those seeds (up to 1.3×). In arm
A the prediction head is idle (`pred_coef=0`, `drive_coef=0`). The only differences that reach the
policy are therefore the CoordConv input channels and the different weight initialization they
imply. Plain-arch seeds 2–3 are needed to tell whether CoordConv slows RL or this is seed variance.

All arms converge to the same policy: success 1.00 and a median of 5–7 steps, against 52 for the
random policy.

### R4. Reward-free babbling pre-training transfers: PPO learns about 3× faster (2 seeds)

Same architecture for both runs (skip arch, context 8, inverse head), same PPO hyperparameters,
same PPO seed, radius 12–16. "Babbler init" loads all weights from the 1M checkpoint of a
reward-free babbler: run 012 for pair 1, run 015 (independent babbling seed) for pair 2.
Babbling used a random policy and no reward. The optimizer state is reset and the policy/value
heads are untouched by babbling.

| run | steps to succ ≥ .95 | steps to det ≥ .9 | succ / det @98k | succ / det @180k | succ / det @344k | succ / det @508k | median steps (end) |
|---|---|---|---|---|---|---|---|
| 016 PPO from scratch, seed 1 | 344k | 508k | 0.72 / 0.25 | 0.75 / 0.27 | 0.96 / 0.52 | 1.00 / 0.95 | 7 |
| 017 PPO from babbler-1 (012 @1M), seed 1 | **98k** | **180k** | **1.00 / 0.88** | **1.00 / 1.00** | 1.00 / 1.00 | 1.00 / 1.00 | **5** |
| 020 PPO from scratch, seed 2 | 590k | > 590k | 0.54 / 0.17 | 0.67 / 0.00 | 0.73 / 0.25 | 0.88 / 0.29 | 36 (at 590k) |
| 019 PPO from babbler-2 (015 @1M), seed 2 | **180k** | **262k** | 0.79 / 0.25 | **1.00 / 0.85** | 1.00 / 1.00 | 1.00 / 1.00 | **5** |

In both pairs, starting from reward-free sensorimotor knowledge cuts the rewarded steps needed:
- 95% success: 3.3–3.5× fewer steps.
- A reliable deterministic policy: 2.3–2.8× fewer, or more (scratch seed 2 had not reached it at
  590k).

The babbler-initialized agents also end with more efficient policies: median 5 steps and 3.5–4.1
clicks per success, against 7 steps and 5.9 clicks for scratch seed 1 at the end.

Accounting: the babbler itself used 1M environment steps, so this is not a saving in total
interaction. The claim is that *reward-free* experience substitutes for most of the rewarded
experience. Figure: `exp_00001/rl_curves.png`.

### R5. Small targets (radius 4–8): PPO with and without the prediction auxiliary (1 seed each)

Skip arch, 2M steps, seed 1. Arm A is `018_A_small_arch_s1` (PPO only). Arm C is
`021_C_small_arch_s1` (PPO + prediction, `pred_coef=1`, `pred_changed_weight=10`). Random-policy
success at this radius is 26–39%.

| env steps | 16k | 262k | 508k | 754k | 1.0M | 1.25M | 1.5M | 2.0M |
|---|---|---|---|---|---|---|---|---|
| A: succ / det | 0.30 / 0.00 | 0.33 / 0.05 | 0.32 / 0.07 | 0.53 / 0.09 | 0.86 / 0.58 | 0.96 / 0.86 | 0.99 / 0.96 | 1.00 / 0.98 |
| A: median steps | 67 | 53 | 67 | 61 | 40 | 22 | 13 | 10 |
| C: succ / det | 0.30 / 0.11 | 0.35 / 0.07 | 0.27 / 0.09 | 0.72 / 0.32 | 0.94 / 0.83 | 0.96 / 0.89 | 0.98 / 0.96 | 1.00 / 0.99 |
| C: median steps | 63 | 57 | 43 | 47 | 23 | 17 | 14 | 11 |

Both arms stay at the random level for about 0.6M steps and then solve the task. A reaches 95%
success at 1.25M steps, about 2.2× later than at radius 12–16 (mean of 3 seeds: 563k).

C leaves the plateau about 80k steps earlier and reaches both thresholds sooner:

| | A | C |
|---|---|---|
| 95% success | 1.25M | 1.08M |
| det ≥ 0.9 | 1.41M | 1.33M |

These gaps are one or two evaluation intervals in a single seed, within the seed spread seen at
radius 12–16. Both arms end at 1.00 success and a median of 10–11 steps. This long plateau with no
reward signal is the regime where reward-free pre-training should matter most.

## Analysis

**Reward-free learning of action consequences.** Learning is ordered by frequency and locality:
1. Local movement effects come first: every transition carries them.
2. The rare, non-local contingencies come about 0.5–1M steps later. The circle vanishing requires
   relating the cursor to the whole circle. Hunger relief requires mapping that event onto an
   internal variable. Under the random policy a successful click happens about once per 200
   steps, so by 1M the model has seen about 5k of them.

The model's predictions are also *specific*: it does not predict vanishing for clicks outside the
circle. So the click semantics are conditional on position, which is the causal structure of the
task. Nothing in the loss told the model that the circle, clicks or hunger matter. It learned them
because they are the parts of the world its actions affect.

**Why plain pixel prediction fails.** The controllable object is about 1% of the pixels. Copying
the last frame is 98–99% correct, so the gradient toward modelling the cursor is tiny relative to
the static scene. A decoder that renders from a global vector also struggles to place a small
sprite (the CoordConv observation). This matches the literature: UNREAL predicts changes, SimPLe
clips the loss, and Oh et al. 2015 use multiplicative action conditioning. Each fix alone was
insufficient here. Change prediction, U-Net skips and a nonlinearity between action and local
features together made it learnable. The number of gradient updates was the remaining bottleneck.

**Inverse dynamics as a self-localization signal.** Predicting one's own movement from two
consecutive frames forces the encoder to find the thing that moved because of the action. The
cursor becomes linearly decodable (R² 0.96) within about 100k steps, and forward-model learning
then accelerates 2–4×. The cost is that the representation discards what the agent cannot control
(target R² falls from 0.96 to about 0.5).

**Reward builds a different representation.** The RL agent's z encodes the cursor *and* the target
(R² about 0.9 each), because acting needs their relation. The self-centered reward-free
representation is still a much better starting point than random weights (R4: 3.5× faster).
- The encoder already localizes the controllable cursor.
- The forward model already encodes that clicking on the red region is consequential.
- PPO then only has to learn the policy mapping, plus recover the target information that
  babbling let fade.

**Caveats.**
- Replicated babbling result: run 015 (`P_babble_inv` seed 2) at 1M gives action retrieval 0.906,
  displacement correlation 0.95/0.97, vanish 0.92 inside vs 0.09 outside, and predicted hunger
  0.035 after an inside click vs 0.265 after an outside one. All other babbling configurations
  have one seed.
- Counterfactual tests use single-frame contexts, i.e. episode-start-like inputs, which are in
  distribution.
- Predicted cursors are blurry or faint. Displacements are read off with a whiteness-centroid
  detector, so the "± px" numbers are approximate.
- Successful clicks in babbling come from a random policy whose click probability is about 0.5.
  The contingency is learned from about 1 event per 200 steps.

## Conclusions

- A small pixels-to-mouse agent was built: a Python world plus a C++ model (CNN, causal Transformer,
  PPO), with every gradient verified against a PyTorch reference. It learns to aim and click from a
  sparse reward in about 0.35–0.5M steps at radius 12–16.
- **Without any reward, motor babbling plus prediction teaches the model what its actions do:**
  - how each move displaces its cursor (correlation 0.94–0.96, retrieval 0.91 vs 0.11 chance);
  - that clicking *on* the circle removes it (0.97) while clicking elsewhere does not (0.04);
  - that this click relieves hunger (0.29 vs about 0.27 true).
- This requires decoder inductive biases for small controllable objects and enough gradient
  updates. Inverse dynamics speeds it up 2–4× and makes the encoder localize the controllable
  cursor.
- Reward-only learning builds a representation of cursor *and* target; reward-free learning builds
  one of the controllable cursor and lets the target fade.
- **That reward-free knowledge pays off once reward appears.** PPO initialized from the babbler
  reaches 95% success 3.3–3.5× faster than from scratch: 98k vs 344k steps, and 180k vs 590k (2
  seeds, each with its own babbler).
- The prediction auxiliary during PPO (run C) also helps, but only modestly:
  - radius 12–16: 1.2–1.5× fewer steps, in the same direction in all 3 seeds;
  - radius 4–8: no clear effect (1 seed).
  PPO's update schedule is too sparse for the forward model to learn much.

## Next Steps

1. **Act with the learned model, without reward:**
   - Use the babbler's forward model and its learned "click on circle → hunger relief" prediction
     to choose actions by imagined multi-step rollouts, minimizing predicted hunger (plan phase 5).
   - The movement model is already accurate; if this works, goal-directed clicking emerges from a
     drive plus a learned world model, with no RL.
   - Feasibility check (run 012 at 1M, 128 states, repeated cardinal moves, predicted frames fed
     back as inputs): cursor error per imagined step is **0.84 px** at step 1 (no-motion baseline
     3.8), then 8.2, 27, 34 px. The blurry, faint predicted cursor is out of distribution for the
     encoder.
   - Multi-step imagination therefore needs latent dynamics (predict v_{t+1} from v_t and a_t;
     the inverse-trained v already encodes the cursor at R² 0.96), or sharper pixel predictions.
2. **Phasic auxiliary training for run C.** PPO's schedule gives the prediction head only about
   1k updates per 1M steps, too few to learn movement (C's head is at chance at 1M). Add PPG-style
   auxiliary epochs on replayed data.
3. **Harder conditions** where exploration matters:
   - radius 4–8: PPO needs about 1.1–1.25M steps, with or without prediction (R5); test babbler
     init there;
   - radius 3 / 100 steps / click p 0.1 (random success about 5%);
   - an empty-click cost.
4. Plain-arch PPO seeds 2–3, to test whether CoordConv slows RL (R3). Seeds for the babbling runs
   (2–3 each). Bootstrap at timeouts in GAE. Separate entropy
   coefficients for movement and click (see `RESEARCH.md` §9).
5. A GRU vs Transformer comparison (plan §9), and the browser twin driven by a trained agent in a
   real page (Playwright).
