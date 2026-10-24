# vm-00006: The Basic Agent: Predict the Next State, Make That Move

**Date:** 2026-10-23
**Status:** Complete
**Source:** `exp/vm-00006` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00006/`
**Task plan:** [`../tasks/00006-basic-next-state-predictor.md`](../tasks/00006-basic-next-state-predictor.md)

## Goal

The simplest agent the owner asked for, with no planning:
- **One world, one agent.** It reads its last 8 real states: frame, hunger and restlessness, and its
  previous move.
- **One prediction:** the next state, i.e. its next move, plus the next frame and next drives
  given that move.
- **Act:** it makes the predicted move.
- **Learn on every step:** it compares the predictions with what actually happened.
No reward is used anywhere.

## Hypothesis

1. The agent learns the world (next frame, next drives) as the babblers did.
2. Its moves cannot become goal-directed. Learning to predict its own sampled moves has zero
   expected gradient, so the move distribution only drifts.
3. One extra input fixes this with no planning and no reward: the hunger relieved within the next
   12 steps (A+H). In training it is the true value from memory (hindsight); when acting, the
   agent asks for relief of its current hunger. It then repeats the moves that came before hunger
   relief.

## Method

- **World:** one `CircleWorld` (radius 12–16, drives observed, no bonuses, auto-reset after a hit or
  200 steps).
- **Model:** the vm-00001 architecture with a context of 8 steps (CoordConv, residual prediction,
  U-Net skips). About 1.4M parameters, randomly initialized.
  - The policy head predicts the next move (dx, dy, click).
  - The prediction head predicts the next frame and drives given that move.
  - There is no latent world model and no inverse dynamics.
- **Every step** (`dev/online.py`, `online.actor=self`):
  1. **Act.** Forward the window of the last 8 real states; sample the move from the predicted
     distribution and make it. With probability ε, babble instead.
  2. **Learn.** One AdamW step (lr 3e-4) on the newest window of 8 steps. The losses:
     - next frame;
     - next drives;
     - the move actually made: −log π(a), `ppo.move_coef`.
  - **A+H:** the windows end 11 steps back, so that their hunger-relief labels are complete.
- **Initial moves:** dx, dy ~ N(0, 0.6); click probability 0.1.
- **Arms**, each an independent agent in its own world (8 in parallel, 4 threads each, 100k
  steps):

  | Arm | Hunger input | Babbling ε | Learns from | Seeds |
  |---|:-:|---:|---|---|
  | A | | 0.05 | newest window | 1, 2 |
  | A+H | ✓ | 0.05 | newest window | 1, 2 |
  | A+H | ✓ | 0.2 | newest window | 1, 2 |
  | A+H+R | ✓ | 0.2 | newest window + 7 replayed (memory 100k steps) | 1, 2 |
- **Evaluation** every 10k steps: the agent's own moves on vm-00002's 128 test episodes (A+H asks
  for relief of its current hunger), plus online hits per 1,000 steps.

## Log

### 2026-10-23
- Added `ppo.move_coef`: the next-move loss on the move actually made, on top of
  frame and drive prediction. It works without PPO. Gradients match the PyTorch reference; 192
  tests pass.
- Added `online.actor=self`, the hindsight hunger input (`online.goal`) and `online.py --eval`.
- **Smoke tests (3,000–5,000 steps) confirm hypothesis 2.**
  - **Without babbling (A):** the predicted click probability fell from 0.10 to 0.011 in 3,000
    steps; the agent drifted into never clicking, with no hits after the first 1,000 steps.
  - **A+H with ε 0.05:** the same collapse, no hits in 3,000 steps. Test success fell from 0.48
    (the random initial model, close to random actions at 0.46) to 0.06.
  - **A+H with ε 0.2:** hits keep coming (1 → 4 per 1,000 steps), the click probability recovers
    to 0.09–0.13, and test success at 5k steps is 0.27.
- Launched the 8 runs at 21:34.

- **Up to 50k steps** (23:33; the runs are at 79k, A+H+R at 44k):
  - **Test success of its own moves** (seeds 1, 2) stays below the random initial model:

    | Arm | birth | 10k | 20k | 30k | 40k | 50k |
    |---|---|---|---|---|---|---|
    | A, ε 0.05 | 0.48, 0.44 | 0.17, 0.14 | 0.27, 0.06 | 0.07, 0.09 | 0.02, 0.16 | 0.15, 0.10 |
    | A+H, ε 0.05 | 0.48, 0.44 | 0.18, 0.21 | 0.16, 0.20 | 0.10, 0.23 | 0.07, 0.30 | 0.25, 0.21 |
    | A+H, ε 0.2 | 0.48, 0.44 | 0.17, 0.17 | 0.30, 0.51 | 0.12, 0.27 | 0.30, 0.16 | 0.40, 0.31 |
    | A+H+R, ε 0.2 | 0.48, 0.44 | 0.02, 0.23 | 0.04, 0.26 | 0.11, 0.20 | 0.16, 0.25 | |

  - **Online**, in the last 5k steps: 5–7 hits in 28–30 episodes with ε 0.05, 13–14 in 33–34 with
    ε 0.2, and 7–11 in 29–30 for A+H+R. Random babbling hits in about half of the episodes
    (vm-00005 B).
  - **The moves narrow without aiming.** The std of dx falls from 0.60 to about 0.40 (A, A+H with
    ε 0.05), and to 0.24 with replay (A+H+R, whose move NLL falls from 1.55 to 0.15). The agents
    still click 6–48 times per test episode (15 at birth). The hindsight hunger input has not
    produced a measurable preference for circles: A+H with ε 0.05 is close to A.

### 2026-10-24
- **All 8 runs reached 100k steps** (about 01:50), and every checkpoint has been evaluated.
  - **Test success** of its own moves (seeds 1, 2), 60k–100k steps:

    | Arm | 60k | 70k | 80k | 90k | 100k |
    |---|---|---|---|---|---|
    | A, ε 0.05 | 0.23, 0.02 | 0.03, 0.16 | 0.08, 0.05 | 0.27, 0.15 | 0.14, 0.09 |
    | A+H, ε 0.05 | 0.24, 0.14 | 0.08, 0.12 | 0.18, 0.27 | 0.27, 0.18 | 0.20, 0.09 |
    | A+H, ε 0.2 | 0.31, 0.38 | 0.20, 0.41 | 0.21, 0.34 | 0.38, 0.30 | 0.34, 0.38 |
    | A+H+R, ε 0.2 | 0.22, 0.24 | 0.11, 0.27 | 0.21, 0.24 | 0.27, 0.27 | 0.23, 0.30 |

    No checkpoint of any arm beats the random initial model (0.48, 0.44), except A+H ε 0.2 seed 2
    at 20k (0.51), once.
  - **Online hits (seeds 1, 2), including the babbled moves:**

    | Arm | 0–10k | 40–50k | 90–100k |
    |---|---|---|---|
    | A, ε 0.05 | 7/55, 9/55 | 7/55, 10/57 | 16/63, 7/56 |
    | A+H, ε 0.05 | 14/60, 18/65 | 10/59, 15/59 | 18/62, 12/59 |
    | A+H, ε 0.2 | 20/64, 14/58 | 20/59, 20/64 | 34/68, 26/68 |
    | A+H+R, ε 0.2 | 1/50, 11/57 | 10/56, 20/58 | 24/63, 26/66 |

    Random babbling hits in 0.47–0.48 of its episodes (vm-00005 B, 200k steps). A+H ε 0.2 seed 1
    reaches that level in the last 10k steps (0.50); the others stay below it, but A+H, ε 0.2
    and A+H+R rise slowly.
  - **Move std at 100k:** dx 0.35–0.38 with ε 0.05, 0.53–0.56 for A+H ε 0.2, and 0.24–0.26 for A+H+R.
    The predicted click probability stays near its initial 0.1 (0.05–0.14).
- **Without replay, the hunger input became invisible: A vs A+H was not a test of it.** Found
  while checking why asking for no relief (command 0) changes nothing at test.
  - **The encoder features blow up in 5 of the 6 runs without replay.** Mean norm of the visual
    features v on 32 test frames (3.8 at birth):

    | Run | 10k | 20k | 30k | 50k | 70k | 100k |
    |---|---|---|---|---|---|---|
    | A, ε 0.05, s1 | 1,120 | 955 | 1,940 | 2,200 | 35,800 | 99,000 |
    | A, ε 0.05, s2 | 1.4 | 5.0 | 1.1 | 0.9 | 2.7 | 2.3 |
    | A+H, ε 0.05, s1 | 369 | 2,380 | 5,390 | 4,600 | 7,200 | 16,300 |
    | A+H, ε 0.05, s2 | 208 | 293 | 415 | 1,650 | 3,400 | 596 |
    | A+H, ε 0.2, s1 | 61 | 445 | 878 | 9,470 | 14,400 | 10,400 |
    | A+H, ε 0.2, s2 | 562 | 151 | 1,200 | 8,920 | 488 | 22,800 |
    | A+H+R, s1 | 2.0 | 3.0 | 3.8 | 5.0 | 6.0 | 6.9 |
    | A+H+R, s2 | 1.7 | 2.7 | 3.5 | 4.6 | 5.8 | 7.1 |

  - **The drives and the command (both of size about 1) then vanish from the tokens.** At 100k,
    replacing the drives or the command changes the predicted move (mean |Δμ| over the 128 test
    states):
    - in the 5 runs that blew up: by 1e-8 to 3e-6;
    - in A seed 2, which did not: by 0.026 for the drives;
    - in A+H+R: by 0.09–0.12 for the drives and by 0.017–0.026 for the command.

    Another image changes it by 0.42–0.48 in A+H+R, but only by 0.0002–0.03 without replay, so those
    moves are also nearly state-independent.
  - **The drive prediction fails along with it.** Its error is 0.06–0.14 from 10k on in all 6 runs
    without replay, against 0.003–0.005 in A+H+R (A seed 2, which did not blow up, recovers to
    0.013 by 100k).
  - **Not caused by batch 1 alone.** vm-00005 P also learns from the newest window only, but with no
    move loss. Its features stay at 1.3–2.6 (100k). The run-away seems to come from the move loss;
    the mechanism is not known. There is no weight decay (`ppo.weight_decay=0`).
  - **Consequence:** in A and A+H, the agent could not see its hunger or the command, so "A+H ≈ A"
    says nothing about the hindsight input. Even with replay, where everything is visible, the
    command has only a small effect on the moves (0.026, against a move std of 0.26).
- **No agent aims; they get stuck at the walls** (100k, the 128 test episodes, scored with the
  ground truth):
  - **Aiming:** take the steps with the cursor outside the circle and at least 10 px from every
    wall. There, the cosine between the predicted mean move and the direction to the circle is −0.21
    to +0.02 in every run, against +0.02 and −0.01 at birth. Asking for no relief changes it by at
    most 0.014.
  - **Walls:** the cursor is within 2 px of a wall in 58–85% of the steps with ε 0.05, 27–33% for
    A+H ε 0.2, and 45–58% for A+H+R, against 23% at birth. This, not aiming away, is why test success
    falls below birth.
  - **Not momentum:** the predicted move follows the previous move only in A seed 2 (slope 0.24).
    A+H+R's mean moves are large (0.49–0.54 of a full step) and depend on the image, but not on the
    previous move (slope −0.05 to 0.00).
  - **A likely reading:** self-imitation keeps whatever field of moves it has, since every field
    reproduces itself. Where the field points into a wall, the cursor stays, so those states fill
    the memory.

## Results

All test numbers are for the agent's own moves on vm-00002's 128 test episodes (radius 12–16,
200-step limit). A+H asks for relief of its current hunger. "Birth" is the random initial model
(0.48, 0.44); random actions score 0.46.

### R1. Test success over the lifetime (seeds 1, 2)

| Arm | birth | mean of the 10k–100k checkpoints | best checkpoint | 100k |
|---|---|---|---|---|
| A, ε 0.05 | 0.48, 0.44 | 0.14, 0.10 | 0.27 (s1, 90k) | 0.14, 0.09 |
| A+H, ε 0.05 | 0.48, 0.44 | 0.17, 0.20 | 0.30 (s2, 40k) | 0.20, 0.09 |
| A+H, ε 0.2 | 0.48, 0.44 | 0.27, 0.33 | 0.51 (s2, 20k) | 0.34, 0.38 |
| A+H+R, ε 0.2 | 0.48, 0.44 | 0.15, 0.25 | 0.30 (s2, 100k) | 0.23, 0.30 |

The Log has every checkpoint.

### R2. Online hits in the last 10k steps, including the babbled moves

| Arm | seed 1 | seed 2 |
|---|---|---|
| A, ε 0.05 | 16/63 (0.25) | 7/56 (0.13) |
| A+H, ε 0.05 | 18/62 (0.29) | 12/59 (0.20) |
| A+H, ε 0.2 | 34/68 (0.50) | 26/68 (0.38) |
| A+H+R, ε 0.2 | 24/63 (0.38) | 26/66 (0.39) |

Random babbling hits in 0.47–0.48 of its episodes (vm-00005 B).

### R3. What the agents learned, at 100k (seeds 1, 2)

| Arm | feature norm \|v\| (birth 3.8) | aim cosine (≥ 10 px from the walls) | steps at a wall (birth 0.23) | dx std (birth 0.60) |
|---|---|---|---|---|
| A, ε 0.05 | 99,000, 2.3 | +0.02, −0.02 | 0.82, 0.85 | 0.35–0.38 |
| A+H, ε 0.05 | 16,300, 596 | −0.02, −0.05 | 0.58, 0.77 | 0.35–0.38 |
| A+H, ε 0.2 | 10,400, 22,800 | −0.21, +0.02 | 0.27, 0.33 | 0.53–0.56 |
| A+H+R, ε 0.2 | 6.9, 7.1 | +0.02, +0.01 | 0.58, 0.45 | 0.24–0.26 |

- The aim cosine is between the predicted mean move and the direction to the circle, with the
  cursor outside the circle. At birth: +0.02, −0.01.
- **What the predicted move depends on** (mean |Δμ| over the 128 test states when one input is
  replaced):

  | Input replaced | the 5 runs that blew up | A, ε 0.05, s2 | A+H+R |
  |---|---|---|---|
  | drives | 1e-8 to 3e-6 | 0.026 | 0.09–0.12 |
  | hindsight command | 1e-8 to 3e-6 | (none) | 0.017–0.026 |
  | image | 0.0002–0.03 (all six runs without replay) | | 0.42–0.48 |

- **Drive prediction error:** 0.06–0.14 from 10k on in the 6 runs without replay (A seed 2
  recovers to 0.013 by 100k), against 0.003–0.005 in A+H+R.
- **Next-frame loss** on each agent's own stream falls in every run, from 0.07–0.09 (first 1,000
  steps) to 0.03–0.04 (steps 95–100k). vm-00005's babbler, a different model with different
  data, reaches 0.018–0.020, so this comparison is only rough.

## Analysis

1. **The world is learned, but the drives only with replay (hypothesis 1, partly).**
   - The next-frame loss falls in all 8 runs.
   - Without replay, the encoder features run away in 5 of 6 runs, to norms of 10²–10⁵. Next to
     them, the drives and the command (both of size about 1) vanish from the tokens. The drive
     prediction fails, and the moves no longer depend on the drives or the command.
   - vm-00005's P learns from the newest window only too, but has no move loss, and its features
     stay at 1.3–2.6. The move loss, with no weight decay, is the likely trigger; the mechanism is
     not known.
   - With replay (A+H+R), the features stay healthy (|v| about 7) and the drive error is more than
     ten times lower.
2. **Self-imitation does not aim, and its drift hurts (hypothesis 2, confirmed).**
   - No run aims at 100k: the cosine is −0.21 to +0.02, against +0.02 and −0.01 at birth.
   - The moves narrow, from a dx std of 0.60 at birth to 0.24–0.38, except with ε 0.2 and no
     replay (0.53–0.56), where 20% of the imitated moves are babbled.
   - On average, every arm then does worse than the random model it started from: mean test
     success is 0.12–0.30, against 0.44–0.48 at birth. The cause is the walls, not aiming away: the cursor
     sits within 2 px of a wall in 27–85% of the test steps, against 23% at birth.
   - **A likely reading:** imitating its own moves keeps whatever field of moves the model has,
     since every field reproduces itself. Where the field points into a wall, the cursor stays,
     and the memory fills with those states.
   - Momentum is ruled out: the predicted move follows the previous move only in A seed 2.
3. **The hindsight relief input did not make the moves goal-directed (hypothesis 3, not
   confirmed).**
   - Without replay, it was not tested: the input was invisible (|Δμ| ≤ 3e-6). "A+H ≈ A"
     therefore says nothing about it. A+H with ε 0.2 does best (0.27–0.33), probably because 20%
     of its moves are babbled: wider moves, fewer steps at a wall (0.27–0.33).
   - With replay, the input is visible but weak: replacing it moves the mean by 0.017–0.026,
     against a move std of 0.26, and there is no aiming.
   - Hits are rare in the agent's own experience (about 1–3 per 1,000 steps at the end), so few
     training windows carry a nonzero relief label.
   - vm-00007 tested storing those windows preferentially and found no gain. Its probes found a
     second problem: the basic agent's features lose the cursor during training.

## Conclusions

- **A pure next-state predictor that imitates its own moves does not become goal-directed.**
  - In all 8 lifetimes (100k steps each), no agent aims.
  - Every arm ends below the random model it started from: mean test success 0.12–0.30, against
    0.44–0.48. Its moves narrow into fields that pin the cursor at the walls.
- **The self actor needs replay.** Learning from the newest window only makes its encoder
  features blow up (norm 10²–10⁵ in 5 of 6 runs), which hides its hunger and the command. So the
  A vs A+H comparison without replay is no test of the hindsight input.
- **With replay, the hindsight relief input is visible but has almost no effect** on the moves
  (0.017–0.026, against a move std of 0.26), and produces no aiming within 100k steps.

## Next Steps

- [x] Raise the share of relief in what the agent replays: memory gated by relief and surprise
  (vm-00007). Result: no gain for the basic agent.
- [ ] Keep the move loss's gradient out of the shared features (detach the policy head), to test
  whether self-imitation is what erodes them (vm-00007).
- [ ] Find the cause of the feature run-away without replay, e.g. with weight decay.
- [ ] Use the planner as a teacher (habit): the topic README's open questions.
