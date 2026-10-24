# vm-00009: Planner as Teacher: a Habit Learned From the Planner's Own Moves

**Date:** 2026-10-24
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00009`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00009/`
**Task plan:** [`../tasks/00009-planner-as-teacher.md`](../tasks/00009-planner-as-teacher.md)

## Goal

So far the planner agent decides outside the network: a hand-written search (`plan.py`) looks for
imagined futures with lower hunger, and the network only predicts. Here its policy head also learns
to predict the moves the agent actually makes, mostly the planner's. Then the head acts alone,
without the planner.

The questions:
- Can a habit learned from the planner move to the circle and click?
- Does that learning harm the world model when it reaches the shared features?

## Hypothesis

1. **The habit aims.** The planner agent's features hold the direction to the circle (R² 0.84 from z
   at 100k), so the head's own moves beat its random initial model (0.53 test success) and every basic
   agent so far.
2. **Clicking is harder than moving for a linear head.** "Inside the circle" is not linear in the
   cursor and circle positions; a head with a hidden layer clicks more selectively.
3. **Through the features, the move loss may erode them**, as in the basic agent. The inverse-dynamics
   head may protect them.

## Method

- **Agent:** the planner agent of vm-00007 (`configs/L_gated.json`, memory U: every chunk stored),
  30k steps, 4 threads, seeds 1–2. The planner acts (5% babbled moves, and babbling when it imagines
  no gain). The policy head learns `-log π(a_t)` of the executed moves (`ppo.move_coef=1`), babbled
  ones included.
- **Arms:**

  | Arm | Policy head | Move loss reaches the features |
  |---|---|---|
  | T | linear | yes |
  | TD | linear | no (`ppo.pi_detach=1`) |
  | TDM | hidden layer of 128 | no |
  | TD+ | linear | no, and the heads clipped on their own (`ppo.pi_own_clip=1`; added at 14:00, see Log) |
  | TDM+ | hidden layer of 128 | no, own clip |

- **Controls:** vm-00007's L_U runs (the same agent without the move loss): exp_00007/009–010 and
  025–026.
- **Evaluation:**
  - the head's own moves, without the planner, on the 128 test episodes at 0, 10k, 20k and 30k
    (`online.py --eval`);
  - the planner's test success at 10k, 20k and 30k (`plan.py`) and its online hits;
  - feature probes and the head's aim cosine at 30k (`analyze_self_actor.py`).

## Log

### 2026-10-24
- No new code: `ppo.move_coef`, `ppo.pi_detach` and `model.pi_hidden` (vm-00008) exist, and
  `online.py --eval` scores any run's policy head.
- **Smoke runs** (300 steps, T and TDM): no errors, and the eval works on planner runs. The untrained
  head (checkpoint 0) hits 0.53 of the test episodes with 66 clicks per episode: its click
  probability is 0.5.
- **Launched** (`make runs`, runs 001–006) at 13:25, when the first 8 runs of the vm-00007 follow-up had
  finished. The configs are as intended: `move_coef` 1 in all; `pi_detach` 1 in TD and TDM;
  `pi_hidden` 128 in TDM (1,682,174 parameters).
- **The move loss slows the planner agent even when detached** (13:50, steps 0–5k; hits; the controls
  are vm-00007's L_U with the same seed, world and initial weights):

  | Seed | control | TD (detached) | T (into the features) |
  |---|---|---|---|
  | 1 | 63 | 38 | 30 |
  | 2 | 47 | 1 | 8 |

  - The likely cause is the global gradient-norm clip (`max_grad_norm` 0.5). The head's move-loss gradient
    raises the total norm from 1.4–1.9 to 3.4–7.0, so every update of the world model is scaled down two to
    four times more, by an amount that changes from step to step.
  - **Added `ppo.pi_own_clip`** (C++, default off, so all earlier runs are unchanged): the policy and value
    heads' gradient is clipped by its own norm and left out of the rest's. A test checks that with
    `pi_detach` and `pi_own_clip` a move loss leaves every parameter outside the heads bitwise unchanged,
    and that without `pi_own_clip` it does not. 300-step runs with the defaults are byte-identical.
  - **Launched TD+ and TDM+** (`make own-clip`, runs 007–010) at 14:00: TD and TDM with `pi_own_clip=1`.
- **The heads acting alone at 10k** (test success, seeds 1, 2; birth 0.56, 0.52, TDM 0.53, 0.54): T 0.59,
  0.31; TD 0.44, 0.36; TDM 0.21, 0.59. The controls' untrained heads: 0.53, 0.55 (their z changes as the
  model learns). No habit yet.
- **With `pi_own_clip`, the head no longer touches the world model** (14:45; online hits in steps
  0–5k / 0–10k):

  | Seed | control | TD | **TD+** | TDM | TDM+ |
  |---|---|---|---|---|---|
  | 1 | 63 / 119 | 38 / 100 | **63 / 119** | 3 / 14 | 21 / 70 |
  | 2 | 47 / 136 | 1 / 39 | **47 / 136** | 25 / 88 | 2 / 14 |

  - TD+ has the control's seed, world and initial weights, and its hits are identical: the planner acts
    exactly as without the move loss, while the head learns on the side. The shared clip was the whole
    cause of TD's slowdown.
  - The hidden layer changes the initial random draws, so TDM and TDM+ start from other weights than the
    control. Their planners learn at seed-dependent rates (vm-00005: 69–218 hits in the first 10k).
  - **The isolated linear habit beats birth at 10k:** TD+ 0.67, 0.64 (birth 0.56, 0.52), against TD 0.44,
    0.36. TDM+: 0.43, 0.41.
- **Runs 001–006 at 30k** (15:48–16:12). The head's own moves, without the planner, on the 128 test episodes;
  the frozen planner at 30k; probes from z; aim of the head's mean move:

  | Arm, seed | Head alone: 0 / 10k / 20k / 30k | Median steps, clicks per episode (30k) | Planner at 30k (median steps) | z: cursor / circle / direction | Aim | Online hits 0–30k |
  |---|---|---|---|---|---|---|
  | T s1 | 0.56 / 0.59 / 0.62 / 0.66 | 62.5, 47.9 | 0.79 (41) | 0.25 / 0.22 / 0.17 | +0.13 | 278 |
  | T s2 | 0.52 / 0.31 / 0.35 / 0.52 | 176, 35.6 | 0.78 (26) | 0.27 / 0.16 / 0.22 | +0.13 | 128 |
  | TD s1 | 0.56 / 0.44 / 0.80 / **0.96** | 26, 28.0 | 0.93 (9) | 0.96 / 0.69 / 0.83 | +0.47 | 1,310 |
  | TD s2 | 0.52 / 0.36 / 0.76 / **0.93** | 30, 30.0 | 0.88 (14) | 0.89 / 0.64 / 0.80 | +0.57 | 951 |
  | TDM s1 | 0.53 / 0.21 / 0.51 / 0.68 | 65.5, 54.0 | 0.86 (9) | 0.90 / 0.58 / 0.78 | +0.23 | 398 |
  | TDM s2 | 0.54 / 0.59 / 0.80 / **0.89** | 27.5, 26.5 | 0.81 (9) | 0.94 / 0.62 / 0.80 | +0.47 | 781 |
  | control s1 | – | – | 0.92 (8) | 0.90 / 0.62 / 0.78 | +0.12 (untrained head) | 892 |
  | control s2 | – | – | 0.95 (8) | 0.92 / 0.61 / 0.80 | −0.02 (untrained head) | 1,341 |

  - **The detached heads learned a habit that works without the planner:** 0.68–0.96 at 30k, against
    0.52–0.56 at birth. The best basic agent so far reached 0.81–0.82, and only at 50k (vm-00008 DMI). In three
    of the four detached runs the habit matches or beats its own frozen planner (0.96 vs 0.93, 0.93 vs 0.88,
    0.89 vs 0.81), but it needs two to three times the steps (median 26–30 against 9–14).
  - **Through the features, the move loss erodes them** (hypothesis 3): cursor 0.25–0.27 against 0.89–0.96,
    direction 0.17–0.22 against 0.78–0.83. The planner gets weaker (0.78–0.79), the agent hits 3–10 times less
    often online, and the habit stays at 0.52–0.66.
  - **The hidden layer did not help** (hypothesis 2 not supported so far): TDM 0.68, 0.89 against TD 0.96, 0.93.
  - The detached arms still share the clip with the world model, so their experience differs from the
    control's: TD s1 hit more often than its control online, the other three less. TD+ (own clip: the
    control's exact experience) is the clean test, at about 16:25.
- **TD+ and TDM+ at 30k** (16:19–16:32):
  - **TD+ is the control plus a head on the side:** its logged metrics equal the control's at every 1k step
    through 30k, and so do its planner test (0.92, 0.95) and its probes. Its head alone: 0.73, 0.89 at 20k and
    **0.80, 0.91** at 30k; aim +0.48, +0.55.
  - TDM+: 0.56, 0.64 at 20k and 0.77, 0.55 at 30k. The hidden layer changes the initial weights, and seed 2's
    planner learned slowly (244 online hits in 0–30k; planner test 0.79).
  - **The controls' untrained heads stay at birth** (0.54, 0.55 at 20k; 0.54, 0.56 at 30k): the habit comes from
    the move loss, not from the features changing under a fixed random head.

## Results

### R1. The head's own moves, without the planner (the 128 test episodes; seeds 1, 2)

| Arm | Head | Move loss in the features | Clip | birth | 10k | 20k | 30k | 30k: median steps (clicks per episode) |
|---|---|---|---|---|---|---|---|---|
| control (no move loss) | linear, untrained | – | – | 0.56, 0.52 | 0.53, 0.55 | 0.54, 0.55 | 0.54, 0.56 | 168.5 (66), 154 (66) |
| T | linear | yes | shared | 0.56, 0.52 | 0.59, 0.31 | 0.62, 0.35 | 0.66, 0.52 | 62.5 (48), 176 (36) |
| TD | linear | no | shared | 0.56, 0.52 | 0.44, 0.36 | 0.80, 0.76 | **0.96, 0.93** | 26 (28), 30 (30) |
| **TD+** | linear | no | own | 0.56, 0.52 | 0.67, 0.64 | 0.73, 0.89 | **0.80, 0.91** | 44 (31), 27 (27) |
| TDM | hidden layer | no | shared | 0.53, 0.54 | 0.21, 0.59 | 0.51, 0.80 | 0.68, 0.89 | 65.5 (54), 27.5 (27) |
| TDM+ | hidden layer | no | own | 0.53, 0.54 | 0.43, 0.41 | 0.56, 0.64 | 0.77, 0.55 | 30.5 (29), 116.5 (33) |

- With 128 episodes, one success rate has a standard error of about 0.04.
- For comparison, the best basic agent so far (vm-00008 DMI, a habit learned from its own moves with the
  hindsight command) reached 0.63–0.82, at 50k.

### R2. The planner (frozen, the same test episodes) and online hits

| Arm | Planner 10k | 20k | 30k | 30k median steps | Online hits 0–30k (episodes) | 20k–30k |
|---|---|---|---|---|---|---|
| control | 0.70, 0.93 | 0.80, 0.84 | 0.92, 0.95 | 8, 8 | 892 (963), 1,341 (1,365) | 466 (479), 692 (695) |
| T | 0.58, 0.36 | 0.45, 0.52 | 0.79, 0.78 | 41, 26 | 278 (376), 128 (258) | 140 (167), 68 (107) |
| TD | 0.81, 0.86 | 0.85, 0.80 | 0.93, 0.88 | 9, 14 | 1,310 (1,361), 951 (996) | 725 (732), 536 (538) |
| TD+ | = control | = control | 0.92, 0.95 | 8, 8 | = control | = control |
| TDM | 0.18, 0.94 | 0.59, 0.96 | 0.86, 0.81 | 9, 9 | 398 (493), 781 (829) | 302 (309), 373 (384) |
| TDM+ | 0.42, 0.38 | 0.95, 0.57 | 0.96, 0.79 | 7, 15 | 787 (858), 244 (347) | 511 (515), 167 (185) |

TD+'s world model is bitwise the control's: every logged metric matched at every 1k step through 30k, and so
did the planner test and the probes.

### R3. Features at 30k (linear R² on 1,024 held-out random states; from z, and from v in brackets)

| Arm | cursor | circle | direction |
|---|---|---|---|
| control (= TD+) | 0.90, 0.92 (0.92, 0.94) | 0.62, 0.61 (0.65, 0.68) | 0.78, 0.80 (0.80, 0.82) |
| T | 0.25, 0.27 (0.39, 0.47) | 0.22, 0.16 (0.38, 0.31) | 0.17, 0.22 (0.27, 0.36) |
| TD | 0.96, 0.89 (0.97, 0.91) | 0.69, 0.64 (0.75, 0.71) | 0.83, 0.80 (0.84, 0.82) |
| TDM | 0.90, 0.94 (0.92, 0.96) | 0.58, 0.62 (0.63, 0.70) | 0.78, 0.80 (0.80, 0.82) |
| TDM+ | 0.90, 0.86 (0.93, 0.89) | 0.66, 0.33 (0.68, 0.39) | 0.80, 0.69 (0.82, 0.71) |

### R4. Aim at 30k (cosine between the head's mean move and the direction to the circle, away from the walls)

| Arm | aim | share of steps at a wall |
|---|---|---|
| control (untrained head) | +0.12, −0.02 | 0.24, 0.24 |
| T | +0.13, +0.13 | 0.61, 0.71 |
| TD | +0.47, +0.57 | 0.36, 0.51 |
| TD+ | +0.48, +0.55 | 0.54, 0.42 |
| TDM | +0.23, +0.47 | 0.52, 0.47 |
| TDM+ | +0.38, +0.21 | 0.57, 0.71 |

## Analysis

### The habit works without the planner (hypothesis 1, confirmed)

- **Every detached head beats its birth by 30k**, and the linear ones by far: TD 0.96, 0.93 and TD+ 0.80, 0.91,
  against 0.56, 0.52. The controls' untrained heads stay at birth (0.54, 0.56). So the habit comes from the
  move loss, not from features changing under a random head.
- **TD+ is the cleanest test.** Its planner, memory and world model are bitwise the control's. The head only
  watches the planner's executed moves and learns to predict them. Acting alone, it hits in 80% and 91% of the
  test episodes.
- **The habit aims** (cosine +0.47 to +0.57, where the untrained head scores −0.02 to +0.12) and **clicks
  freely** (27–31 clicks per episode). Clicks cost nothing in this world.
- **It is slower than its teacher**: median 26–44 steps against the planner's 8–14. It averages the planner's
  moves in each state, while the planner searches for the best move each time.
- **It beats the basic agent of vm-00008** (0.63–0.82 at 50k) in 30k steps, with a linear head. The basic
  agent imitates its own moves, mostly random ones, and its memory over-represents the long failures. Here the
  teacher hits in 97% or more of its online episodes between 20k and 30k (TD, control). The memory holds
  approaches and clicks, and z holds the direction (R² 0.78–0.83), so a linear readout suffices.

### The move loss through the features erodes them (hypothesis 3, confirmed)

- **T's features lose the cursor and the direction**: from z 0.25–0.27 and 0.17–0.22, against the
  control's 0.90–0.92 and 0.78–0.80.
- **Both the planner and the habit suffer.** The planner tests at 0.79, 0.78 (control 0.92, 0.95), with median
  steps 26–41 against 8. T hits 3–10 times less often online (278 and 128 hits in 0–30k, against 892 and 1,341),
  and its habit reaches only 0.66, 0.52.
- This is the erosion vm-00008 found in the basic agent, now in the planner agent: predicting one's own moves
  through the shared features wears them down. Detached, the features match the control's (TD, TDM, TDM+:
  cursor 0.86–0.96).

### A hidden layer does not help (hypothesis 2, not supported)

- **TDM and TDM+ reach 0.68, 0.89 and 0.77, 0.55**, against the linear TD 0.96, 0.93 and TD+ 0.80, 0.91. The hidden
  layer also changes the initial weights, so the planners differ. TDM+ seed 2's planner learned slowly (244
  online hits in 0–30k, planner 0.79), and its habit is the weakest (0.55).
- In vm-00008 the hidden layer mattered because the command had to steer the head: "aim, if relief is
  asked for" is not linear. Here the teacher's moves are good on their own, and the planner agent has no
  command. The direction, linear in z, is enough.

### The shared clip

- Sharing the gradient-norm clip with the world model slowed the planner early (log, 13:50: seed 2 hit once in
  the first 5k steps, against the control's 47). It changed the whole lifetime, but did not spoil the habit:
  TD's 0.96, 0.93 against TD+'s 0.80, 0.91. Two seeds cannot tell whether TD's higher seed-1 value (0.96 against
  0.80) is the clip or that lifetime: TD seed 1 hit 1,310 times online, the control 892.

### Caveats

- 2 seeds per arm, 30k steps. The habit was still rising at 30k in 7 of the 8 detached runs.
- The tests use frozen checkpoints, which understate an online learner (vm-00005): the planner and the habit
  alike.
- Clicks are free. A habit that clicks in most steps succeeds here; with a cost on clicks it would have to
  learn when to click.
- The hidden-layer arms start from other initial weights than the linear ones and the control.

## Conclusions

1. **A habit learned from the planner's own moves acts alone.** The policy head predicts the moves the agent
   executes, mostly the planner's. Without reward and without the planner, in one forward pass per step, it
   hits in 0.80–0.96 of the test episodes at 30k with a linear head (birth 0.52–0.56), aiming at the circle.
   The skill that a search outside the network found now sits inside the network.
2. **The move loss must not reach the features.** Through them it erodes the world model (cursor 0.25–0.27
   against 0.90–0.92), and both the planner and the habit suffer.
3. **Fully isolated (`ppo.pi_detach=1`, `ppo.pi_own_clip=1`), the head learns on the side** and leaves the world
   model's learning bitwise unchanged.
4. **A hidden layer does not help here**: 0.55–0.89 against 0.80–0.96 for the linear head.
5. **The habit is slower than the planner**: 26–44 median steps against 8–14.

## Next Steps

- **Hand over:** after 30k, turn the planner off and let the habit act with learning on. Its own moves are now
  good, so self-imitation (vm-00006–8) starts from a good teacher. Does the skill hold, improve or decay?
- **The habit as the planner's proposal:** start the search from the habit's move (amortized planning). Fewer
  samples, and the planner refines what the habit proposes.
- **Longer:** TD+ to 100k. Does the habit catch up with the planner's speed?
- **A cost on clicks:** does the habit learn when to click?
