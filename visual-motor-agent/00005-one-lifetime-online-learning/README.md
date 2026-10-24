# vm-00005: One Lifetime: a Single World, Learning on Every Step

**Date:** 2026-10-23
**Status:** Complete
**Source:** `exp/vm-00005` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00005/`
**Task plan:** [`../tasks/00005-one-lifetime-online-learning.md`](../tasks/00005-one-lifetime-online-learning.md)

## Goal

In vm-00001 to vm-00004, learning was split from acting in two ways:
- **Batches.** 128 parallel worlds acted with fixed weights for 128 steps, then the model learned
  from those 16,384 transitions.
- **Phases.** Babbling first (2.1M steps), then planning.

Here, a single agent lives in a single world. It acts and learns at the same time, with one weight
update after every step, from its own experience only. Each prediction uses only the current frame.
There is no reward: hunger and restlessness are internal states, which it observes and learns to
predict.

## Hypothesis

1. Learning on every step from its own stream teaches the same action consequences as the batched
   babbler: cursor movement, the click contingency and hunger relief.
2. Its behavior turns from exploratory into goal-directed clicking within one lifetime, with no
   phase switch and no reward. Test success of its planner rises from the untrained level (0.16)
   toward vm-00002 and vm-00004 (0.89–1.00).
3. A memory of its own past (replay) is needed. Learning only from the newest steps forgets rare
   events such as hits.

## Method

- **World:** one `CircleWorld` (radius 12–16, drives observed, no bonuses). After a hit or 200
  steps, a new circle and cursor appear.
- **Model:** as vm-00002's `P_dyn`, plus `model.one_frame=1`: each token attends only to itself, so
  every prediction uses the current frame only. The token also carries the drives and the previous
  action.
  - Heads: next frame, next drives, inverse dynamics, and latent dynamics with a hunger head.
  - About 1.67M parameters, randomly initialized.
- **Every step** (`dev/online.py`):
  1. **Act.** The CEM planner (vm-00002 settings) plans with the current weights. The agent
     babbles instead (dx, dy ~ N(0, 0.607), click with probability 0.1) with probability 0.05, or
     when its plan does not lower imagined hunger below doing nothing by 0.02 per step.
  2. **Step** the world and store the transition in the agent's memory.
  3. **Learn.** One gradient step (AdamW, lr 3e-4) on the newest 8 steps, plus 7 sequences of 8
     steps replayed from memory. The losses are reward-free.
- **Arms**, each an independent model in its own world, run in parallel for 200k steps:

  | Arm | Learns from | Acts by | Seeds |
  |---|---|---|---|
  | **L** | newest 8 steps + 7 replayed sequences (memory 100k steps) | planner, 5% babbling | 1–4 |
  | **P** | newest 8 steps only (memory 8 steps) | planner, 5% babbling | 1–3 |
  | **B** | newest 8 steps + 7 replayed sequences | always babbling | 1–2 |
- **Evaluation** (frozen checkpoints every 10k steps, measurement only):
  - the planner on vm-00002's 128 test episodes (`plan.py`);
  - imagination tests (`analyze_imagination.py`);
  - online hits and episode lengths during the lifetime (`metrics.jsonl`).

## Log

### 2026-10-23
- Implemented `model.one_frame` (C++, checkpoint header slot 36, PyTorch reference) and
  `vm_rollout_store` (training from stored experience).
- New tests:
  - with `one_frame`, acting equals training bitwise (C++), and all gradients match the
    reference (a new fixture variant);
  - a rollout filled from stored experience gives identical gradients.
  176 Python tests and the C++ tests pass.
- Smoke runs (3,000 steps): the agent hit 17 of 17 circles in steps 2,500–3,000 (mean episode
  length 27). Inverse-dynamics R² reached 0.85.
  - **The exploration gate rarely fires.** The planner's plan beat doing nothing by an imagined
    0.2–0.4 hunger per step from the start: the CEM optimizes against even an untrained hunger
    head. So the agent plans on about 95% of steps from step 0.
  - Exploration then comes from the planner chasing its own model's errors, which learning
    corrects, plus 5% babbling. Arm B is the control that always babbles.
- Speed: the update takes about 56 ms and the planner about 41 ms per step, i.e. 10 steps/s per
  lifetime (17 without planning). Launched the 9 lifetimes at 20:35, 8 threads each.

- **First 20–30k steps** (21:51):
  - **L** (planner + memory): online hits per episode rose to 99% within 15–25k steps (e.g. seed 1:
    391 of 392 episodes in steps 15–20k, about 13 steps per episode). Test success of the planner at
    20k: 0.81–0.92 (median 7–12 steps).
  - **P** (no memory): online hits stay at 30–40% of episodes, below babbling. Test success at 20k:
    0.02–0.46.
  - **B** (online babbler with memory): online hits 45–55%, as random babbling. Its world model,
    used by the planner on the test episodes, already reaches 0.79–0.89 at 20k (median 17–36).
  - For comparison, vm-00002's batched babbler needed 131k environment steps, over 128 worlds, to
    support 0.87.

- **50k steps** (23:33):
  - **L:** online hits in steps 53–58k: 1,970 of 1,972 episodes, about 10 steps per episode. Test
    success at 50k: 0.87, 0.99, 0.99 and 0.98, median 6 steps in all four. This is the level of
    vm-00004 (1.00, median 5–6) and of PPO with reward (vm-00001: 1.00, median 5–7), from one world
    and 50k steps.
  - **P:** test success 0.08–0.24 at 50k. Online, 7–20 hits in 28–36 episodes in the last 5k steps.
  - **B** (at 109k steps; babbling is faster): test success at 50k is 0.71 (median 86) and 0.94
    (median 16). Seed 1 has fallen since 10k (0.84). At the same number of steps, learning from its
    own plans (L) beats learning from babbling, as in vm-00004.

### 2026-10-24
- **The Makefile asked for a 75k checkpoint that is never written** (checkpoints come every 10k
  steps). It now evaluates 10k, 20k, 30k, 50k, 100k, 150k and 200k.
- **100k steps** (04:30; L and P are at 146–150k, B finished 200k at 02:00):

  | Arm | test success at 100k (median steps) |
  |---|---|
  | L | 0.97 (5), 0.95 (6), 0.82 (7), 0.95 (5) |
  | P | 0.04, 0.08, 0.40 (all median 200) |
  | B | 0.74 (36), 0.80 (9) |

  - L holds its level; seed 3 dropped from 0.99 at 50k to 0.82.
  - Online, L hits in every episode, in 9–13 steps (steps 145–146k).
  - P is still below babbling.
- **150k and 200k steps** (all lifetimes done; L and P finished at about 06:35):

  | Arm | test success at 150k (median steps) | at 200k |
  |---|---|---|
  | L | 0.97 (5), 0.98 (5), 0.91 (6), 0.99 (5) | 0.98 (5), 0.82 (9), 0.83 (6), 0.92 (5) |
  | P | 0.39, 0.11, 0.30 (all median 200) | 0.26, 0.28, 0.56 (median 200, 200, 172.5) |
  | B | 0.99 (11), 1.00 (7) | 0.90 (10), 0.93 (8.5) |

  - **Online, L hits in every episode to the end:** all 436–543 episodes of steps 195–200k, in
    9–12 steps each.
  - **The test disagrees with online play.** L's test success moves by up to 0.17 between
    checkpoints (seed 2: 0.98 at 150k, 0.82 at 200k), while online play stays above 99.9%.
    - The test runs the planner alone.
    - Online, the agent babbles when the plan's imagined relief is under 0.02 per step, and on 5%
      of steps at random. The share of steps the planner acted on fell from 90–94% (first 5k steps)
      to 76–81% (last 5k).
  - **The imagination keeps improving in L** (`analyze_imagination.py`; results in the Results
    section).
  - **The linear-probe readout of the imagined cursor is unreliable.** For B seed 1 at 200k, it is
    off by 340 px at the first imagined step, where retrieval is off by 1.8 px. The probe misreads
    latents that leave the manifold of real frames, as the script's docstring warns. The Results
    quote retrieval only.
- **Why frozen checkpoints fall short of online play** (Results R4). New options, all off by
  default:
  - `plan.py --explore-gain/--explore-eps`: online.py's acting rule with frozen weights.
    `--env-seed`: other episodes. With these options off, the results are unchanged (identical
    per-episode outcomes).
  - `online.py --init CKPT`: a lifetime that starts from saved weights, with a fresh optimizer and an
    empty memory.

  Findings:
  - Neither the acting rule nor the test episodes explain the gap.
  - Frozen, the planner misses far starts.
  - Continuing from the 200k weights in the same loop: learning hits every episode; frozen (lr 0),
    it misses 1–19% of them, as in the test.

## Results

All runs reached 200k steps: L and P in about 9.7–9.8 hours (9–10 steps/s), B in 5.4 hours.

### R1. The planner on the 128 test episodes, at frozen lifetime checkpoints

| Run | 10k | 20k | 30k | 50k | 100k | 150k | 200k | mean 50k–200k |
|---|---|---|---|---|---|---|---|---|
| L s1 | 0.88 | 0.92 | 0.90 | 0.87 | 0.97 | 0.97 | 0.98 | 0.95 |
| L s2 | 0.77 | 0.81 | 0.99 | 0.99 | 0.95 | 0.98 | 0.82 | 0.94 |
| L s3 | 0.58 | 0.86 | 0.76 | 0.99 | 0.82 | 0.91 | 0.83 | 0.89 |
| L s4 | 0.80 | 0.89 | 0.98 | 0.98 | 0.95 | 0.99 | 0.92 | 0.96 |
| P s1 | 0.20 | 0.02 | 0.28 | 0.08 | 0.04 | 0.39 | 0.26 | 0.19 |
| P s2 | 0.17 | 0.15 | 0.11 | 0.15 | 0.08 | 0.11 | 0.28 | 0.15 |
| P s3 | 0.06 | 0.46 | 0.05 | 0.24 | 0.40 | 0.30 | 0.56 | 0.38 |
| B s1 | 0.84 | 0.79 | 0.67 | 0.71 | 0.74 | 0.99 | 0.90 | 0.84 |
| B s2 | 0.68 | 0.89 | 0.76 | 0.94 | 0.80 | 1.00 | 0.93 | 0.92 |

- **References:**
  - vm-00002's untrained model: 0.16.
  - vm-00002's batched babblers at 2.1M steps: 0.89 and 0.88 (0.41–0.91 from 131k steps on).
  - vm-00004 (the planner's own experience) and PPO with reward (vm-00001): 1.00, median 5–7
    steps.
- **Median steps:**
  - L from 50k on: 5–9.
  - P: 200 at every checkpoint (172.5 once).
  - B: 9–86 up to 100k, then 7–11.

### R2. Online play during the lifetime

Share of episodes with a hit (mean episode length in steps), range over seeds. An episode ends at
a hit or after 200 steps.

| Arm | 0–5k | 5–10k | 15–20k | 45–50k | 95–100k | 195–200k | whole lifetime |
|---|---|---|---|---|---|---|---|
| L | 0.42–0.91 (47–135) | 0.76–0.98 (30–72) | 0.93–1.00 (13–39) | 1.00 (10–11) | 1.00 (11) | 1.00 (9–12) | 0.994–0.998: 15,919–17,752 hits |
| P | 0.12–0.42 (151–190) | 0.19–0.47 (151–186) | 0.39–0.50 (132–162) | 0.22–0.54 (141–186) | 0.28–0.55 (150–169) | 0.44–0.64 (96–157) | 0.32–0.41: 401–547 hits |
| B | 0.39–0.53 (145–157) | 0.47–0.59 (136–149) | 0.34–0.38 (152–156) | 0.27–0.51 (136–168) | 0.40–0.44 (138–144) | 0.50–0.53 (135–146) | 0.47–0.48: 643–675 hits |

- **L's planner share:** L acted with the planner on 90–94% of its steps in the first 5k, falling
  to 76–81% in the last 5k. The other steps are babbled: by the gate when the plan's imagined
  relief is under 0.02 per step, or at random (5%).
- **P's planner share:** 45–56% at the end.

### R3. What the world model learned (`analyze_imagination.py`, frozen checkpoints)

**Imagined cursor along random action sequences.** Error at imagined step 12, in px, read out by
retrieval. Assuming the cursor never moves gives 8.5 px; the readout's own floor is 1.0 px.

| Arm | 50k | 100k | 200k |
|---|---|---|---|
| L | 7.3–10.8 | 4.1–6.4 | 3.1–4.1 |
| B | 2.5–3.7 | 2.0–2.5 | 2.0–2.8 |
| P | 10.4–22.1 | 13.4–24.7 | 18.0–25.6 |

**Imagined hunger** error at step 12 of the same sequences:

| Arm | 50k | 100k | 200k |
|---|---|---|---|
| L | 0.10–0.15 | 0.08–0.09 | 0.06–0.12 |
| B | 0.05 | 0.03–0.05 | 0.04 |
| P | 0.24–0.58 | 0.12–0.70 | 0.16–0.33 |

**Hunger relief (edge test) at 200k.** Imagined relief after one click with the cursor at a
signed distance g from the circle's edge: 1 means the hunger is gone, 0 means no change. The truth
is 1 inside the circle and 0 outside.

| | g −8 | −3 | −1 | +1 | +2 | +3 | +5 | +12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L (4 seeds) | .83–.94 | .84–.92 | .77–.85 | .62–.73 | .49–.62 | .28–.46 | .01–.05 | −.02–.00 |
| B (2 seeds) | .87–.88 | .55–.62 | .33–.37 | .16–.18 | .11–.13 | .05–.07 | .02 | .01 |
| vm-00002 babbler 005, 2.1M steps | .99 | .89 | .64 | .36 | .20 | .09 | .01 | .00 |

- **P** (3 seeds): −.04 to +.01 at every g, i.e. no relief anywhere.
- At 50k, B's relief was still weak: .36–.39 deep inside (g −8). L's was already .83–.99 at 50k.
- **Inverse dynamics R²** (online, last 5k steps): L 0.96, B 0.95, P 0.88–0.90.

### R4. Why frozen checkpoints fall short of online play

From 50k steps on, L misses only 3–5 of its 14–15k episodes online (none after 158k). Its frozen
checkpoints miss up to 18% of the test episodes. The checks below use the 200k checkpoints unless
noted (`make r4`, `make continue`).

1. **Not the acting rule.** Online, the agent babbles when the plan's imagined relief is under 0.02
   per step (the gate), and on 5% of steps at random. The same rule with frozen weights, on the
   test episodes:

   | Checkpoint | planner alone | gate only | 5% babbling only | online rule |
   |---|---|---|---|---|
   | L s2, 200k | 0.82 (0.84 with planner seed 1) | 0.86 | 0.85 | 0.87 |
   | L s3, 100k | 0.82 | 0.84 | 0.81 | 0.84 |
   | L s3, 200k | 0.83 (0.84) | 0.87 | 0.82 | 0.84 |

   Over all 12 L checkpoints at 100k, 150k and 200k, the online rule changes success by −0.02 to
   +0.06 (mean +0.01).
2. **Not the test episodes.** Online and test episodes come from the same distribution: circle,
   cursor, and hunger uniform in 0–0.5. On 128 fresh episodes (seeds 1–4):
   - frozen 200k weights alone: 0.97, 0.79, 0.85, 0.95;
   - with the online rule: 0.98, 0.82, 0.84, 0.95;
   - on the test episodes, for comparison: 0.98, 0.82, 0.83, 0.92.
3. **Frozen, the planner misses far starts** (`analyze_planner.py`, seeds 2 and 3):
   - **Hits by start distance from the edge:**
     - within 8 px: 19 of 19;
     - within 24 px: 85–87 of 94;
     - 24 px or more: 18 and 21 of 34.
   - **It never finds the circle.** In every miss, the cursor stays at least 9 px from the edge (a
     median closest approach of 22 px). vm-00002's babbler, by contrast, stopped 0–5 px short and
     clicked there.
   - **Some misses click over and over far from the circle**, up to 148 times, 30–57 px away.
   - **Imagined relief:** in misses, the best plan imagines a median relief of 0.52, against 1.00
     in hits.
4. **Learning during play closes the gap** (`make continue`). Each L lifetime continues from its
   200k weights in a new world for 5k steps, in the same online loop, in two ways:
   - learning as in the lifetime, but with a fresh optimizer and an empty memory;
   - frozen (lr 0), with the weights checked to stay bit-identical.

   | Seed | learning: hits / episodes (mean length) | frozen |
   |---|---|---|
   | 1 | 378/378 (13.1) | 437/442 (11.3) |
   | 2 | 385/385 (13.0) | 72/89 (56.2) |
   | 3 | 396/396 (12.5) | 91/110 (44.5) |
   | 4 | 360/360 (13.8) | 200/214 (23.0) |

   - **Learning hits every episode from its first 1,000 steps**, even without its memory and
     optimizer state.
   - **Frozen online play matches the frozen test:** 0.81–0.99 here, against 0.84–0.98 on the test
     episodes with the same rule.
   - **The weights never settle.** In the lifetimes, they move by 8% of their norm per 10k steps at
     190k–200k, and by 43–45% between 100k and 200k. With a fresh optimizer they move 13% in 5k
     steps.

## Analysis

1. **One lifetime is enough (hypotheses 1 and 2, confirmed).**
   - L's play turns from babbling into goal-directed clicking within 10–20k steps, with no phase
     switch and no reward. Online, it hits in 93–100% of the episodes in steps 15–20k.
   - From 50k to 200k, it misses only 3–5 of 14–15k episodes (none after 158k), hitting in 9–12
     steps. Over its whole life, it hit 99.4–99.8% of its episodes.
   - Both learners with memory, L and B, learn the action consequences:
     - inverse dynamics R² 0.95–0.96;
     - the imagined cursor 12 steps ahead within 2–4 px (assuming it stands still: 8.5 px);
     - the relief of a click (edge test).
2. **Learning on every step uses experience far more intensively than the batched babbler.**
   - **Planning:** B's model supports the planner at 0.68–0.89 after 10–20k steps of babbling.
     vm-00002's batched babbler needed 131k steps for 0.87.
   - **Relief:** B's imagined relief deep inside the circle reaches .87–.88 by 200k. The batched
     babbler had .15 at 262k steps and reached .74 only at 786k.
   - **A likely reason is reuse.** One update per step on 64 transitions uses each transition about
     64 times. The batched pipeline makes 128 updates on 512 transitions per 16,384 steps, so it
     uses each transition 4 times. That is 16 times more training per environment step.
3. **Memory is necessary (hypothesis 3, confirmed).**
   - **P learns how the cursor moves** (inverse dynamics R² 0.88–0.90), **but never what a click
     does:**
     - no imagined relief at any distance (−.04 to +.01);
     - imagined hunger off by 0.16–0.33;
     - the imagined cursor 18–26 px off at step 12, worse than assuming it stands still (8.5 px).
   - **It plays below random babbling all its life:** 32–41% of episodes, against 47–48%. Its
     planner tests stay at 0.02–0.56.
   - **Hits are rare in its own play,** one per 370–500 steps. Without a memory, each one is
     trained on only 8 times, in consecutive updates.
4. **Its own plans teach faster than babbling, and specialize the model** (as in vm-00004).
   - At 50k, L's planner scores 0.87–0.99, against B's 0.71 and 0.94. B's model catches up later
     (0.99 and 1.00 at 150k).
   - Along random actions, L's imagination is worse than B's: the cursor is off by 3.1–4.1 px
     against 2.0–2.8, and the hunger by 0.06–0.12 against 0.04.
   - L's imagined relief spreads outward from the edge: .49–.62 at 2 px outside, against .11–.13
     for B. vm-00004's planner-trained model had .52 there.
5. **A frozen snapshot understates the learning agent.**
   - After 50k, the planner test on frozen checkpoints swings between 0.82 and 0.99, while online L
     misses only 3–5 of 14–15k episodes.
   - Neither the acting rule nor the test episodes explain the gap (R4). Frozen, the planner misses
     1–19% of episodes. The misses are mostly far starts, where it never finds the circle or keeps
     clicking where nothing happens.
   - Learning during play closes the gap. From the same weights and in the same loop, learning hits
     every episode, even with a fresh optimizer and no memory. Frozen, it misses as in the test.
   - **A likely reading:** the model is never finished. Its weights move 8% per 10k steps, so every
     snapshot holds some errors. A learning agent corrects an error within steps of making it:
     a click that relieves nothing is in the next update. A frozen planner repeats the error until
     the episode times out.
   - **Consequence:** for this agent, success in its own life is the measure. A checkpoint test
     measures the model at one instant and bounds the agent from below.

## Conclusions

- **One agent in one world can learn the task with no reward and no phase switch.** Learning on
  every step from its own replayed experience, it learns to hit (nearly) every circle. From about
  50k steps on, it misses only 3–5 of 14–15k episodes and hits in 9–12 steps. Over its whole life it
  hit 99.4–99.8% of its 16–18k episodes.
- **Replay of its own past is necessary.** Without it, the agent learns how the cursor moves but
  never what a click does, and it plays below random babbling for all 200k steps.
- **Per environment step, online learning with replay is far more efficient than vm-00002's
  batched babbling.** It needs 10–20k steps instead of 131k for a model the planner can use, likely
  because each transition is used about 16 times more.
- **Frozen checkpoints understate a learning agent.** Frozen, the planner misses 1–19% of
  episodes, mostly far starts. Learning during play is what removes these misses.

## Next Steps

- [x] Memory gated by relief and surprise (vm-00007): storing relief speeds up L's early learning.
- [ ] Move the decision into the network: the planner as teacher (habit), from the topic README's
  open questions.
- [ ] Evaluate learning agents with learning on, e.g. `online.py --init` on a fixed set of episodes;
  frozen tests only bound them from below.
- [ ] Smaller targets (radius 4–8, vm-00003), where hits are rarer and memory should matter more.
