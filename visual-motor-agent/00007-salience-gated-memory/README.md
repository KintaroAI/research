# vm-00007: Salience-Gated Memory: Remember What Relieves Hunger or Surprises

**Date:** 2026-10-24
**Status:** Complete
**Source:** `exp/vm-00007` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00007/`
**Task plan:** [`../tasks/00007-salience-gated-memory.md`](../tasks/00007-salience-gated-memory.md)

## Goal

Until now, every step entered memory and replay drew from it uniformly. Here, experience enters
long-term memory with a probability that depends on what happened, as the hippocampus seems to do:
- hunger relief and surprise make storage likely;
- nothing happening makes it unlikely.

The questions:
- Does this let the basic agent (vm-00006) make goal-directed moves at last?
- How does it change the planner agent (vm-00005 L)?

The losses stay reward-free. Surprise only decides what is stored; the model never sees it.

## Hypothesis

1. **Basic agent.** Storing relief raises the share of hindsight-labelled steps in memory from
   about 3% to 10–20%. Its hunger-conditioned moves then beat the random initial model (test
   success 0.44–0.48) for the first time. Online, it hits in more than half of the episodes, the
   level of random babbling.
2. **Surprise** also keeps hits while they are unexpected, plus the failed clicks; alone, it enriches
   relief less. In the basic agent, its move part keeps the unexpected moves, mostly babbled ones.
   This may slow the narrowing of its moves (vm-00006: the std of dx fell from 0.60 to 0.40, and to
   0.24 with replay).
3. **Planner agent.**
   - Relief alone makes the hunger head optimistic at the circle's edge. Adding surprise removes
     this.
   - Gating may speed up early learning.

## Method

- **Memory** (`online.memory=gated` in `dev/online.py`):
  - **Short-term buffer:** the last 256 steps. It provides the acting window and the newest training
    sequence.
  - **Long-term memory:** 100k steps, kept as chunks of 8 consecutive steps. Each chunk is decided 11
    steps after it ends, once it is known whether relief follows:
    - **relief:** a hit in the chunk or in the next 11 steps of the same episode → stored;
    - **surprise:** in the top 10% of the last 1,000 chunks by surprise → stored;
    - **otherwise:** stored with probability 0.1;
    - **warm-up:** the first 1,000 steps are stored in full, in every arm.
  - **Surprise of a step:** how wrong the model's prediction of the next state was. It is measured
    right after the step, before the model learns from it.
    - **frame:** squared error, with changed pixels weighted 11× as in training;
    - **drives:** squared error;
    - **move** (basic agent only): 0.5 Σ ((a − μ)/σ)², plus −log p of the click outcome;
    - **combined:** each part is divided by its mean over the last 1,000 steps, and the parts are
      summed. A chunk's surprise is the largest of its steps.
  - **Training:** each update uses the newest 8 steps plus 7 chunks drawn uniformly from long-term
    memory.
- **Agents**, each in its own world, with seeds 1–2:
  - **Basic agent** (`configs/AHR_gated.json`, as vm-00006 A+H+R):
    - a window of 8 real states;
    - predicts the next move, frame and drives;
    - hindsight hunger input; 20% babbled moves; batch 8;
    - 50k steps, 4 threads.
  - **Planner agent** (`configs/L_gated.json`, as vm-00005 L):
    - one-frame model;
    - CEM planner on the latent dynamics; 5% babbled moves;
    - 30k steps, 4 threads (vm-00005 used 8).
- **Arms:**

  | Arm | Relief chunks | Top-10% surprise chunks | Other chunks | Agents |
  |---|---|---|---|---|
  | U (control) | stored | stored | stored | basic, planner |
  | R | stored | p = 0.1 | p = 0.1 | basic, planner |
  | S | p = 0.1 | stored | p = 0.1 | basic |
  | RS | stored | stored | p = 0.1 | basic, planner |
  | U10 (size control, added at 02:00) | p = 0.1 | p = 0.1 | p = 0.1 | basic |
  | I+U, I+R (inverse dynamics, added at 03:32) | as U, R | as U, R | as U, R | basic |

- **Evaluation:**
  - **Basic agent:** its own moves on vm-00002's 128 test episodes, every 10k steps
    (`online.py --eval`).
  - **Planner agent:**
    - `plan.py` at 10k, 20k and 30k;
    - `analyze_imagination.py` at 30k (edge test, false relief).
  - **Online, all arms:** hits, episode lengths, and memory composition (stored chunks by reason,
    share of relief chunks, share of labelled steps).

## Log

### 2026-10-24
- **Implemented** the gated memory in `online.py`: `GatedMemory`, `surprise_parts`, and hindsight
  labels that stop at the episode's end.
- **The default (`memory="stream"`) still reproduces vm-00005/6.** 300-step runs of L_online, A_basic
  and A+H+R give byte-identical checkpoints before and after the change.
- **Tests:** 9 new ones in `tests/test_online_memory.py`, covering:
  - labels and relief chunks;
  - order when storing everything;
  - gate rates and the surprise top fraction;
  - the combined surprise;
  - downsampling vs the reference;
  - surprise parts vs a direct computation;
  - gated smoke runs of both actors.
  All 290 Python tests pass.
- **Launch:**
  - The 8 basic-agent runs started at 00:39.
  - The planner runs start when vm-00005 B and vm-00006 A+H+R finish (about 02:15), which keeps the
    machine at about 112 threads.
  - There were no separate smoke runs: the first 3,000 steps of these runs serve as the smoke test.
- **First 3,000 steps of the 8 basic runs** (00:48):
  - **Health:** about 6 steps/s (50k steps in about 2.3 hours); losses fall; no NaNs.
  - **The gate works as intended:**
    - the warm-up stores all 123 chunks of the first 1,000 steps;
    - U stores all 125 chunks of every 1,000 steps;
    - R stores the relief chunks plus 7–10% of the rest. In seed 2, 3 hits gave 9 relief chunks.
      Its long-term memory then holds 9% relief chunks and 4.7% labelled steps, against 2% and 1.2%
      in U.
  - **S stores more than planned:** 19–28% of the chunks by surprise, instead of 10%.
    - The threshold comes from the last 1,000 chunks, which still include the fast-learning warm-up.
      It rose from 5.6–6.7 to 8.7–9.4.
    - It should settle as the window rolls over, after about 8,000 steps.
  - **What makes a chunk surprising: mostly the move.** Of the chunks S stored by surprise:
    - about 70% were flagged by the move part (76 of 107): babbled moves and unlikely clicks;
    - about 30% by the drives part (30 of 107): unexpected hunger changes;
    - only one by the frame part.
  - **Identical arms so far.** RS equals S exactly until the first relief chunk (same seed, same
    draws), and R equals U through the warm-up.
- **At 27k steps (01:55), the gated arms narrow their moves faster than U**, against hypothesis 2:

  | Arm | long-term memory (chunks) | std of dx | move NLL | test success at 20k |
  |---|---|---|---|---|
  | U | 3,373 | 0.20–0.21 | −0.20 | 0.22, 0.16 |
  | R | 500–565 | 0.14 | −0.87, −0.89 | 0.14, 0.20 |
  | S | 832–862 | 0.16 | −0.69, −0.65 | 0.14, 0.19 |
  | RS | 841–896 | 0.16 | −0.68, −0.66 | 0.17, 0.16 |

  - The gated memories are 4–7× smaller, and every update still draws 7 chunks from them. So each
    stored chunk is replayed 4–7× as often as in U. The lower move NLL suggests the model memorizes
    the stored moves.
  - This confounds what is stored with how much is stored. **Added the size control U10**
    (`make control`, seeds 1–2, runs 015–016, from 01:58): it stores every chunk with probability
    0.1, the base rate of the gated arms, with no relief or surprise rule. Each gated arm is U10
    plus its salient chunks.
  - The relief arm already has the planned enrichment: 11–21% of its long-term chunks are relief
    chunks and 7–13% of its steps are labelled, against 2% and 1.2% in U. Its test success does not
    rise with it so far.
- **The 6 planner-agent runs started at 02:00**, when vm-00005 B and vm-00006 finished.
- **First 3,000 steps of the planner runs** (02:16):
  - **Health:** 3.1 steps/s with 4 threads (30k steps in about 2.7 hours); losses fall; no NaNs.
  - **The relief rule stores much more here.** The planner hits often, so many chunks are relief
    chunks: R and RS store 26–43% of the chunks from 1k to 3k steps. Their long-term memory then
    holds 29–48% relief chunks, against 13% in U.
  - **Gating speeds up early learning in seed 1** (hits / episodes):

    | Arm | 1k–2k, s1 | 2k–3k, s1 | 1k–2k, s2 | 2k–3k, s2 |
    |---|---|---|---|---|
    | U | 7/11 | 8/12 | 10/12 | 14/15 |
    | R | 21/23 | 35/37 | 12/12 | 12/14 |
    | RS | 22/23 | 28/29 | 13/13 | 20/21 |

    Seed 2 learns fast in all arms.
- **Diagnostics of the basic agent at 30k steps** (the 128 test episodes, scored with the ground
  truth; the vm-00006 checks of 02:00–02:30):
  - **No arm aims.** For steps with the cursor outside the circle and at least 10 px from every wall,
    the cosine between the predicted mean move and the direction to the circle is −0.05 to +0.06 in
    all 8 runs. At birth it is +0.02 and −0.01.
  - **The cursor sits at a wall in 47–69% of the test steps**, against 23% at birth.
  - **The command matters little.** Asking for no relief instead of relief changes that cosine by
    at most 0.02.
  - **The encoder features stay healthy** in U, R and S seed 1 (norm 3.7–3.8, 127–128 of 128
    active), unlike in the vm-00006 runs without replay.
- **The relief chunks do carry an aiming signal** (03:25; a simulation of random babbling in 512
  worlds for 4,000 steps, scored with the ground truth):
  - **Labelled steps** (a hit within the next 12 steps of the same episode, 4.0% of the steps;
    scored where the cursor is outside the circle): the move points toward the circle, with a mean
    cosine of +0.34. That is +0.61 one step before the hit and +0.20 eleven steps before.
  - **Other steps:** −0.01.
  - So a model that imitated the labelled moves when asked for relief would aim clearly. The agents'
    aim cosine is at most +0.04, with or without the command.
- **The agents' own relief steps carry the same signal** (R seed 2 and U seed 1 at 50k, replayed
  as they live online in 256 worlds for 1,000 steps):
  - labelled steps are 2.5–2.6% of the steps; about 20% of them are babbled;
  - in labelled steps, own moves point toward the circle with a cosine of +0.35 to +0.39, and
    babbled moves with +0.35 to +0.38;
  - in the other steps, own moves point away from it (−0.10 to −0.13), and babbled ones are random
    (−0.01).
- **The basic agent's representation does not hold the direction to the circle.** Linear (ridge)
  probes on 4,096 random states, R² on 1,024 held-out ones:

  | Model | z: cursor | z: circle | z: direction | v: cursor | v: circle | v: direction |
  |---|---|---|---|---|---|---|
  | birth (random weights) | 0.06 | 0.95 | 0.24 | 0.06 | 0.94 | 0.24 |
  | basic agent at 50k: U s1, R s2, S s2 | 0.29–0.33 | 0.11–0.18 | 0.21–0.26 | 0.46–0.47 | 0.28–0.39 | 0.35–0.40 |
  | vm-00006 A+H+R s1 at 100k | 0.33 | 0.10 | 0.22 | 0.50 | 0.24 | 0.36 |
  | planner agent (vm-00005 L s1) at 10k | 0.95 | 0.56 | 0.79 | 0.97 | 0.62 | 0.80 |
  | planner agent (vm-00005 L s1) at 100k | 0.95 | 0.54 | 0.84 | 0.96 | 0.55 | 0.83 |

  - z is the Transformer output that the linear policy head reads; v holds the visual features.
  - The basic agent even loses the circle position that the random encoder provides.
  - The planner agent's encoder is also trained by the inverse-dynamics head. That head predicts
    the executed move from consecutive v, without detaching them. The basic agent has no such head,
    and its frame decoder reads the encoder maps through skip connections. So nothing forces the
    cursor into v or z.
- **Added the arms I+U and I+R** (`make inverse`, runs 017–020, from 03:32, 50k steps): the basic
  agent with the inverse-dynamics head (`model.inv_hidden=128`, `ppo.inv_coef=0.1`, as in L), under
  U and R memory, seeds 1–2.
  - The question: with the cursor in its features, does the hindsight-conditioned agent aim? And
    does the relief gate then help?
  - A 1,000-step smoke run: inverse-dynamics R² 0.09 → 0.41, 6.5 steps/s.
- **The inverse-dynamics arms at 10k–27k steps** (04:45):
  - **Their features hold more of the cursor.** R² from z at 10k: cursor 0.49–0.65, direction
    0.38–0.50, against 0.29–0.33 and 0.21–0.26 without the head (50k). The planner agent reaches
    0.95 and 0.79 by 10k.
  - **The head learns less well than in the planner agent.** Its R² on the training batches levels
    off at 0.48–0.53 in I+U and 0.77–0.81 in I+R, against 0.96 in the planner agent.
    - I+R's smaller memory is replayed more often, which may inflate its R².
    - A possible cause: the basic agent's own moves are narrow (dx std 0.13–0.27), so there is less
      motion to learn from.
  - **Test success:** 0.13, 0.13 (I+U) and 0.20, 0.15 (I+R) at 10k; 0.36, 0.21 and 0.11, 0.09 at 20k.
- **All runs finished by 05:36.** Results below.
- **Follow-up after the tag (from 11:07, same code; `make planner-control`, runs 021–032):** the
  planner agent's missing size controls, with 4 seeds per arm:
  - U10: every chunk with probability 0.1; U50: with probability 0.5. Seeds 1–4 each;
  - plus seeds 3–4 of U and R.
  - In the planner agent, most of R's chunks are relief chunks, so its memory is 40–60% of U's. U50
    matches that size; U10 is much smaller.
- **First 10k steps of the follow-up** (12:18; online hits, seeds 1–4):

  | Arm | hits in steps 0–10k | mean | long-term memory at 10k (chunks) |
  |---|---|---|---|
  | U | 119, 136, 101, 73 | 107 | 1,248 |
  | R | 293, 189, 114, 147 | 186 | 407–626 |
  | U50 | 119, 151, 83, 78 | 108 | 669–698 |
  | U10 | 99, 110, 44, 43 | 74 | 228–247 |

  - **Content, not size:** at about R's size, a random memory learns no faster than storing everything
    (U50 ≈ U). R beats it in every seed.
  - A much smaller random memory (U10) learns more slowly.

## Results

### Basic agent (50k steps)

**The gate filled the memory as planned.** At 50k steps, seeds 1, 2:

| Arm | long-term memory (chunks) | relief chunks | labelled steps |
|---|---|---|---|
| U | 6,248 | 2.9%, 2.4% | 1.8%, 1.5% |
| R | 876, 964 | 14%, 22% | 9.2%, 14.2% |
| S | 1,415, 1,436 | 5.3%, 6.4% | 2.9%, 3.4% |
| RS | 1,430, 1,506 | 7.1%, 10.8% | 4.7%, 6.8% |
| U10 | 762, 770 | 1.0%, 2.2% | 0.8%, 1.3% |

- In S and RS, the surprise rule stores 11–12% of the chunks, close to the planned 10%.
- 83–91% of those chunks are flagged by the move part.

**No arm beats the random initial model.** Test success of the agent's own moves:

| Arm | birth | 10k | 20k | 30k | 40k | 50k | mean, 10k–50k |
|---|---|---|---|---|---|---|---|
| U | 0.48, 0.44 | 0.15, 0.11 | 0.22, 0.16 | 0.26, 0.20 | 0.22, 0.17 | 0.29, 0.16 | 0.19 |
| R | 0.48, 0.44 | 0.05, 0.23 | 0.14, 0.20 | 0.13, 0.19 | 0.17, 0.25 | 0.13, 0.25 | 0.17 |
| S | 0.48, 0.44 | 0.13, 0.13 | 0.14, 0.19 | 0.16, 0.30 | 0.20, 0.27 | 0.20, 0.29 | 0.20 |
| RS | 0.48, 0.44 | 0.09, 0.26 | 0.17, 0.16 | 0.16, 0.27 | 0.22, 0.20 | 0.16, 0.22 | 0.19 |
| U10 | 0.48, 0.44 | 0.05, 0.27 | 0.06, 0.23 | 0.05, 0.20 | 0.12, 0.24 | 0.10, 0.17 | 0.15 |
| vm-00006 A+H+R (stream) | 0.48, 0.44 | 0.02, 0.23 | 0.04, 0.26 | 0.11, 0.20 | 0.16, 0.25 | 0.19, 0.25 | 0.17 |

With 128 episodes, one success rate has a standard error of about 0.035. The seeds differ by up to
0.18.

**Behaviour at 50k:**

| Arm | online hits, last 5k steps | std of dx | click probability | test clicks per episode | aim cosine | cursor at a wall |
|---|---|---|---|---|---|---|
| U | 10/32, 7/29 | 0.20 | 0.10–0.12 | 28, 20 | +0.00, −0.02 | 51%, 51% |
| R | 7/29, 10/33 | 0.13 | 0.05–0.07 | 4, 12 | +0.02, +0.01 | 61%, 56% |
| S | 11/32, 4/28 | 0.16 | 0.07–0.09 | 12, 24 | +0.02, +0.04 | 41%, 56% |
| RS | 10/31, 8/31 | 0.15–0.16 | 0.06–0.07 | 13, 9 | −0.05, +0.04 | 54%, 57% |
| U10 | 7/29, 6/30 | 0.12–0.13 | 0.05–0.13 | 7, 16 | −0.01, −0.02 | 38%, 54% |
| birth | — | 0.6 | 0.10 | 15 | +0.02, −0.01 | 23% |

- **Online hits:** random babbling hits in 0.47–0.48 of its episodes (vm-00005 B). All arms hit in
  14–34% of theirs.
- **Aim cosine:** taken where the cursor is outside the circle and at least 10 px from every wall.
  It is the mean cosine between the predicted mean move and the direction to the circle.
  - Asking for no relief instead changes it by at most 0.017.
- **Cursor at a wall:** the share of test steps within 2 px of a wall.
- **The move NLL** is −1.06 in R, −1.06 to −1.15 in U10, −0.69 to −0.76 in S and RS, and −0.20 to
  −0.25 in U.
- **The world model is the same in every arm:** frame error 0.0007–0.0010, drive error 0.0032–0.0045.

### Basic agent with inverse dynamics (50k steps)

**The inverse-dynamics head lifts the cursor into the features early, but the gain fades and the
agent still does not aim.** Seeds 1, 2:

| Arm | 10k | 20k | 30k | 40k | 50k | mean, 10k–50k | aim cosine at 50k | cursor at a wall |
|---|---|---|---|---|---|---|---|---|
| U | 0.15, 0.11 | 0.22, 0.16 | 0.26, 0.20 | 0.22, 0.17 | 0.29, 0.16 | 0.19 | +0.00, −0.02 | 51%, 51% |
| I+U | 0.12, 0.12 | 0.36, 0.21 | 0.34, 0.15 | 0.42, 0.21 | 0.41, 0.16 | 0.25 | +0.02, −0.05 | 43%, 53% |
| R | 0.05, 0.23 | 0.14, 0.20 | 0.13, 0.19 | 0.17, 0.25 | 0.13, 0.25 | 0.17 | +0.02, +0.01 | 61%, 56% |
| I+R | 0.20, 0.15 | 0.11, 0.09 | 0.18, 0.27 | 0.23, 0.34 | 0.16, 0.25 | 0.20 | +0.05, −0.02 | 75%, 36% |

- **I+U seed 1 is the best basic agent so far:** 0.41–0.42 at 40–50k, close to birth (0.48). Its
  seed 2 stays at 0.15–0.21.
- **I+R's memory** holds 17–20% relief chunks and 11–13% labelled steps, as in R. Its online hits
  are 6–7 in 30 episodes (I+U: 11/31, 6/28).
- **Asking for no relief** changes the aim cosine by at most 0.028.
- **Feature content peaks early and then decays** (R², seed 1):

  | Run | z: direction at 10k / 30k / 50k | v: cursor at 10k / 30k / 50k |
  |---|---|---|
  | U | 0.38 / 0.28 / 0.21 | 0.62 / 0.53 / 0.46 |
  | I+U | 0.48 / 0.35 / 0.30 | 0.67 / 0.57 / 0.52 |

  - At 50k, the 4 inverse-dynamics runs have z direction R² 0.16–0.30 and z cursor R² 0.22–0.41.
  - The planner agent keeps 0.84 and 0.95 at 100k.

### Planner agent (30k steps)

**Storing relief speeds up early learning.** Planner test success (median steps), seeds 1, 2:

| Arm | 10k | 20k | 30k | hits in steps 0–10k | hits in steps 0–30k |
|---|---|---|---|---|---|
| U | 0.70 (16), 0.93 (10) | 0.80 (12), 0.84 (10) | 0.92 (8), 0.95 (8) | 119, 136 | 892, 1,341 |
| R | 0.90 (14), 0.95 (12) | 0.93 (7), 0.92 (8) | 0.92 (7), 0.95 (8) | 293, 189 | 1,210, 1,590 |
| RS | 0.92 (13), 0.59 (85) | 0.98 (8), 0.85 (12) | 0.96 (7), 0.95 (10) | 191, 111 | 1,329, 607 |
| vm-00005 L (stream, seeds 1–4, 8 threads) | 0.88, 0.77, 0.58, 0.81 | 0.92, 0.81, 0.86, 0.89 | 0.90, 0.99, 0.76, 0.98 | 187, 69, 96, 218 | 1,576, 574, 1,024, 1,401 |

- **Memory:** U holds 3,748 chunks, 32–46% of them relief chunks. R holds 1,902–2,252 chunks, with
  84–88% relief chunks. RS holds 1,383–2,034, with 67–84%.
- **RS is almost R.** Its surprise rule adds at most 1% of the chunks. About 10% of the chunks are
  in the top 10% by surprise, and a chunk counts under relief first, so at least 90% of the
  surprising chunks lead to relief. The few others are flagged by the drives part. RS seed 2 lagged
  early (0.59 at 10k) and caught up.
- **The world model ends up the same in every arm** at 30k:
  - frame error 0.0011–0.0015;
  - inverse-dynamics R² 0.96–0.97;
  - one-step error of the latent hunger head 0.005–0.007.

**Relief storage does not make the hunger head optimistic.** Imagined relief for a click at the
circle's edge (`analyze_imagination.py`, 30k):

| Arm | inside, 1–3 px | outside, 1–3 px | outside, 5–8 px | false relief |
|---|---|---|---|---|
| U | 0.64, 0.70 | 0.36, 0.36 | 0.07, 0.12 | 0.052, 0.149 |
| R | 0.66, 0.83 | 0.36, 0.48 | 0.06, 0.07 | 0.062, 0.037 |
| RS | 0.81, 0.58 | 0.47, 0.31 | 0.07, 0.08 | 0.030, 0.071 |
| vm-00005 L at 50k | 0.90, 0.73 | 0.55, 0.43 | 0.04, 0.10 | 0.015, 0.245 |

Relative to inside, R's imagined relief outside is the same as U's (0.55–0.58 against 0.51–0.56).

## Analysis

### Basic agent: the gate changes the memory, not the moves

- **Hypothesis 1 is rejected.** The relief rule reached the planned enrichment: 9–14% labelled
  steps, against 1.5–1.8% in U. But no arm moves toward the circle:
  - test success stays below birth in every arm (means 0.15–0.20, against 0.44–0.48);
  - online, the agents hit in 14–34% of episodes, below random babbling (47–48%);
  - the aim cosine is −0.05 to +0.04.
- **It is not the data.** In the agents' own experience, the moves of the 12 steps before a hit
  point at the circle (cosine +0.35 to +0.39). The other moves point away from it (−0.10 to
  −0.13). The memory holds a clear lesson.
- **It is the representation.** A linear policy head on z can only aim if z holds where the circle
  lies relative to the cursor.
  - The basic agent's z does not: the direction is linearly readable with R² 0.21–0.26. The cursor
    is at 0.29–0.33, and the circle at 0.11–0.18, worse than the random encoder's 0.95.
  - The planner agent's z does: 0.84 for the direction and 0.95 for the cursor. Its encoder is
    also trained by the inverse-dynamics head.
  - Without such features, the command has nothing to act on: asking for no relief changes the aim
    cosine by at most 0.017.
- **The basic agent learns the cursor early and then loses it.**
  - U seed 1: the cursor's R² from v falls from 0.62 at 10k to 0.46 at 50k, and the direction's R²
    from z from 0.38 to 0.21.
  - The inverse-dynamics head adds about 0.1 at every point but does not stop the decline, and the
    agent does not aim with it either (I+U, I+R).
  - The planner agent keeps 0.95 for the cursor. Its encoder is trained only by frame, drive and
    inverse-dynamics prediction (its latent heads do not reach the encoder), and it has no move
    loss.
  - A likely cause is the move loss, but this is untested:
    - it trains the shared features to reproduce the agent's own sampled moves;
    - per sample, its gradient grows about as 1/σ as the moves narrow, and σ falls from 0.6 to
      0.13–0.20.
- **Below birth because of habits.** The trained agents' mean moves are large and depend on the
  state, but not on the previous move. In 38–61% of the test steps the cursor sits at a wall,
  against 23% at birth. Self-imitation keeps whatever field of moves the agent has; where that field
  points into a wall, the cursor stays, and those states fill the memory.

### Memory size, not content, narrows the moves

- **Hypothesis 2 is rejected.** S narrows faster than U, not slower: a dx std of 0.16 against 0.20.
- **U10 shows why.** It has the base rate of the gated arms and uniform content:
  - it narrows the most (0.12–0.13) and memorizes the most (move NLL −1.06 to −1.15);
  - every update still draws 7 chunks, so in a memory 4–8× smaller, each chunk is replayed 4–8× as
    often as in U;
  - the salient chunks of S and RS roughly double the memory against U10, and those arms narrow
    less.
- **What is surprising in the basic agent: mostly the move part** (83–91% of the chunks stored for
  surprise), i.e. babbled moves and unlikely clicks. Frame errors rarely stand out.
- **Content may matter a little at equal size.** U10 has the lowest test success (mean 0.15), the
  gated arms 0.17–0.20, and U 0.19. This is within the noise: seeds differ by up to 0.18.

### Planner agent: relief storage speeds up learning, without optimism

- **Hypothesis 3, second part, holds for R in both seeds.**
  - In the first 10k steps, R has 1.4–2.5× as many hits as U.
  - Its test success is 0.90 and 0.95 at 10k (U: 0.70, 0.93), and 0.93 and 0.92 at 20k (U: 0.80,
    0.84).
  - A likely reading: the world model trains more on the chunks that end in a hit, i.e. on approach
    and click, the planner's most useful experience.
- **The arms converge by 30k** (0.92–0.96). By then:
  - the planner hits every 13–26 steps;
  - 32–54% of all chunks are relief chunks;
  - every arm's world model has learned approach and click.
- **Hypothesis 3, first part, is rejected: relief storage does not make the hunger head optimistic.**
  - Imagined relief 1–3 px outside the circle, relative to inside, is 0.55–0.58 in R against
    0.51–0.56 in U. False relief is 0.04–0.06 against 0.05–0.15.
  - A likely reason: the relief chunks also hold the failed clicks. The planner clicks on about
    half its steps (6–12 clicks in episodes of 13–26 steps at 30k), so the 12 steps before a hit
    contain several misses.
- **Surprise adds nothing beyond relief here.** At least 90% of the surprising chunks lead to relief
  anyway. The rest are flagged by drive errors, probably relief that was expected and did not come.

### Caveats

- 2 seeds per arm. Seeds differ by up to 0.18 in the basic agent and by up to 0.33 in the planner
  agent at 10k.
- The planner arms used 4 threads (vm-00005: 8). U also differs from vm-00005 L in how it draws its
  training sequences (whole chunks), so the two are not bitwise comparable.
- **The planner arms had no equal-size control.** At 30k, R's memory holds 1,902–2,252 chunks against
  U's 3,748, so each stored chunk is replayed 1.7–2.0× as often. The follow-up below adds it: the speedup
  comes from what R stores, not from its size.
- No importance weights. The gated memories are biased by design, and the bias is measured, not
  corrected.

## Conclusions

1. **Planner agent:** remembering what came before relief speeds up early learning, in both seeds,
   and costs nothing later. It does not bias the world model toward optimism. With 4 seeds and
   equal-size controls (follow-up), this comes from what it stores: a random memory of its size learns
   more slowly than storing everything.
2. **Basic agent:** salience-gated memory does not create goal-directed moves, with or without the
   inverse-dynamics head.
   - The bottleneck is the representation, not what is remembered. The memory holds a clear aiming
     signal (+0.35 to +0.39), but the features the policy reads lose the cursor during training.
   - The best basic agent (I+U seed 1, 0.41–0.42) only approaches the random initial model.
3. **A smaller memory means more replay per stored step.** Gates must be compared at equal memory
   size (U10).
4. **Surprise over the whole state adds little beyond relief.**
   - In the basic agent it is dominated by the move part.
   - In the planner agent, the surprising chunks are mostly relief chunks.

## Next Steps

- **Protect the basic agent's features from the move loss:** stop its gradient at the policy head,
  so the policy reads features that only prediction and inverse dynamics train (as in the planner
  agent). Then re-test the hindsight command and the relief gate.
- **Planner as teacher** (the topic README's open question): L learns to predict the planner's
  moves.
- **The feature blow-up without replay** (vm-00006): weight decay or a normalized v, then re-run
  A against A+H as a real test of the hindsight input.
- **Salience as replay priority instead of storage:** keep everything and replay salient chunks
  more. This keeps the memory size equal across arms. (The planner agent's equal-size controls: see the
  follow-up.)

## Follow-up (after the tag): equal-size memory controls for the planner agent

Same code as the tag (`make planner-control`, runs 021–032, 4 threads, 30k steps; `make
planner-control-analysis`). In the planner agent most of R's chunks are relief chunks, so its memory is
40–60% of U's. Two uniform controls store every chunk with a fixed probability: U50 (0.5, about R's size)
and U10 (0.1, the gated arms' base rate). Seeds 1–4 for both, plus seeds 3–4 of U and R.

**Online hits** (sum of 4 seeds; per seed in steps 0–10k):

| Arm | memory at 30k (chunks) | relief chunks at 30k | 0–10k | 10k–20k | 20k–30k | 0–10k by seed |
|---|---|---|---|---|---|---|
| R | 1,700–2,252 | 79–88% | **743** | **1,931** | **2,304** | 293, 189, 114, 147 |
| U | 3,748 | 27–46% | 429 | 1,424 | 2,218 | 119, 136, 101, 73 |
| U50 | 1,904–1,967 | 16–36% | 431 | 877 | 1,278 | 119, 151, 83, 78 |
| U10 | 484–519 | 6–24% | 296 | 537 | 748 | 99, 110, 44, 43 |

**Planner test success** (frozen checkpoints, 128 test episodes; seeds 1–4, mean):

| Arm | 10k | 20k | 30k |
|---|---|---|---|
| R | 0.90, 0.95, 0.65, 0.88 (0.84) | 0.93, 0.92, 0.86, 0.96 (0.92) | 0.92, 0.95, 0.89, 0.96 (0.93) |
| U | 0.70, 0.93, 0.80, 0.69 (0.78) | 0.80, 0.84, 0.98, 0.98 (0.90) | 0.92, 0.95, 0.88, 0.96 (0.93) |
| U50 | 0.46, 0.84, 0.60, 0.53 (0.61) | 0.85, 0.76, 0.75, 0.74 (0.78) | 0.78, 0.92, 0.92, 0.72 (0.84) |
| U10 | 0.42, 0.80, 0.55, 0.53 (0.57) | 0.69, 0.75, 0.48, 0.47 (0.60) | 0.58, 0.87, 0.90, 0.45 (0.70) |

**Imagined relief for a click at the circle's edge** (30k; inside 1–3 px, outside 1–3 px, outside 5–8 px,
false relief; seeds 1–4):

| Arm | inside | outside 1–3 px | outside 5–8 px | false relief |
|---|---|---|---|---|
| R | 0.51–0.83 | 0.27–0.48 | 0.06–0.11 | 0.037–0.072 |
| U | 0.48–0.70 | 0.24–0.36 | 0.04–0.12 | 0.022–0.149 |
| U50 | 0.18–0.59 | 0.12–0.27 | 0.04–0.06 | 0.000–0.038 |
| U10 | 0.02–0.32 | 0.02–0.20 | 0.01–0.13 | 0.003–0.101 |

- **Content, not size.** At R's memory size, a random memory (U50) learns more slowly than storing
  everything: after the first 10k steps it gets 877 hits against U's 1,424, and its planner tests trail
  (0.61, 0.78, 0.84 against 0.78, 0.90, 0.93). R, just as small, is the fastest. So R's speedup comes from
  storing the chunks that lead to relief.
- **A smaller random memory hurts.** U10 is the slowest on every measure. Replaying fewer chunks more
  often does not help the planner agent.
- **Online, R leads in all 4 seeds** in the first 10k steps. On the frozen test its lead at 10k is small
  (0.84 against 0.78) and holds in 2 of 4 seeds: frozen checkpoints understate online learners
  (vm-00005).
- **Still no optimism.** R's imagined relief just outside the circle, relative to inside, is 0.53–0.60,
  against 0.46–0.56 for U. The smaller random memories imagine less relief everywhere: they have seen
  fewer relief chunks.

