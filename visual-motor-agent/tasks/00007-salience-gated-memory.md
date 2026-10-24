# Task 00007: Salience-Gated Memory: Remember What Relieves Hunger or Surprises

**Date:** 2026-10-24
**Status:** Complete

## Context

- **So far, every step enters memory**, and replay draws from it uniformly (vm-00005, vm-00006).
- **The basic agent is limited by rare relief.** With the hindsight hunger input and replay (vm-00006
  A+H+R), about 2 hits happen per 1,000 steps, each labelling up to 12 steps. So only about 3% of
  the stored steps lead up to a hit. Its own moves stay below the random initial model: test success
  0.16–0.25 at 40k steps, against 0.44–0.48 at birth.
- **The owner's idea:** store experience the way the hippocampus seems to, with a probability that
  depends on what happened:
  - hunger relief: likely stored; nothing happening: rarely stored;
  - high surprise, a large error in the prediction of the whole next state (frame, internal state
    and move): likely stored.
  Surprise only decides what is written. The model never sees it.
- **Precedents:**
  - In the brain, novelty and dopamine gate what enters long-term memory (Lisman & Grace 2005), and
    replay is stronger after rewarded runs (Singer & Frank 2009).
  - Prioritized experience replay (Schaul et al. 2016) replays transitions with large errors more
    often, with importance weights against the bias.
  - Selective storage for lifelong RL (Isele & Cosgun 2018) uses a short-term FIFO buffer plus a
    long-term memory with a storage rule. Keeping a representative sample forgot least.
- **The two signals do different jobs.**
  - **Relief** is a built-in tag that works from birth. It selects on the outcome, so a memory heavy
    in relief can make the hunger head optimistic: it may imagine relief just outside the circle,
    the vm-00002 failure.
  - **Surprise** flags where the model's prediction of the next state was wrong, in any part: the
    frame, the drives or the move. That includes the failed clicks that fixed the planner in
    vm-00004 (relief was expected and did not come). Hits stop being favored once they are
    predicted.
- **The world has no unpredictable "noisy TV" event.**
  - Within an episode the world is deterministic.
  - The random new circle after a reset is never a prediction target: the model predicts the
    terminal frame and drives, before the reset.
  - So a large frame or drive error marks something the model has not learned yet. Clicks near the
    edge stay partly ambiguous from pixels, and storing more of them is wanted.
  - The move is different: the basic agent's moves are random draws from its own prediction.
    Measuring a move against the predicted spread keeps ordinary draws at their usual level. Only
    moves the prediction did not expect stand out: babbled moves, and clicks it gave a low
    probability.
- **Note on reward-freeness:**
  - The losses stay reward-free.
  - For the planner agent, the gate changes only the world model's training data.
  - For the basic agent, remembering relief more often and then imitating the remembered moves is
    selection: a mild form of reinforcement, without a reward gradient.

## Hypothesis

1. **Basic agent (A+H+R).** Storing relief raises the share of hindsight-labelled steps in long-term
   memory from about 3% to 10–20%. Its hunger-conditioned moves become goal-directed:
   - test success rises above the random initial model (0.44–0.48), for the first time;
   - online, it hits in more than about half of the episodes, the level of random babbling.
2. **Surprise** also keeps hits while they are still unexpected, plus the failed clicks. Alone, it
   enriches relief less than the relief rule. In the basic agent, its move part keeps the moves its
   prediction did not expect, mostly babbled ones, in memory. This may slow the narrowing of its
   moves (vm-00006: the std of dx fell from 0.60 to 0.40, and to 0.24 with replay).
3. **Planner agent (L).**
   - Relief alone makes the hunger head optimistic at the edge: more imagined relief just outside
     the circle. Adding surprise removes this.
   - Gating may speed up early learning (uniform storage: test success 0.58–0.88 at 10k). It changes
     little later, when almost every chunk leads to a hit.

## Deliverables

1. **Gated memory in `online.py`**, with `online.memory="gated"`. The default, `"stream"`, keeps
   vm-00005/6 bitwise.
   - **Short-term buffer:** the last 256 steps (the current ring buffer, made small). It serves the
     acting window, the newest training sequence, and the steps waiting for a decision.
   - **Chunks:** the stream is cut into consecutive chunks of `ctx` (8) steps, independent of
     episodes. Episode starts and terminal steps inside a chunk are marked as now. A chunk is
     decided `relief_k − 1` (11) steps after its last step, when it is known whether relief follows.
   - **Gate.** A chunk is written to long-term memory when:
     - **relief** (`store_relief=1`): a hit happens in it or in the next `relief_k − 1` steps, so
       every step followed by a hit within `relief_k` steps is kept;
     - **surprise** (`store_surprise=q`): its surprise is in the top fraction q of the last 1,000
       chunks;
     - **otherwise**, with probability `store_base` (1 = everything: the uniform control).
     - **Warm-up:** the first `store_warmup` (1,000) steps are stored in full in every arm. This
       also gives the surprise quantile its first history.
   - **Surprise** of a step: how wrong the model's prediction of the whole next state was. It is
     measured right after the world step, before the model learns from it, with one extra forward
     pass (`agent.forward` with `pred_actions`, on the window ending at that step). Its parts are
     measured as in the model's own losses:
     - **frame:** the error of the predicted next frame, with changed pixels weighted 11× as in
       training;
     - **drives:** the squared error of the predicted next hunger and restlessness;
     - **move:** how unexpected the move made was under the predicted move distribution:
       0.5 Σ ((a − μ)/σ)² over dx and dy, plus −log p of the click outcome. This is the part of the
       move loss that depends on the move. Basic agent only: in L the planner chooses the moves, and
       the model does not predict them (it would with the planner-as-teacher idea).
     - **Combined:** each part is divided by its running mean over the last 1,000 steps, and the
       parts are summed. A step where every part is as wrong as usual scores 3 (2 in L). One where a
       single part is 20× as wrong as usual scores about 22.
     - A chunk's surprise is the maximum over its steps.
   - **Long-term memory:** a chunk-aligned ring buffer of `replay` steps (100k = 12,500 chunks).
     - Stored chunks keep their hindsight labels. These are computed at writing from the short-term
       buffer, and they stop at the episode's end.
     - The stream's look-ahead does not stop there: in the rare case of a hit early in the next
       episode, it labels the last steps of a timed-out episode. `"stream"` keeps this, for
       reproducibility.
   - **Training batch:** the newest sequence from the short-term buffer (ending `goal_k − 1` steps
     back with `goal`), plus B − 1 chunks drawn uniformly, with replacement, from long-term memory.
     Until the first chunk is stored, they come from the short-term buffer, as now.
   - **No importance weights** in this task. The losses are unchanged, and the bias is measured
     instead (see Evaluation).
   - **Metrics** (`metrics.jsonl`):
     - chunks seen and stored, by reason (relief, surprise, base; a chunk counts under the first
       reason that applies);
     - for the chunks stored by surprise, which part contributed most;
     - long-term memory fill;
     - the share of labelled steps in long-term memory;
     - the surprise threshold, and the running mean of each part;
     - in L, the one-step error of the latent hunger head, as a diagnostic: the planner imagines
       with it, but the gate does not use it.
2. **Tests** (`tests/test_online_memory.py`), on synthetic streams with known hits:
   - the relief rule marks exactly the chunks with a hit in them or in the next k − 1 steps;
   - the labels written with a chunk equal the stream's look-ahead within an episode, and stop at its
     end;
   - with `store_base=1`, every chunk is stored, in order, and sampled sequences are always whole
     chunks;
   - each part of a step's surprise, and the combined score, match a direct computation from the
     `forward` outputs;
   - the surprise gate stores the top fraction q.
3. **Configs**, both with `memory="gated"`; the arms set `store_*`:
   - `configs/AHR_gated.json`: vm-00006's A+H+R (A_basic with `goal=1`, `use_prev_reward=1`,
     ε 0.2, batch 8, memory 100k);
   - `configs/L_gated.json`: L_online.
4. **Arms (vm-00007)**, each an independent agent in its own world:

   | Arm | store_relief | store_surprise | store_base |
   |---|---|---|---|
   | U (uniform control) | 0 | 0 | 1 |
   | R (relief) | 1 | 0 | 0.1 |
   | S (surprise) | 0 | 0.1 | 0.1 |
   | RS (both) | 1 | 0.1 | 0.1 |
   | U10 (size control, added during the runs) | 0 | 0 | 0.1 |

   - **Basic agent:** U, R, S and RS; seeds 1–2; 50k steps; 4 threads.
   - **U10**, added after 27k steps: the gated memories turned out 4–7× smaller than U's, so their
     chunks are replayed more often. U10 has the same base rate with uniform content, which
     separates what is stored from how much.
   - **I+U and I+R**, added after the basic arms finished: the basic agent with the inverse-dynamics
     head (`model.inv_hidden=128`, `ppo.inv_coef=0.1`), seeds 1–2, 50k steps. Probes showed that the
     basic agent's features do not hold the direction to the circle, while the planner agent's,
     trained with that head, do.
   - **Planner agent:** U, R and RS; seeds 1–2; 30k steps; 8 threads.
5. **Evaluation:**
   - **Online:** hits and episode lengths per 1,000 steps, plus the memory metrics above.
   - **Basic agent:** its own moves on the 128 test episodes, every 10k steps (`online.py --eval`).
     Compared with birth (0.44–0.48) and with vm-00006 A+H+R.
   - **Planner agent:**
     - `plan.py` at 10k, 20k and 30k;
     - `analyze_imagination.py` at 30k, for optimism (edge test, `false_relief_frac`).

## Implementation details

- **Decision timing.** The chunk of steps c … c+7 is decided after step c + 7 + (relief_k − 1) is
  stored. A short-term buffer of 256 steps is ample.
- **Context starts.** A stored chunk is a whole training sequence, so its first token always starts a
  context, as t = 0 does now.
- **Speed.** The surprise costs one extra forward pass per step, with the prediction heads: 8 tokens
  for the basic agent (about 6 ms), 1 token for L (one-frame model). Measure before launching.
- **Frame part.** The target is the next frame downsampled as in training (`downsample_u8`: block
  average with rounding). Pixels that differ from the downsampled current frame count 11×.
- **Seeds.** The gate draws from its own seeded RNG.

## Verification

- `memory="stream"` reproduces the current code: 300-step runs of L_online, and of A_basic with
  `goal=1`, give byte-identical checkpoints before and after the change.
- The new tests and the existing ones (281 Python tests plus the C++ tests) pass.
- Smoke runs (3,000 steps) of RS for both agents:
  - store rates and memory composition as expected;
  - no NaNs;
  - speed measured.
