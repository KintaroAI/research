# Task 00114: Labels from Imagined Choices, and One Model Learning from Another's Dreams (to explore later)

**Date:** 2026-11-20
**Status:** Proposed, for later (the owner, 2026-11-20, during vm-00100: "lets add this to explore later - extended version of what we are doing now - now we replay memory and then we learn label out of it. But also we can try to replay as if model did different choice and try to predict what will happen and learn label from those imagined outcomes. it's one part and another part - we need a mechanism where one model imagine things and another model act as it was real and also learn from it - some sort of inter model's distilation")

No experiment number yet: the design is settled after vm-00100 (task 00112) and vm-00101 (task 00113).

## Context

- **What replay does now** (`online.label_refresh`, task 00112):
  - memory keeps what happened; at every update each transformer re-labels each replayed step from its current prediction for the next move;
  - only the move actually made gets a target. Every other move's label comes only from generalization;
  - but the instinct chooses among 16 candidate moves, scored by exactly those untrained labels.
- **At 30–40k ticks, re-labeling eats 4.6 times the food of storing once** (RD 46.9 per 1k against SD 10.1, all 12 seeds). Fresh labels on real steps work. The two ideas below extend this past what really happened.
- **What earlier runs say about learning from imagination:**
  - **vm-00004:** a planner exploits the model's errors (relief imagined just outside the circle). A hunger head trained mostly on imagined latents smeared its edge. Real failures in the training data fixed it.
  - **vm-00023:** imagined frames lose the moving cursor within a few steps. A 4-step lookahead through them ranked moves no better than 1 step, and the lives clicked more poison.
  - **vm-00032 and vm-00034 (self-forcing):** training on the model's own predicted inputs made the dream truer and prediction better, but not behaviour; its own samples got worse.
  - **vm-00068 (mimic, text):** half 2 learned only from half 1's predictions, never saw the text, and beat half 1 on held-out text by 0.05–0.30 nats. That is the second idea in a small form, and it worked.

## Part A: re-labeling moves that were not made (imagined replay)

- **The idea:** for a replayed step t, besides the real move a_t, take other moves a′ and let the model predict what they would have done. Then train the label of (window up to t, a′) toward the label of that imagined outcome.
- **The target, by the same rule as the real label:**
  - a predicted meal (hunger falls under a′): V = 1;
  - a predicted poison: minus the predicted rise (full dose), plus γ × V̂ of the imagined next step;
  - otherwise γ × V̂(imagined next state, the model's next move there).
- **Which moves a′:**
  - the instinct's own candidates at that step, so the labels it compares get targets;
  - or moves from fixed directions (the 8 compass pushes and staying), which works even when the candidates collapse (vm-00098 R8).
- **Stages:**
  - **A0, no imagined frames:**
    - the meal or poison part comes from the predicted hunger for a′, which is trained on real data;
    - the bootstrap V̂ comes from the real next window, with a′ written in place of the move made.
    - This is cheap but wrong where a′ leads somewhere else. Under inertia a one-tick push moves the next position by at most 1 px, so it is close.
  - **A1, one imagined step:**
    - the next frame and body are decoded from the model's own prediction (`pair` decoding; each quarter piece imagines its own block, as in task 00107);
    - V̂ is read there.
    - One step keeps the vm-00023 cursor loss small. This is MBPO's choice: short rollouts branched from real states.
  - **A2, k imagined steps:** only if A1 helps and the dream holds (`eval_dream.py`).
- **Guards against exploiting errors (vm-00004):**
  - the 4 transformers are an ensemble: train on an imagined target only where they agree on the outcome (their spread below a threshold), or weight it by agreement (as in STEVE);
  - imagined targets get a smaller weight than real ones, and every batch keeps its real windows;
  - each transformer re-labels with its own predictions, as now.
- **What it should change:**
  - the labels of moves never made become informed rather than smooth guesses;
  - the candidates' spread grows (`eval_candidates.py` read .0001 or less in SB);
  - the instinct overrides with reason.
- **The first measure:** frozen, how well the label of moves not made predicts what they really do. A copy of the world can be stepped with each candidate (the `click-real` machinery, task 00027). This is checked before any lifetime.

## Part B: one model dreams, another learns as if it were real (peer distillation)

- **The idea:** transformer i, the dreamer, generates a rollout: frames, body and its own moves, started from a real replayed window. Transformer j, the learner, trains on it as if it were memory: dreamed inputs and dreamed targets, including the labels re-labeled on the dream.
- **Why another model, not itself:**
  - training on its own samples reinforces its own errors (vm-00034: generation got worse);
  - a peer's errors are different errors, so the learner gains the peer's knowledge without its own biases being confirmed;
  - in vm-00068 the learner even overtook the teacher.
- **What we have:** the 4 quarter transformers are already peers with their own weights and their own views. Each sees its quarter and every summary.
- **Variants:**
  - **B1, round-robin:** each transformer is dreamer for the next one; a fixed share of each batch (say 1 window in 8) is a peer's dream.
  - **B2, a dreamer that does not learn from dreams:** a frozen snapshot of the life a few thousand ticks back, as in born-again networks. The teacher then does not drift with its students.
  - **B3, labels only:** the learner takes only the dreamer's labels for the real window (codistillation, deep mutual learning). There are no dreamed frames, so the cursor is never lost; this is the cheapest and safest first step.
- **Guards:**
  - dreams start from real windows and are short at first (1–3 steps, the length self-forcing used);
  - real windows dominate each batch;
  - a dream is dropped when the peers disagree strongly on it.
- **The known risk:** models trained on generated data narrow and drift over generations (model collapse). The mix ratio and the real share of each batch are the levers. Measure it with the dream curve (`eval_dream.py`) and with prediction on held-out real windows.

## How the two parts relate

- A is replay with a different choice; B is replay of someone else's imagination.
- Together they form one loop: the dreamer imagines a different choice (A), and a peer learns from that dream as if it had lived it (B).
- **The order to try:**
  1. A0, then A1 (one step, ensemble-filtered), against RD;
  2. B3 (peer labels, no dreamed frames);
  3. B1 or B2 with short dreams;
  4. the combination.

## Measures (as vm-00100)

- **Lived** by 5k and at 40–60k: food, poison, mean hunger, overrides, move CE, smoothness; the seeds that starve.
- **The monitors** (`meal`, `meal_stored`).
- **Frozen:** `chain`, `value`, `reach`, `move`, `eval_candidates.py`, plus:
  - the label's accuracy on moves not made (Part A's first measure);
  - for Part B, the dream curve and prediction on real windows, learner against dreamer.
- **Cost:** ticks per second against RD's 3.0 on the RTX 3090. A1 adds about one forward pass per imagined move per replayed step, so a′ must be few (or a subsample of steps).

## Related work (RESEARCH.md §4 and §11)

- **Imagined replay:**
  - Dyna (Sutton 1990);
  - model-based value expansion (MVE, Feinberg 2018);
  - STEVE (Buckman 2018), which weights horizons by ensemble agreement;
  - MBPO (Janner 2019), short model rollouts branched from real states;
  - Dreamer (Hafner 2020–2023), which learns value and policy wholly in imagination.
- **Peer distillation:**
  - policy distillation (Rusu 2016);
  - deep mutual learning (Zhang 2018);
  - codistillation (Anil 2018);
  - born-again networks (Furlanello 2018);
  - model collapse from training on generated data (Shumailov 2024).
