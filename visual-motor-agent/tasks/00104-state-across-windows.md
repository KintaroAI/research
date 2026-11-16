# Task 00104: A State Carried Across Windows (experiment vm-00093)

**Date:** 2026-11-15
**Status:** In progress (approved 2026-11-16, the owner: "yes - lets do this - utilize both gpus for model run")

## Context

- **vm-00091 (task 00102):** in a 128 × 128 world with 2 foods, seen through a 64 × 64 window, the value agent (VW, here C8) eats 47.6 food per 1k ticks.
  - It turns away from food that has just left the view less than a blind baseline does, for 1–8 ticks (+.062 ± .017, without the ticks after a meal). After that there is nothing.
- **vm-00092 (task 00103):** the same agent with a context of 32 (C32) has the frames of the lost food in its input for 32 ticks, but does not use them.
  - Its aim at the unseen food is no better than C8's at 9–16 ticks (+.018 ± .025) or 17–32 (−.008 ± .032).
  - It eats half as much (23.4 per 1k) and almost never learns to avoid poison.
  - The outcome set before its results: the context is not the bottleneck, and the model does not keep what is in its input as a place to return to. Next is a summary carried across windows, written up for the owner's decision.
- **The owner (2026-11-15):** chose option (b) of vm-00092's Next steps: two consecutive windows per sample, the second starting from the first's state, with gradient through both.
- **Earlier loops without that gradient:** in vm-00056 the model's own prediction of a thought field came back as its next input. The thought stayed alive and formed attractors, but held nothing of the world: nothing trained it to keep what later predictions need.
  - The gradient through the carry is what this task adds.
- **The other memory in the plan:** the flexible layer (W = S + F, ARCHITECTURE.md, "Three memories") is still untested. It is the next step if this one fails (below).

## Design

### The state

- **Per step and per quarter transformer, two more tokens** (32 → 34 tokens per step):
  - **write,** the last token of the step. It has no input values (only its slot embedding) and no target. Its expected output values, V = 64 numbers in [0, 1] from the shared head, are **the state.**
  - **read,** the first token of the step. Its input values are the state written by the same transformer **8 ticks earlier** (one context back), centred like every input. It has no target.
- **Every other token's target is unchanged:** the move still predicts the next body, and the patches and summaries predict as before.
- **Why 8 ticks back, not 1:** the steps of the last 8 ticks are already in the context, so the state only needs to carry what is older. Reading one context back also keeps training parallel: within a window no read depends on a write in the same window.
- **The state stays in the one token format:** it is the model's own output read back as values, as the owner's "reflection is generation read back" asks, not a new kind of input.
- **The state chains without limit:** in acting, a write at tick t is read at t + 8, where the next write sees it, and so on. Training (below) teaches one link of the chain: whether it carries further is measured (17–32 ticks).

### Training: two windows per sample

- **The update keeps C8's 8 windows of 8 steps,** taken as **4 pairs of consecutive windows** of one body (the newest pair and 3 replayed). The update is spread over no more ticks than C8's, which matters because spreading it over more is the candidate cause of C32's cost (vm-00082, vm-00092).
- **Window 1** of a pair reads the states stored while acting: no gradient, since older weights wrote them. Its writes are computed fresh.
- **Window 2** reads window 1's fresh writes: step k reads the write of step k − 8.
- **The carry's gradient:** the expected value E = Σ p_j c_j has dE/d logit_j = p_j (c_j − E), so window 2's loss reaches window 1's write logits and through them everything window 1 computed.
- **Both windows' losses count** as now: every token's next-token loss, and the value labels.
- **The arms differ only in the carry:**
  - **SB:** `online.state_grad = 1`, the gradient flows through the carry;
  - **SA:** the same reads, detached. SA is option (a) of vm-00092's Next steps with the same pairs and tokens, so SB against SA isolates the gradient.

### Acting

- Each transformer of each body keeps a queue of its last 8 writes. The new step's read is the write from 8 ticks back. At birth, and for the first 8 ticks, the read is 0.5 (centred 0).
- The ring computes the new step's tokens, read and write included, on its cache, as now.
- **Memory** stores each tick's read state per transformer: field `st`, 4 × 64 values per tick, in short-term memory and in the long-term chunks. A replayed window 1 thus starts from the state it had while acting.
  - Long-term chunks must keep consecutive pairs: chunks of 16 steps, as C32's were 32.
- **The value refresh** (`online.value_refresh`), whose windows lie n ticks later, reads their stored states.

### Arms

| Arm | Label | Model | State | Seeds |
|---|---|---|---|---|
| C8 = vm-00091's VW (not rerun, linked as in vm-00092) | value, γ 0.9, n 4, fresh tail | 4 quarters × 8 layers, tolerance 0.002, ctx 8 | none | 1–12 |
| **SA** | the same | the same | read and write tokens, carry detached | 1–12 |
| **SB** | the same | the same | read and write tokens, gradient through the carry | 1–12 |

- **The world and everything else are C8's:** `--config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 --set online.value_gamma=0.9 --set online.value_refresh=1`, plus `--set model.state=1` and `--set online.state_grad=0` or `1`.
- **Dropping SA** halves the runs but leaves reading 3 unanswered.

## Code

1. **`model.state = 1`:**
   - two slot kinds in `agent_lib.uni_layout` and `vm_uni_layout`: read first and write last in each step;
   - slot embeddings and the 2-D position codes, placed as task 00052 requires of any new slot;
   - no targets for either slot;
   - a checkpoint header slot (an extension field at 32+).
   - `model.state = 0` must stay byte-identical to today.
2. **The update** (`update_rollout`, CUDA):
   - with `model.state`, the batch is `online.batch / 2` pairs;
   - window 1 runs, then window 2, with window 2's reads taken from window 1's write outputs;
   - the backward adds dE/d logit at window 1's write tokens when `online.state_grad = 1`.
3. **The reference** (`reference/torch_reference.py`, float64): the same pair forward and backward.
4. **Acting and memory** (`lifetime.py`, `uni_act`, `online.py` memory):
   - the queue of writes per transformer and body;
   - the `st` field and chunks of pairs;
   - the newest pair in the batch;
   - refresh windows that read stored states.
5. **`eval_hunger.py`:**
   - the frozen `Life` acts with the queue;
   - `--state blank` reads 0.5 every tick (the ablation);
   - `chain` also logs the state's spread over ticks.
6. **Tests** (`tests/test_state.py`):
   - CUDA against the reference: loss and every gradient, including the carry (window 1's weights get a gradient through it with window 1's own loss masked), at about 1e-7;
   - acting on the ring equals a training forward with the same reads;
   - `model.state = 0` is unchanged;
   - **a carry test:** a tiny model on a copy task across the window boundary (a value shown in window 1, asked for in window 2) learns it with the gradient through the carry, and not with the carry detached.
7. **Timing** before the lives: 1,000-tick lives at 12 per card, SA and SB. The expectation, not yet measured, is near C8's (4.8 ticks/s on the 4090), since the windows per update are the same, with two half-batch passes in place of one.

## Measures

- **Lived, 40–60k:** food, poison, hunger, overrides, move CE, smoothness; each arm against C8, and SB against SA, paired by seed.
- **Frozen `chain`** (1,500 ticks): food, the share with no food in view, and aim at unseen food by ticks since the last sight. Seen − blind, without the ticks after a meal, in vm-00092's bins (1–8, 9–16, 17–32, 33–64, 65+).
  - The same with the state blanked (`--state blank`), for SA and SB.
  - The state's spread: the mean over its 64 values of their standard deviation over ticks.
- **Frozen `value` and `reach`,** as in vm-00092.

## The readings, set before the results

1. **Plumbing:**
   - the tests pass;
   - all 24 lives run 60k ticks with `model.state` 1, without errors;
   - SB's state is alive: a spread of at least .02 in at least 10 of 12 seeds.
2. **Memory reaches further (the main one):** SB's aim at unseen food at 9–16 ticks (seen − blind, without meals) is above C8's, paired over 12 seeds, by more than its interval.
3. **The gradient does it:** SB's 9–16 memory is above SA's, paired, by more than its interval.
4. **The memory is in the state:** blanking SB's state lowers its 9–16 memory, paired by seed, by more than its interval.
5. **Memory pays:** SB's lived food at 40–60k is above C8's, paired, by more than its interval.
6. **The pairs cost nothing:** SA's lived food is not below C8's by more than its interval.
7. **Further than one link** (descriptive, no bar): SB's 17–32 memory against C8's and SA's.

**What follows:**
- **If 2, 3 and 4 hold:** a learned state carries memory. Reading 5 then says whether it pays. Next: longer chains in training (three windows), and the reach at 17–32.
- **If 2 holds and 4 fails:** the gain does not come through the state (perhaps the pairs alone). Look again before going on.
- **If 2 fails:** one link of gradient through a 64-number state is not enough to make the model keep where the food was. Next is the flexible layer F (W = S + F), the other memory the plan names.
- **If 6 fails, whatever else holds:** the pairs or the extra tokens cost, as C32's longer windows did. Separate that before reading 5.

## Implementation

- [x] The owner approves this design (2026-11-16).
- **As built (2026-11-16):**
  - **The slots:** the body, then the write, then the read. The move stays last and the token before it still predicts it, so no kernel that finds tokens by position changes.
    - The body's prediction (of the write) and the write's (of the read) carry no loss; the read predicts the first patch, as the body did.
    - The write sees the step's body and every earlier step, but not its own frame. A frame reaches the state one tick later, through the next step's write.
  - **The GPU:**
    - `vmc_set_uni_state` (the reads of the next pass or ring step; none: blank, 0.5);
    - `vmc_uni_writes` (the writes' expected values);
    - for the pairs: `vmc_set_uni_train` (loss scale 0.5, accumulate, keep the gradient at the reads), `vmc_uni_read_grads`, `vmc_set_uni_carry` (the gradient at the first window's writes, through dE/dlogit = p (c − E) in `k_uni_loss`) and `vmc_apply_update` (clip and AdamW on the accumulated gradients).
    - `cuda_lib.CudaModel.update_pairs` runs a forward pass of the first windows for their writes, the second windows' pass, then the first windows' pass with the carry. It costs about one extra half-batch forward pass over C8.
  - **The C++ model** reads a blank state (it has no way to be given one); the GPU without a state given is the same model.
  - **The queue of writes** lives in `CudaModel` (`uni_ring_reset`, `uni_ring_step`; `last_read` for memory; `state_blank` for the ablation).
  - **Memory:** the field `st` (half precision, transformers × 64 per tick), chunks of 16 steps, `b_st` for the refreshed value's bootstrap windows (`boot_values` reads them, a context at a time). `lifetime.py` refuses the state with self-forcing, clog, input noise, outages, delays, the thought world or surprise gating.
  - **Metrics:** `state_spread` (body 0's reads, their standard deviation over the log window, the mean over transformers and values). `eval_hunger.py chain` reports the same over its test; `--state blank` names the rows ARM-blank_sN.
- **The copy task (tests/test_state.py):**
  - With 32 bins, as the runs use, SB learned it by update 750 in 3 of 3 seeds.
  - With 8 bins it sat on a plateau, and learned within 3,000 updates in only half the runs.
  - Detached (SA), the writes are never trained, but the reader still learns to decode what they happen to carry: error 0.04 against 0.083 at chance, against SB's 0.0001.
  - So SA is not a model without memory. Reading 3 measures what training the writes adds.
- [ ] Code 1–5, with the tests (6) passing; `model.state = 0` byte-identical.
- [x] Timing (7), 2026-11-16 14:40, 600-tick lives of both arms under MPS, 14 on the 4090 and 10 on the 3090: 3.58 and 2.60 ticks/s per life (C8 in vm-00091: 4.8 and 3.3), about 0.8 GB per life. The state costs about a fifth of the speed. Balanced: seeds 1–8 of both arms on the 4090 (16 lives), 9–12 on the 3090 (8), about 5.3 h each.
- [ ] `00093-state-across-windows/` (README, Makefile); the runs; frozen tests (C8 included, linked); write-up and videos.

## Verification

- Every run's `info.json` has `model.state` 1, `online.state_grad` as its arm, `model.ctx` 8, `env.view` 64, `n_layer` 8, `instinct_tol` 0.002, `value_gamma` 0.9 and `value_refresh` 1.
- C8's frozen rows repeat vm-00092's exactly.
- With the state blanked, SA's and SB's frozen food is reported beside their own, so a state the agent relies on for anything shows.
