# The Target Architecture: One Universal Model, One Lifetime

**Date:** 2026-10-25
**Status:** Design (universal model; not yet built). The earlier design with per-type heads, a detached policy and an
act token (vm-00016) is preserved in git (`exp/vm-00016`, this file's history); §12 says what changed and why.

## 1. Goals, in Order

1. **A universal model.** One network that treats everything it senses and does the same way: the same kind of token,
   the same prediction, the same loss, the same gradient path. No special handling for one kind of token (no detached
   heads, no per-type losses or weights, no per-type output heads). Special cases are where the earlier designs broke:
   each fix made the next one necessary (§12).
2. **Good prediction.** The model predicts the next of everything: the next frame at full resolution, its own next
   drives, its own next move. Prediction quality is measured directly (vm-00017's click test, frame error where pixels
   change, token probes).
3. **Basic motivation (instincts).** A minimal drive, here "lower my hunger", acts through the model's own predictions
   (§6), not through a separate network, a reward or a special loss.

Behaviour is not the primary goal at this stage; whether the predictions are right is.

## 2. Principles

- **Prediction is the only teacher.** Every token's output predicts the next token. Nothing is rewarded.
- **One lifetime.** One agent, one world that never resets; it acts and learns on every tick (§7).
- **Many bodies early, one brain throughout.** The same weights may drive many bodies in parallel worlds at the
  start, shrinking to one (vm-00016: the strongest basic agent so far).
- **The model senses only what a body would:** the RGB frame, its drives, its own previous move. True positions are
  for scoring only.
- **The model does not know its world** (the owner's, 2026-10-26): pixels and text (task 00032) are interchangeable;
  only the world draws its state into a frame. Nothing in the model treats one kind of world differently. When a world
  shows a limitation of the model, the model is changed so that it goes away for every world alike.

## 3. The World

- 64×64 RGB, a continual torus: a new red circle appears after each one eaten; nothing resets (vm-00011).
- Food: clicking the solid disc relieves hunger. Poison: clicking a checkerboard disc (every other pixel missing,
  `env.hazard_pattern="checker"`) adds 0.5 to hunger (`env.hazard_click_add`, vm-00018); contact alone costs nothing.
- **Grayscale** (`env.gray`, task 00024): food and poison differ by texture, not colour, and every patch is one channel,
  a third of the values to encode and predict.
- **Eight moving grey dots** (`env.distractors=8`, vm-00010): content and motion the agent does not cause, so it must
  work out which moving thing obeys its moves.
- All of this is the universal model's default world: `dev/configs/U_universal.json`.
- **The text world** (`world/text_world.py`, task 00032): a text as grey pages, a letter as a brightness (letter k of
  the sorted alphabet at level round(k · 255 / (n − 1))), each step the next page; drives 0, moves ignored. The same
  model, unchanged, with `bins = 256` (every grey level a bin centre, so a value's prediction is a softmax over
  letters); its prediction is measured in nats per letter against a classical character-level GPT of the same size
  (`reference/char_gpt.py`, `eval_predict.py`).
- Slow hunger (0.001 per step) so that hunger rarely sits at its cap of 1, where changes leave no trace (vm-00015).

## 4. Tokens: One Format for Everything

Every token carries a **value vector** of V = 64 numbers in [0, 1] (`model.gray=1`; 192 with RGB), with a mask for
the numbers it does not use:

| Slot | Per step | Values (in [0, 1]) |
|---|---|---|
| body | 1 | hunger, restlessness (2 used) |
| patch 1..64 | 64 | an 8×8 grayscale patch of the 64×64 frame, row by row: 64 (pixels / 255; RGB: 8·8·3 = 192) |
| move | 1 | the executed move: (dx + 1)/2, (dy + 1)/2, click (3 used) |

That is 66 tokens per step, 528 per window of 8 steps.

**One scale for every value.** Each value is mapped to [0, 1] by its fixed, known range (pixels / 255, hunger and
restlessness as they are, (dx + 1)/2, the click as 0 or 1; never by data statistics), so no kind of input is larger
than another: raw 0–255 pixels beside a hunger in [0, 1] would let the pixels dominate the shared weights. At the
input the values are centred, `values − 0.5`, in [−0.5, 0.5]; unused (masked) entries are 0, and a mask is passed to
the loss, not to the input. The outputs predict the same [0, 1] values (§5).

**One input map for all:** `x = W_in · values + b + slot[s] + step[t − window start]`, with one shared `W_in` (192 → d).
`slot[s]` is a learned embedding per slot (66 of them: the body, each patch position, the move); the patch slots start
from a 2D sine-cosine code of the patch centre. The slot embedding is the only thing that says what a token is, as a
position embedding is in an LLM. The cost of sharing `W_in`: hunger enters through the same weights as pixel 0 of a
patch, and the network must separate them using the slot embedding.

## 5. One Prediction, One Loss

**The sequence is fully causal, as in an LLM:** each step is `body_t, patch_1,t … patch_64,t, move_t`, and every token
predicts **the next token's values**. That one rule covers everything:

| Token | predicts | which is |
|---|---|---|
| body_t | patch_1,t | the first patch of the frame, from the past and the drives |
| patch_i,t | patch_i+1,t | the rest of the frame, patch by patch |
| patch_64,t | move_t | **the policy**: the next move, from the whole observation |
| move_t | body_t+1 | **the consequence for the body**: next hunger, given the move just made |
| (then) | patch_1..64,t+1 | the next frame, with the move in context, at full resolution |

- **One output head** shared by every position: LayerNorm → linear (d → V·K). Each value gets K = 32 logits, a
  probability over 32 bins of [0, 1]. Values the next token does not use are masked out of the loss.
- **One loss: two-hot cross-entropy.** A target value between two bin centres puts its probability on those two bins in
  proportion to its distance, so values finer than a bin (hunger rises by 0.001 per step) are still learned. Each
  token's loss is the mean over its used values; the total is the mean over tokens. Every token counts the same; there
  are no hand-set weights. Cross-entropy also scores a surprising change by its probability, not its size: the small
  but systematic effects that squared error drowned (hazard contact, vm-00013–15) count.
- **Predicted values** (for acting and for analysis) are the expected bin centre.
- **Nothing is detached.** One gradient-norm clip for everything.

Why the move loss should no longer erase the scene (a hypothesis, §10): the erasing (vm-00007/14) came from the move
loss being a separate objective whose gradient was hundreds of times larger than the pixel loss. Here the move is one
token of 66 per step, in the same units as every other token.

## 6. Acting: the Model's Own Prediction, Plus One Instinct

- **The policy is the model's prediction of its own next move:** the distribution the token patch_64,t predicts for
  move_t.
- **The instinct (hunger) acts by one-step lookahead with the same model:** sample M candidate moves (M = 16) from
  that distribution; for each, put it in the move_t slot and read the predicted next hunger from the move token's
  output (the prediction of body_t+1); make the candidate with the lowest predicted hunger. One batched forward pass
  for the M candidates. Where every candidate predicts the same hunger (most of the time: away from a circle), the
  choice is the model's first sample, so the agent acts as its policy.
- **The instinct teaches the policy through ordinary prediction:** the move made is the move_t token in memory, and the
  model learns to predict its own moves, so what the instinct chose becomes habit (as the planner taught a habit in
  vm-00009).
- **Exploration** comes from the model's own uncertainty (its move distribution is broad at birth), with no babbling
  rule. If the agent's moves collapse (the earlier agents drifted into never clicking without babbling), that is a
  finding to fix generally, e.g. by an entropy floor that applies to every token's distribution.
- One step ahead only reaches what one move can change: a click on food or poison, not the approach to a far circle.
- **Longer lookahead (task 00029, `online.instinct_h`):** each candidate is held for H steps and scored by the
  predicted hunger after the last. The steps after the first are imagined by the model itself: the body from the move
  token's prediction, the patches decoded from the model's own expected values by Jacobi iteration from the current
  frame (`online.instinct_jacobi` passes per step; patches + 1 passes are exactly sequential decoding). Only the first
  move is made; the next tick plans again.

## 7. Learning: One Update per Tick, Many Bodies at First

```
every tick:
  each body b:  its window → the move (policy + instinct) ;  its world steps ;  the transition is stored
  one gradient step on:  the newest window of every body  +  windows replayed from memory
```

- One AdamW update per tick, lr 3e-4. The newest window is always in the batch; replayed windows keep it stable
  (vm-00006).
- Parallel bodies: 32 for the first 10k ticks, halving every 5k down to one (vm-00016: test episodes 0.91–0.98, forage
  rate 1.8× a single body). The weights, optimizer and memory carry through. vm-00018 separates the extra experience
  from the bigger batch.

## 8. Memory

| Kind | What | Size |
|---|---|---|
| working | the attention window | 8 steps (528 tokens) |
| short-term | each body's recent steps | 256 steps per body |
| long-term | 8-step chunks, shared by all bodies | 100k steps |
| weights | everything learned | 0.68M parameters in grayscale (the output head 0.26M; RGB: 1.22M, head 0.79M) |

- To begin with, long-term memory keeps every chunk (vm-00007's uniform control). A universal salience rule is the
  model's own surprise, which counts every kind of token the same way. Gating by hunger relief is a special case and
  is dropped.
- **Surprise gating (task 00028, `online.store_surprise` on the GPU):** each token's cross-entropy in the update that
  trains on its step (as the newest window), divided by that slot's mean over the last `surprise_window` steps; a
  step's surprise is the largest ratio, a chunk's its largest step's; the top fraction of recent chunks is kept. A
  token is surprising when it is much more surprising than usual for its own slot, whatever kind it is.

## 9. To Do Later

- **Learned embedding selection, as in a text LLM** (the user's idea, 2026-10-25): in an LLM a token id selects a row of
  a randomly initialized, trainable embedding table, so the token and its embedding are decoupled. Here: a table of E
  learned embeddings (random at birth) and a small trainable network that reads a token's values and chooses among
  them, softly (a softmax over the E rows, weighting them) or hard (the argmax, with a straight-through gradient). The
  embedding is then `Σ_e softmax(g(values))_e · table_e + slot + step` instead of `W_in · values`. What a token *is*
  becomes a learned choice from a vocabulary of embeddings. It replaces only the input map; the rest stays universal.
  The hard version with a reconstruction target is the VQ codebook below.
- **Learned discrete tokens: a patch tokenizer (VQ-VAE or FSQ)** (to test; the owner, 2026-10-26): one
  encoder → bottleneck → decoder for every token's values, trained to reconstruct them; the bottleneck snapped to one of
  K codes (a learned codebook, VQ-VAE; or FSQ: a few latent numbers each rounded to a few levels, e.g. 8·8·8·5·5·5 =
  64,000 codes, no codebook to collapse). The code is the token: an embedding table in, a softmax over K out, as GPT-2.
  Gains: a coherent joint prediction of a whole patch (the cursor here *or* there, not a per-pixel smear: task 00029),
  an embedding per code (vm-00027: faster learning), one loss in nats per token for every world. Risks:
  - **Rate allocation (the owner's concern):** reconstruction error spends codes where variation is large (patches),
    so low-variance tokens (the body: hunger drifting 0.001 per step) get few codes and lose their fine changes, though
    they matter most to the instinct. Universal countermeasures, the same for every token: each token's reconstruction
    loss is the mean over its own values (as the prediction loss); each value slot normalized by its own running spread
    before encoding; FSQ's fixed levels (no token type takes codes from another); residual quantization (a second,
    finer code) only if still short.
  - Codebook collapse (VQ): k-means initialisation, running-average codebook updates, dead-code restarts; or FSQ.
  - Lossy: small things (the 3-pixel cursor) may merge into the background's code.
  - Online learning: code meanings drift under the transformer; train the tokenizer first, or update it slowly.
  - **Two arms to compare:** (a) one tokenizer shared by all slots, the embedding = code embedding + slot embedding;
    (b) **a tokenizer per slot** (the owner's idea, 2026-10-26): slot i has its own small codebook of K codes, and the
    token index is i·K + j, so the embedding table learns content and position jointly and no slot can take codes from
    another (it answers the rate-allocation risk directly). Its costs: no sharing across positions (an edge at patch 5
    and at patch 37 get unrelated codes and embeddings; each slot learns from 1/64 of the patches), and K must cover a
    slot's contents (10 codes cannot hold the cursor's 64 positions or 65 letters: several small codes per slot, as
    FSQ, would). Expected: (b) wins on the body and move tokens, (a) on patches.
  Checks: reconstruction error and codes used per token type; hunger resolution (the smallest hunger difference that
  changes the code, against today's 32 bins: 0.03); whether relief and poison jumps and the cursor survive the round
  trip. The bins of today are already a fixed per-value quantizer; the tokenizer's gain is for patches.
- **An analogue of GPT-2's `wte`: one matrix for input and output** (the owner, 2026-10-27). GPT-2 embeds a token with a
  row of `wte` and scores the next token with the same rows (logits = h · `wte`ᵀ). With the bin-embedding input
  (`model.in_bins`, vm-00030) our two maps already have transposed shapes: `W_in` d × (V·K) embeds "value j at bin k",
  `W_out` (V·K) × d scores "the next token's value j at bin k". Tie them (`W_out = W_inᵀ`, own biases kept; a flag such
  as `model.tie_io`, the two gradients summed into the one matrix in C++, CUDA and the reference). Gains: every bin's row
  is trained at every position through the output, also bins rarely seen as input (rare grey levels, rare letters);
  one meaning per bin, the same for every world; ~28% fewer parameters (262k of 933k). Doubts: 32 bins per value is a
  small vocabulary next to GPT-2's 50k, so less to gain; a value position j is shared by every token type (pixel j,
  a drive, a move component) in both maps already, the slot embedding telling them apart. Test: the text world against
  the classical GPT (prediction, nats per letter) and the egocentric touch world (prediction, foraging), 3 seeds, tied
  against untied. With a learned tokenizer (above) the tied matrix is the code embedding table, exactly as `wte`.
- **A faster lookahead:** the imagined steps recompute the whole window for every candidate and pass; the real
  prefix is the same for all of them, so its keys and values could be cached once per tick (as the one-step
  candidates do).
- **Reusing the sliding window while acting** (the owner, 2026-10-30). Each tick now shifts the window by one step and
  recomputes everything: the last 7 raw frames (from the short-term buffer; tokens are never stored) plus the new one,
  a full pass over 8 steps. 7/8 of the input is the same as on the last tick, but the work is not saved.
  - **Input cache (cheap, exact, no model change):** a token's input, `W_in · values + slot`, does not depend on
    history; keep it in a ring per step and add only the step embedding. Skips the thumbnails and `W_in` for 7 steps.
  - **Key and value ring (a model change):** a new tick's tokens read the old tokens' keys and values, per layer, as
    `uni_candidates` does within a tick; the new step overwrites the oldest (~1.4 MB for 87 tokens per step). Attention
    ~4× cheaper, the rest ~8×. It needs two things:
    1. **Positions that do not move.** The step embedding is a position in the window, added into the token, so every
       old token changes on a shift. Position must become relative, applied inside attention from the two tokens' age
       difference: `age = t_reader − t_token`, 0..7. The slot (which token of a tick) never changes and stays in the
       token. The options (the owner, 2026-10-30):
       | method | added when reading | parameters | what it can express |
       |---|---|---|---|
       | age bias | `score += b[head, age]` | 8 per head | recent tokens matter more, the same for every reader |
       | **age key** (task 00062) | `score += q · e[age]` | 8 vectors, the shape of the step embedding | each reader decides how much it cares about age |
       | age key and value | also `v + f[age]` | 16 vectors | the output also knows how old its source is |
       | RoPE | q and k rotated by an angle set by the tick | none | age without limit; the rotation must be built into Q and K |
    2. **A decision on acting against training.** Dropping the oldest step changes what later tokens would have
       attended to, so the reuse is not the same computation: a cached token keeps what it saw when it was new, and
       information reaches further back than 8 steps (2 layers: up to ~2 windows). Probably useful, as a longer memory,
       but a training window computes from scratch, its first tokens with no context. To match them, training needs a
       warm-up of steps without a loss at the start of each window, at the cost of slower updates.
  - **Where it pays:** the update (8 windows, forward and backward) likely takes most of a learning tick (not measured:
    measure the split first), so it matters little while learning every tick. It matters for acting alone, as in the
    estimate of ~1,000 ticks/s, or if acting and learning separate. Order: input cache when speed matters; the ring
    when acting is separated from learning, or for longer memory.
- **How many pieces decide one move** (the owner, 2026-10-30). Randomness as such does not matter while the model
  learns; what matters is how the decisions of 4 or more pieces are addressed. Picking one still feels loose: many
  outcomes can hold the same potential hunger, and choosing among them is close to random. To think through later.
  - **What is known** (vm-00051, tasks 00059, 00060; frozen numbers are on QG's weights):
    - The rule teaches: every piece learns to predict the move made, so a rule must be judged by whole lifetimes.
      A random winner on QG's frozen weights keeps 38.3 food; a lifetime with a random winner makes 13.4.
    - The feedback: the mean shrinks proposals (8.8 food), the largest per axis saturates them (27.5 food, 21.9
      poison), a whole proposal made unchanged keeps their size.
    - Ties: the pieces' predicted hungers differ by more than 0.005 on only ~26% of ticks. On the other ~74%, gain chooses
      by noise.
    - The winner's curse: the lowest of the pieces' own predictions favours the piece that misses a danger. The mover
      foresaw ~1% of its 339 poison touches; another piece did in 60%.
    - Cost: scoring every move by every piece is S² one-token steps per tick, which does not stay cheap at many pieces.
  - **Algorithms to try:**
    | algorithm | how it chooses | answers | open risk |
    |---|---|---|---|
    | gain | the lowest own predicted hunger | what works so far (42.8 food, 4.8 poison) | ties; the winner's curse |
    | council | every piece scores every move, the lowest mean | the curse (frozen: 36.8 food, 1.7 poison) | S² scores; lifetime in vm-00052 |
    | veto | the lowest worst score | the curse, more strictly (33.4, 1.4) | timid; one pessimist blocks |
    | pessimism | the lowest mean + λ · spread of the scores | disagreement as a warning | λ; a middle of council and veto |
    | rank vote (Borda) | each piece ranks the moves, the ranks add | scores of different scale; a piece's bias | discards how much better a move is |
    | confidence on ties | gain, the sharpest move prediction on a tie | ties (frozen: 43.5, 5.8) | a habit spreads between pieces |
    | staying on ties | gain, the last winner on a tie | commitment | frozen: nothing (paths no smoother) |
    | mean of the tied | average only the proposals within a tie | ties, without arbitrary choice | cancels opposite proposals, shrinks them |
    | soft selection | a mean weighted by softmin(predicted hunger) | spans mean and gain by a temperature | shrinking returns at a high one |
    | direction vote | proposals vote in direction bins; the fullest bin decides, its mean is made | ties, opposite proposals | the bin size; loses precise moves |
    | population code | each piece has preferred directions; a move is their activity-weighted sum (as motor cortex) | a fixed split without fixed roles | the shrinking of any mean |
    | negotiation | pieces take turns: each sees earlier proposals (move tokens) before proposing | coordination itself | order; more passes per tick |
    | arbiter | a small model at the pyramid's top reads the pieces' proposals and picks or mixes | a learned rule | what trains it, with no reward |
  - **Checks for any of them:** food and poison over a whole lifetime against QG, three seeds; proposal size stable; the
    share of ticks decided by noise; scores per tick against S, since the point of the pieces is to cut compute.
- **Other instincts** (restlessness) through the same one-step lookahead on other body values.
- **Letters as glyphs** (the owner's idea, 2026-10-26): the text world draws each letter as an 8 × 8 bitmap of its
  shape (a font), one patch per letter, instead of a brightness: reading text as vision. The model must then learn the
  letters' shapes too, and text and images become literally the same kind of input.

## 10. Open Questions, Each an Experiment

| # | Question | Test |
|---|---|---|
| 1 | Does uniform two-hot cross-entropy learn the cursor without the changed-pixel weight (vm-00001: plain squared error ignored it)? | frame error on the pixels that change; cursor probes |
| 2 | Does the move token's loss, attached and uniform, leave the scene intact? | patch-token probes against vm-00014's detached (0.87–0.93) and attached (below 0) |
| 3 | Does the model predict the click outcomes (food, poison, nothing)? | vm-00017's click test, from the move token's prediction of the next body |
| 4 | Does the one-step instinct eat food and avoid poison clicks? | online food and poison clicks per 1,000 steps against the policy alone |
| 5 | Does the policy learn the instinct's choices as habit? | the policy alone (instinct off) after a lifetime with it |
| 6 | Do the agent's own samples keep exploring, without babbling? | click rate and move spread over the lifetime |

## 11. What Exists and What Is New

| Part | Status |
|---|---|
| Patch tokens, Transformer, GPU attention for long windows, lifetime with parallel bodies, shared memory, continual world with food and poison, the click test | exists (vm-00010–18) |
| One value format with masks, one shared input map, slot embeddings | new |
| Fully causal next-token order (body, patches, move) | new (the current model is block-causal per step) |
| One shared output head, K-bin two-hot cross-entropy, equal weight per token | new (replaces the per-type heads and losses) |
| Full-resolution frame prediction (each patch's own 192 values) | built (task 00023) |
| Grayscale patches, a textured poison, moving dots, the default config | built (task 00024) |
| The instinct: one-step lookahead on the predicted next hunger | new |
| Acting by the model's own samples, no babbling | new |

## 12. What Changed From the Earlier Design, and Why

| Earlier (vm-00010–16) | Now | Why |
|---|---|---|
| move loss detached from the network; optional separate clip | attached; one clip | a special path; its cause (scale) is removed by the uniform loss |
| one head per token type (frame MLP + residual, sigmoid drives, Gaussian policy with hidden layer, act-token heads, unused value head) | one shared output head | per-type heads made each type a special case |
| squared error for pixels and drives (×11 changed pixels, ×(1+w) unusual drives), likelihood for moves; four hand-set weights | two-hot cross-entropy for every value; every token weighted the same | different units forced hand-set weights; squared error drowned small effects |
| the executed move fed only into the prediction heads | the move is a token in the sequence; what follows sees it | a side channel into the heads |
| separate input embedding per type | one shared input map plus slot embeddings | per-type maps; the cost is the collision noted in §4 |
| 32×32 residual targets | each patch predicts its own 192 values at full resolution | a downsampling and a residual special to frames |
| hindsight command (a label from the future in training, the current hunger when acting) | removed; the instinct acts through one-step lookahead | the command behaved differently in training and acting |
| 20% babbling outside the model | the model's own samples | a rule outside the model |
| act token (a query trained by auxiliary losses) | removed; the policy is the patch_64 → move prediction | it held contact but not the direction (vm-00016), and was a special token |
| memory gated by hunger relief and surprise | keep everything; later, the model's own surprise | relief gating was specific to hunger |
