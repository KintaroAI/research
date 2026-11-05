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
patch, and the network must separate them using the slot embedding. `model.uni_chan` (task 00075) removes this cost: the body's and the move's values get their own rows in `W_in` and `W_out`; off by default until vm-00066 reads.

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
    - Cost: scoring every move by every piece is S² one-token steps per tick, which does not stay cheap at many pieces (~13% of a tick at four pieces).
    - Lifetimes on one build (vm-00063, eight seeds, last 20k): council has a third of gain's poison with no food lost that eight seeds can show. A fixed convention for ties beats a random one: rank with ties to the lowest piece makes 33.0 food, with fair ties 24.9. Arbitrary choices should be consistent.
    - More pieces (vm-00064, eight seeds, last 20k): at 16 pieces gain makes 57.0 food against 28.7, with 5.7 poison against 3.9; council keeps its food with 0.62 poison against 1.30, and the lowest hunger of all. A piece of the sixteen drawn at random makes 53 food, so the rise is in the pieces and not in the choosing, and gain no longer cuts poison against that random piece. Why smaller pieces aim better is open. At 64 pieces, one transformer per patch (vm-00065): food stays level under gain, 56.5 against 57.0, for about 4 times the compute, so 16 stays the split to use. What gain adds on poison over a random piece goes −1.8, 0.0, +2.7 at 4, 16 and 64 pieces; council holds at 0.65 poison.
  - **Algorithms to try:**
    | algorithm | how it chooses | answers | open risk |
    |---|---|---|---|
    | gain | the lowest own predicted hunger | the default (lifetime: 28.7 ± 5.7 food, 3.90 ± 0.47 poison) | ties; the winner's curse |
    | council | every piece scores every move, the lowest mean | the curse (lifetime: **32.8 ± 10.4, 1.30 ± 0.45**, the best so far) | S² scores |
    | veto | the lowest worst score | the curse, more strictly (lifetime: 19.3 ± 5.5, 1.66 ± 0.66) | tried: council's poison, less food |
    | pessimism | the lowest mean + λ · spread of the scores | disagreement as a warning (lifetime, λ 1: 17.6 ± 6.9, 1.70 ± 0.69) | tried: as veto |
    | rank vote (Borda) | each piece ranks the moves, the ranks add | scores of different scale; a piece's bias (lifetime: `rank_first` 33.0 ± 5.7, 1.88 ± 0.61; `rank`, fair ties, 24.9 ± 6.8, 2.90 ± 1.41) | discards how much better a move is |
    | confidence on ties | gain, the sharpest move prediction on a tie | ties (frozen 43.5, 5.8; lifetime 32.1 ± 7.1, 7.77 ± 1.93) | tried: a habit spreads, poison doubles |
    | staying on ties | gain, the last winner on a tie | commitment | frozen: nothing (paths no smoother) |
    | mean of the tied | average only the proposals within a tie | ties, without arbitrary choice | cancels opposite proposals, shrinks them |
    | soft selection | a mean weighted by softmin(predicted hunger) | spans mean and gain by a temperature (lifetime at 0.01: 8.7 ± 5.1, 7.49 ± 1.46, the mean's level) | a much lower temperature is untried |
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

## 13. Beyond Prediction: the Research Plan (2026-11-01)

The model predicts well, in pixels, text and a camera. What it does not yet show is anything a reactive predictor cannot do: its context is 8 ticks, its long-term memory is only replayed in training and never recalled while acting, and a rule outside the network chooses its moves. Foraging hardly tests any of this, so better food numbers may say little. Further work should aim at what is missing, in this order. This is a judgment, not a result.

| challenge | what is missing | the question | first step |
|---|---|---|---|
| **1. One set of weights, many worlds** | "universal" so far means the same code, not the same mind | Can one life pass through circles, text and sound and come back without having forgotten? | swap worlds within a life (the worlds exist) |
| **2. Learning from one event** | a skill takes tens of thousands of ticks | Can something seen once change behaviour 1,000 ticks later? | a place or rule shown once, needed later |
| **3. Abstraction over time** | the pyramid compresses space, nothing compresses time | Can the model predict beyond its window, at the level of events and not pixels? | a pyramid over ticks |
| **4. Where goals come from** | a hand-written rule picks moves | Can the hunger rule give way to something general, such as making what happens match what is expected? | mimicry: task 00078, vm-00068, then the scrambled keyboard |
| **5. Scale** | with log summaries the cost of a tick grows nearly in line with the world (vm-00073); quality is measured only at 64 x 64 | Does quality hold when the world grows 10× and learning stays local? | vm-00072, the big world, with log summaries |

**Where hunger stands (2026-11-02, the owner's questions, vm-00069's section "After the write-up").** Behaviour at 500k is mostly a learned habit, not a drive. The bins hide nothing: the model predicts a drop of an eighth of a bin. But hunger barely changes the aim, a model told it is full does better than one told the truth (a short frozen test, all 8 seeds), nothing links a move to a food several moves later, and aim fades beyond 2–3 moves though the whole world is in view. Food's relief is only as large as the hunger of the moment and poison's rise is fixed, so poison is the louder signal. A chain needs moves scored on what happens several moves later; urgency needs hunger to change what is done. Both belong to challenges 3 and 4.

**Hunger ahead (vm-00071).** `online.future_k = k` scores a move on the mean hunger of the next k ticks, labelled in hindsight from real futures, with no new token, table or head. At k = 4 food goes from 39.0 to 67.8 per 1k ticks, no more poison, in all six seeds, and the aim at 8–12 pixels doubles, .27 to .58. It is a wider reach and not a plan: aim still fades past about 16 pixels. k = 16 reaches a little further but all but stops seeing poison (poison +6.83 ± 1.39). The price is that the body token no longer predicts next-tick hunger, so a dream's body is wrong and self-forcing is refused. The general form, every token predicting itself k ticks ahead, would keep dreams right and is the road to challenge 3.

**A bigger world (vm-00072).** At twice the side, the same density and log summaries, K4 against K1 holds: food 42.1 against 21.7, no more poison, in all six seeds. Both eat about 60% of the small world's food because the world asks for it: the next food is 24.6 pixels away against 15.3, and even a perfect chaser loses 37%. More than half of the ticks are spent past 24 pixels, where aim is nearly gone.

**Clogged channels (vm-00074).** `online.clog = p`: each transformer gets its own damaged input, black or speckled, against clean targets, with memory and acting clean. Error under long damage is .053 to .064 against .114, and error on clean input does not move. With damage while acting, food is about twice the control's and poison a third or less. Unpredicted and not yet explained: on clean input poison falls to about a third too. Two cases of the design are not tried yet: real closed eyes, and permanent damage on the target as well. How it works: section 14.

**White noise (vm-00076).** `env.noise = q` in the world, so input, targets and memory carry it. Poison breaks first, at about q = .3; food is half gone at .5 and nothing is left between .75 and .9. Every model forages best near the level it was raised on: at .5 the model raised there keeps half of the clean lead, where the clean-raised model keeps nothing. The owner's predicted maximum did not appear in three seeds; a life at q = .1 costs nothing measurable on clean frames and makes the model robust to that noise, where 10% noise quadruples the clean-raised model's poison. Six seeds at .1 (stage 2) confirm it, and still no maximum.

**Input noise, clean targets (vm-00077).** `online.noise_in = q`: the same noise on the model's input only, fresh at each update, with the world, memory and targets clean. The noise in the targets was what did the harm: at .5 the model forages more than twice as well as the one with noise in the world (food 119 against 47, poison 3.1 against 17.7), and better in the noise than the clean-raised model on clean frames (77). Lived food rises from 73.5 (q = 0) to 91.6 (.1) and 116.6 (.5), the first half of the owner's predicted maximum; where it turns down is not yet measured. Each model is still best at its own level: the .5 model takes twice the clean-raised model's poison on clean frames.

**Outage (vm-00078).** `online.off = q`: each transformer off at random q of the time, no forward pass, no move, no update, and no catching up: its ring skips the ticks it missed (ages stay in ticks) and its training masks the steps it missed. With 16 transformers a quarter off for a whole life cost nothing (food 114.6 against 114.9), and frozen with half off food fell by 2%: each transformer holds the coarse summaries of the whole view and can nearly steer alone. Unplanned: 16 transformers forage 1.64 times as well as 4 quarters, with about twice the poison. With the default 4 transformers (vm-00079) an outage costs mildly: half off at test a tenth of the food, half off for a life a quarter; a quadrant alone still keeps .86 of the team's food. Outages lower poison, as damage did.

**Delayed copies (vm-00080, the owner).** `env.delays`: the frame is four copies of the view, now and 1 to 16 ticks ago, each with its own 4 transformers, all trained to output the present. The delayed transformers predict the present nearly as well as those that see it (16 ticks back: error .039 against .236 for their stale input), perhaps by reading it from the shared coarse summaries rather than by carrying the past. Foraging did not change, lived or frozen; DX may learn faster early (+21.8 ± 20.4 at 10–20k, a hint). Blanking the delayed quarters measures blinded voters, not memory: blanking D0's present copies costs more.

**Hidden food (vm-00081).** `env.food_show = v`: the food drawn only in its first v frames. Hidden from birth, the agent never learns to forage, even at 16 frames, where forgetting costs a walker nothing: food stays at the blind walker's level. A random mover eats a still-visible food in only 15% of its meals, so the food's image never gets tied to eating. Memory must be tested in an agent that already forages.

**Food that hides gradually (vm-00083).** `online.env_sched` changes a world setting at given ticks of the life (the world lists which: `RUNTIME`). With the food seen for 20k ticks, then hidden after fewer and fewer frames (the owner's suggestion), foraging under hiding rises from the blind walker's 20 to 30–35 per 1k; a jump to 4 frames ends as well as the gradual schedule. But once the food hides, foraging stops improving, and no model aims at a just-vanished food more than at one long gone: no memory. A likely reason: the target is the next frames, where the hidden food is not drawn, so nothing asks the model to keep its place. Raised with hiding, a model forages under hiding at nearly twice the food of one raised in sight, with a fifth to a tenth of its poison.

**Context length (vm-00082).** One transformer over the whole world at ctx 8 to 32: no difference in food (all 16–25 per 1k, against 70 for 4 quarter transformers), more poison at 32, no memory at any length. The split of the view is worth far more than the context.

**Model size (vm-00084).** The same one transformer at ctx 8, deeper (4 and 8 layers) or wider (d_model 256 and 512, up to 8.8 times the parameters). Width lowers every prediction's error (move cross-entropy six-fold) but not foraging, and the widest eats about twice the poison. Eight layers forage about 40% better than two in all six seeds (29 against 21 food per 1k; not a pre-set reading); four do not. No size makes a chain of moves or remembers a vanished food: the 4-tick label, not the capacity, sets the reach.

**Log summaries (vm-00073, the default since 2026-11-02, the owner).** `model.band_win = 1`: a transformer reads a summary only within 1 cell of itself at that level, sharp near and coarse far. With transformers of a fixed size that is O(log N) summaries each. Quadrupling the area and the transformers costs 5.2 and 5.7 times per tick, against 10.5 and 26 with every summary. At 64 x 64 no loss could be seen in food, poison, reach or hunger, a mild test (7 of 21 summaries) with wide intervals. Not measured yet: quality at 128 and above, and with 16 or 64 transformers.

**Tests.** A ladder of tasks a reactive agent provably cannot solve, each with a blind control: the delayed cue (`env.cue`), then a rule that changes silently in mid-life, a place visited once and needed later, a two-step errand.

**Abstraction lives in the weights** (the owner). There is no separate store for it: what the model knows about time is in its weights, shaped by the error it reduces. The limit is the error, not the place. Next-tick prediction rewards what changes by the next tick, and an event that matters 100 ticks later sends no gradient across a window of 8. Predicting several steps ahead widens this (self-forcing does, to 3 steps), but far pixels are unpredictable and blur. So a time pyramid is not an alternative to weights: it gives them a target at a longer scale, as a summary token gives one in space.

**Length of a life.** At about 100 ms of lived time per tick (an assumption), 60k ticks is 100 minutes old. Some abilities may need far more experience. A 10M-tick life is about 35 hours alone on the 4090 with the default model, so it should be tried. Small models can mislead both ways: an idea may look dead at 1M parameters and live at scale, and a neat mechanism may not survive growth.

**A baby's training** (the owner). A child who is never taught does not learn to speak. So grow the model with an outside trainer, without reinforcement: the trainer's words and answers are only more input, and prediction does the learning. Social teaching is not reward. Steps: a scripted trainer that names what the model is looking at (pixels and text in one world), then one that answers what the model says, later perhaps a real language model as the trainer. It meets challenge 1 (several kinds of input in one life) and challenge 4 (speaking, with the owner's point that input and output are different channels: task 00077).

**Reflection is generation read back** (the owner). A language model's reasoning is one generator whose output returns as its input: a thought, then a thought about it. Dreaming into the future is the same operation, and so is saying a word out loud and hearing it. So reflection needs no new part, only the loop we already have. It can be tested: fill a gap, read the result back with what followed, fill it again, and see whether the second pass is better than the first. A small model may gain nothing from rereading: in a language model this skill was learned from a great deal of human reasoning.

**What to memorize** (the owner). Grow the model until it understands what matters, then let it choose what to keep, by surprise and by its own judgment of whether something is worth keeping. Surprise alone is not enough: noise stays surprising for ever, and our world has moving dots that should test this. Candidates for the judgment, none of them a label: **learning progress** (keep what lowers future error, not what is merely high in error), whether replaying a chunk actually helped later prediction, and, once there is a trainer, what the trainer responds to. Learning also slows on what is already predicted (arm S of vm-00068).

**Three memories** (the owner, 2026-11-03). The context is short-term memory and the weights long-term memory, with two refinements.

| | Context | Replay memory | Weights |
|---|---|---|---|
| Like | working memory | episodic memory, the hippocampus | knowledge and skills |
| Holds | the last ctx ticks, exactly | the life's steps in ctx-tick chunks, exactly | what is generally true |
| Written | every tick | every chunk, by the gate | slowly, by gradient steps |
| Read while acting | yes | no | yes, implicitly |

- **The weights hold what is repeated and stays true, specific or general.** A language model's weights hold a great many specific facts, learned because they recur in its data and stay true; facts seen once are learned poorly. What weights hold badly is the one-off and short-lived: "the food was there 50 ticks ago" is seen for a few ticks and false soon after, and the next food would overwrite it. So where this food is now is a job for the context or a fast store, which is why hidden food (vm-00081) tests working memory. (The owner's correction: a first version said the weights hold kinds, not events.)
- **The replay memory is a store that cannot be recalled.** It holds every episode exactly, but only trains the weights, like consolidation in sleep (the complementary learning systems of McClelland, McNaughton and O'Reilly, 1995). What is missing is recall: bringing a relevant past chunk back into the context while acting.
- **Online learning blurs the line.** A gradient step every tick on the newest window lets a recent event leave a trace in the weights within ticks. That is why a frozen checkpoint understates the living agent (vm-00005), and the weights may carry a little memory past the context. A test: hidden food lived against frozen.
- **Ways to lengthen the short-term memory:** a longer context (vm-00082); past steps at logarithmic spacing, recent ones dense and old ones sparse, with ages in buckets, exact when recent and coarse when old; delayed copies of the view (vm-00080).

**Two layers of weights** (the owner, 2026-11-03). A fundamental layer and a flexible one on top, the model using their sum. The flexible layer learns fast and aggressively, and is thrown out or decays, so the fundamental model is never damaged. The fundamental layer learns by its own method, from the moments chosen as most important.
- **Its lineage:** fast weights on slow ones (Hinton and Plaut, 1987; Ba, Hinton and others, 2016), a low-rank adapter on a frozen base (LoRA), and protection of important weights (elastic weight consolidation). In the brain's terms: the flexible layer is the fast hippocampal trace, the fundamental one the slow cortex, and the choice of what to consolidate is sleep.
- **The simplest form:** W = S + F. Every update trains F, at a high learning rate. Each tick F decays by a factor λ toward 0 (λ = .99 is a half-life of about 70 ticks), and a share η of it moves into S: consolidation by accumulated gradient. With η = 0 the fundamental layer is fixed and the flexible one is a pure short-lived memory. A few lines on the GPU's parameter buffers, no change to the model.
- **The fuller form, the owner's second half:** S learns only from chosen material, at a low rate. The gate already exists: chunks stored by relief or surprise (`online.store_relief`, `online.store_surprise`). In a separate "sleep" pass, S is trained on them alone.
- **What it could give:**
  - A memory between the context and the weights: F, trained every tick, may hold "the food is over there" for tens of ticks (hidden food, vm-00081).
  - Aggressive learning without drift: a higher learning rate in F alone.
  - Adaptation that does not erase: after a change in the world, F adapts and S keeps the old.
- **The risks:**
  - If F and S see the same gradient, F takes up the error first and S learns less. η and the learning rates set the split.
  - Adam's moments would be F's, so S's steps need their own scale.
  - What decays may never be consolidated: a lesson learned and forgotten within the half-life is lost.
- **Tests:**
  - Hidden food lived, with and without F.
  - A higher learning rate with and without F: does F let it be raised without collapse?
  - A world that changes mid-life and changes back, for example food and poison swapping looks (the cue world's `swap`): how fast does each adapt, and how much of the first world is kept?

**The value and the flexible layer are one idea in two directions** (the owner, 2026-11-04, after the value of vm-00085, task 00096).
- **The same math:** both are exponential weightings with a time scale of about 1 / (1 − rate).

  | | Value (γ) | Flexible layer (λ) |
  |---|---|---|
  | Formula | V_t = (1 − γ) Σ γ^i h_{t+i} | F_t = Σ λ^k Δ_{t−k} |
  | Weighs | the future seen from now | the past seen from now |
  | What fades | how much a distant outcome counts in a decision | how much a past lesson still shapes the weights |
  | What it is | anticipation | memory |
  | Time scale | γ .97: about 33 ticks | λ .99: about 100 ticks (half-life 70) |

- **The bridge, eligibility traces** (TD(λ)'s backward view):
  - every recently active weight keeps a trace that decays each tick, as F does;
  - when a surprise arrives (hunger lower or higher than the value predicted), every weight is changed in proportion to its trace, so the last few dozen ticks share the credit, the recent ones more;
  - Sutton showed this gives the same result as the forward λ-return. Looking ahead with a discount and keeping a decaying trace of the past are two views of one computation.
- **The brain's version:**
  - a synapse that was just active carries a tag that decays over seconds to minutes;
  - the change becomes lasting only if a neuromodulator, dopamine as a reward-prediction error, arrives while the tag is alive;
  - this is synaptic tagging and capture (Frey and Morris, 1997), called three-factor learning.
- **What it suggests for W = S + F:**
  - F is the trace: trained fast every tick, decaying by λ, as above.
  - Consolidation is driven by the value's surprise. In place of a constant share η moving into S every tick, F moves into S in proportion to the TD error, the gap between what the value predicted and what followed.
  - When something goes better or worse than expected, the recent trace is written into the fundamental layer. Routine leaves S alone and F simply fades.
  - This answers the risk above that what decays may never be consolidated: the lessons that mattered are the ones that are kept.
- **On hidden food the two cover each other:**
  - the value gives a reason to keep heading somewhere after the food vanishes, since food beyond 4 ticks now counts;
  - F, or the context, has to hold where it was.
  - vm-00081 and vm-00083 had neither. If vm-00085's value does not bring pursuit of hidden food, the missing part is the memory, and F is the next test.
- **The order:**
  1. vm-00085: does a value work at all?
  2. F with a fixed η: the simple two-layer form, on hidden food and in a world that changes mid-life.
  3. η driven by the value's TD error: lasting memory from consolidation that the value gates.

**TD on the whole frame: a discounted future frame** (the owner, 2026-11-04; an idea to explore later, after vm-00085).
- **The idea:** bootstrap every token, not only hunger. The target of a step is (1 − λ) × the real next token + λ × the model's own discounted prediction one step later.
  - The (1 − λ) keeps the target in 0–1; without it the values grow to 1 / (1 − λ) times the range.
  - With two-hot bins the natural target is a mixture of distributions: (1 − λ) × the real token's two-hot + λ × the predicted distribution (distributional TD, as C51). Each value is then predicted as its distribution at a random future time about 1 / (1 − λ) ticks ahead.
- **Its lineage:** the successor representation (Dayan, 1993), successor features (Barreto and others, 2017), γ-models (Janner and others, 2020), and many predictions at many time scales in one network, Horde (Sutton and others, 2011). The value is the special case on one channel, hunger, so vm-00085 is its cheapest feasibility test.
- **What each token would come to predict:**

  | Token | One-step target (now) | Discounted target (λ about .9) |
  |---|---|---|
  | hunger | the next hunger | the value (vm-00085's label) |
  | move | the next move | the mean direction of the moves ahead: where the agent is heading |
  | patches, egocentric | the next view | where things drift in view: food sliding to the centre (soon eaten), food fading (eaten or respawned) |
  | static background | unchanged | unchanged |

- **The estimate, as a replacement for the one-step frame: mostly harm.**
  - The world model is lost: no next frame, only a time-blurred mean.
  - The instinct's lookahead with H > 1, self-forcing and dreams all break, and patch cross-entropy stops being comparable.
  - Imagined frames turn to ghosts: a pixel's most likely value at a random future time is mostly background.
  - Foraging with H = 1 may not drop much, since that instinct reads only the predicted hunger. But the core of the universal model is given up.
- **The estimate, as an addition beside the one-step frame: a plausible modest help.**
  - Predicting further ahead as an auxiliary target tends to speed up RL (CPC; SPR on Atari 100k; successor features). The features would have to carry what a moment leads to, not only what comes next, which could help the value and the chain of moves.
  - Jumpy imagination, the bigger prize: a γ-model can be sampled, one sample a plausible state about 10 ticks ahead in one step. Planning far without rolling forward tick by tick, the weak point of every pixel world model.
- **The risks:**
  - The prediction is of the future under the current habits, so it moves as the agent learns.
  - It learns slowly at high λ: at λ .97 the real frame is 3% of the target, and each update moves information about one tick.
  - Bootstrapping thousands of pixel values from one's own predictions is where TD drifts (function approximation, bootstrapping and replay together). The real frame anchors it, weakly at high λ.
  - Pixels drawn each from its own distribution are noise, not a frame. A true γ-model needs coherent samples.
  - Cost:
    - Soft mixture targets need a change to the CUDA loss kernel.
    - The bootstrap needs the model's predictions one step later, from an extra forward pass (about +50%) or from stale predictions stored in memory.
- **Related here:** the column project's multi-timescale cells (`column/00014-multiscale-temporal`). No single decay captures both fast and slow patterns, but fast and slow cells together do. Here that means predicting at several λ at once: 0 (the next frame, as now), .9 and .97.
- **The order, if vm-00085's value holds:**
  1. Keep the one-step frame and add one discounted prediction (λ .9) beside it, as an auxiliary target. Read the value's accuracy, the chain of moves and lived food against vm-00085.
  2. Several λ at once.
  3. Sampling the discounted prediction for jumpy lookahead in the instinct.

**Learning while off** (the owner, 2026-11-03). Turn the outage around: a transformer taken out of acting is the one that learns. Pick one at random, stop its inference, train it for a few ticks, several updates in a row from memory, and turn it back on.
- **The ground:** with 16 transformers an outage of a quarter cost nothing (vm-00078), and with 4 a tenth of the food at half off (vm-00079).
- **What it changes:**
  - Acting and learning separate in time. A tick's acting costs only the live transformers' forward passes, so its latency drops; the updates go where the acting is not.
  - A transformer learns in bursts, not one step a tick.
  - The ones acting may not learn at all, or less.
- **An analogy:** dolphins and some birds sleep with one half of the brain at a time.
- **To measure:** lived food against the plain outage (off = no learning) and all on, at the same updates per tick in total; ticks per second.
- **The risks:**
  - A transformer coming back has changed more at once than it would have tick by tick, and the others act beside a changed neighbour.
  - Bursts on old memory may lag the world as it is now.

**Acting and learning on different GPUs** (the owner, 2026-11-03). One GPU runs only the forward passes of a life. Others train copies of its transformers on the memory it sends, and send updated weights back, transformer by transformer.
- **The lineage:** actor–learner systems in reinforcement learning (IMPALA, Ape-X, SEED RL), and today's LLM training with separate inference engines and trainers that sync weights.
- **Why it fits here:**
  - The update is about half of a tick: 6 of 13 ms with 4 transformers alone on the 4090, 19 of 35 with 16. So acting alone could run about twice as fast, and the learner could use larger batches.
  - A transformer's weights are about 1M numbers, 4 MB: a sync costs almost nothing.
  - A transformer can be off while its new weights load, which the outage showed the body tolerates.
- **The costs:**
  - The acting weights lag the learner's by a few updates. That matters less for prediction than for a policy trained on its own returns, but it should be measured: lag against food.
  - Memory has to stream from the actor to the learner.
  - Today's MPS setup already fills both GPUs with many lives, so the gain is for one large or long life, not for many small ones.
- **A first step on one machine:** the learner on the 3090 and the actor on the 4090, one life with 16 transformers, synced every N updates. Measure ticks per second and food against the same life on one GPU.

**The white ceiling and white noise** (the owner). A child who looks at a blank ceiling, where nothing happens, learns less than one who watches a world with physics in action. Noise is the same problem from the other end of the range: nothing in it can be learned. Infants themselves look longest at things of middle complexity, and away from what is too simple or too complex (Kidd, Piantadosi and Aslin, 2012). Both should lower what a model learns, and it should turn its learning down on both. Learning progress treats them as one case: on a ceiling error stays low and on noise it stays high, and in neither does it fall. To test: raise one model in an empty world, one in noise and one in the structured world, then move all three to the structured world and compare how well each predicts and how fast it catches up. A second arm lets the model choose: does it look away from either? White noise is uncommon in the real world, as the owner notes, but what a model cannot yet predict looks like noise to it, so early on much of a real world may too; speech does to a newborn. A world can be mixed with noise in stages to find where learning stops.

**Energy as the common measure** (the owner). Behaviour should be tied to energy: reducing consumption is a goal. The example: a transformer that adds nothing should do less. If it predicts too well, it works every other tick. If it does too much while others slacken, it gets tired, so that they take over. To test later, among the pieces of one body:
- **Rest:** a piece that predicts well skips ticks. The saving is measured in forward passes and updates per tick, against what prediction and foraging lose.
- **Fatigue:** a piece that wins too often is held back for a while. This may also answer the collapse to a single winner seen in the column work (ts-00025).
- **The risks:** a resting piece must still notice when its region changes, so it needs a cheap alarm, for example from the summaries it already reads. And lowest energy alone has a trivial answer, to switch everything off (the silence trap of vm-00068's design, again), so energy has to be spent to stay alive and to predict.
- **One currency:** the body's hunger is already an energy budget. If computing drew on it, thinking would make the model hungry, and saving compute would need no new rule beside the instinct.

**Imagery, and what aphantasia says about it** (the owner, 2026-11-01). Most people can recreate images, sounds and smells from memory, and this changes perception. In binocular rivalry each eye gets a different pattern, and imagining one beforehand makes it more likely to win; in aphantasia, the absence of imagery, this priming mostly disappears (Keogh and Pearson, 2018). A caution comes with it: people with aphantasia reason, plan and learn without pictures, so conscious imagery is not essential to intelligence. Thinking may run at a more abstract level, and predicting every pixel may not be what matters.

**Expectation passed between pieces** (the owner). Today pieces share only summaries of the real frame, so no piece hears what another one expects. To try: feed a piece's predicted summaries to the others, so that expectation travels as well as sight. The thought half of vm-00056 is a crude form, where predictions return as input. **An imagery score** can copy the human test: show an ambiguous frame with two patterns overlapped, prime the model with one, and measure how far its prediction leans that way, against no priming and against priming with the other.

**Exercises across senses** (the owner). Speech should be able to recreate an image, and an image a description, which should help with learning to speak. It needs no new mechanism: it is vm-00068's gap again, with a word missing and the model filling it in, now from another sense. It also gives speech a reason: vm-00068's model only continues text, while this one says a particular word because of what it sees. That is naming, how children begin. It must cross real senses (the owner): a caption drawn in the frame is still vision, and it goes when the eyes close, so restoring it from the picture is restoring vision from vision. The smallest real version (the owner: use the two worlds we have, visual and text): text as a separate sense beside vision, not drawn into the picture: a frame of two parts (`model.img_w`), a view and a text strip with its own pieces, joined only by the summaries. Letters are grey levels as in the text world, so the universal model needs no change; a small alphabet of up to about 30 letters keeps the view's 32 bins, as a full 65 needs more. A scripted teacher writes what is near ("food left", "poison near"). The world then drops each sense by itself: the text stops while the thing stays in view, or the eyes close while the text goes on, for longer than the context (the torus is the size of the view, so nothing can leave it). Built as task 00081, vm-00070. Score how well the hidden view is restored from the sound and the silent sound from the view, against a model with that other sense cut. Within a frame earlier slots predict later ones, and the scene barely changes between ticks, so each direction is taught only while the other sense is absent. This is the first rung of the baby trainer above, and sound (task 00077) and the mouth can take the strip's place later. Do this one first: the model needs no change. First result (vm-00070): words came from the picture, and words steered the body with the eyes closed, through the summaries, for the view piece too. No picture measurably came from words: what they gave reached action, not imagery. An exercise that asks for the picture itself is the open test.

## 14. How Damage Works (tasks 00085, 00086)

The owner's idea: damage what a transformer receives, score it against the truth, and the system as a whole learns to restore what does not arrive. Each transformer has its own imperfections, so the four do not share one bad picture.

### The three cases

| Case | Input | Target | Lasts | Built? |
|---|---|---|---|---|
| Temporary damage, clogged channels | damaged, different per transformer | clean | one step, or a whole window | **yes** |
| Closed eyes | blank | blank, the frame itself is blank | while closed | no: a real event in the world, nothing to restore |
| Permanent damage, a blind spot | no signal | no signal | the whole life | no: it would test tracking food through the gap |

Only the first has a clean truth to train against, so only the first is a restoration task. Black means "missing": there is no separate flag.

### One training update

1. **The windows.** Memory gives 8 windows of 8 steps, the newest and 7 replayed, all clean. Every transformer trains on the same windows.
2. **A copy per transformer.** For each transformer in turn, `clog_frames` copies the windows and damages the copy with its own draw. With 4 transformers there are 4 different damaged copies. They are drawn again at every update, so a chunk of memory comes back with different damage each time it is replayed.
3. **Which patches.** The choice is made per 8 x 8 patch, over the whole frame, including the quarters a transformer does not own. About p of the patches of each step are hit: half of that is **long**, the same patches in all 8 steps, and half is **brief**, one step each, drawn step by step. `brief_rate` allows for the two overlapping, so the share comes out at p.
4. **What happens to a patch.** Each chosen patch is black or speckled, a coin flip per patch, held through its window. Black is all zeros. Speckle replaces about half of the patch's pixels with random brightness. Pixels are never hit one at a time.
5. **Input tokens.** From the transformer's damaged copy: its patch tokens from its own quarter only, and its summary tokens averaged from the same copy, level by level. So damage in another quarter reaches it only through summaries, thinned by the averaging. Body and move tokens are not damaged.
6. **Targets.** From the one clean frame, patches and summaries alike, for all four transformers. The loss is the usual one, and nothing tells the model which patches were hit.

### While acting

`online.clog` alone damages training only. Acting and memory are clean, and damage is practice. `online.clog_act = 1` (vm-00075, the owner) makes it part of life:

| | `clog` | `clog` and `clog_act` |
|---|---|---|
| Acting during a life | clean | **damaged, each transformer its own** |
| A transformer's ring, its context | clean | **damaged** |
| The world, food and hunger | clean | clean |
| Memory | clean | clean |
| Training input | fresh damage | fresh damage |
| Training targets | clean | clean |

Acting works tick by tick, so it uses `ClogStream`: the long patches and the kind of each patch are drawn every 16 ticks, all at once, and the brief ones every tick. It has the same p as training, and its own generator per transformer.

Memory stays clean because the targets come from it. Storing what each transformer saw as well would cost five copies of each frame and replay the same damage each time. So the damage seen while choosing a move does not match the damage drawn when that moment is replayed: memory records what happened, not how it looked.

### Measuring it

- `eval_clog.py score`: acts on clean, and shows the same window clean, with brief damage and with long damage. Predictions of the chosen patches are scored against the clean frame, on all pixels and on those holding something, beside copying the last frame.
- `eval_hunger.py chain --clog .3`: each transformer acts on its own damaged view, on any run, including one that never trained on damage.
- `clog_video.py`: the world, what each transformer gets in its own quarter, and what they predict, with the same damage for each run.

### What it has shown

vm-00074, against K4: error under long damage .064 (10%) and .053 (30%) against .114, clean error unchanged. With 30% damage while acting, food 92.7 and 77.7 against 45.2, poison 4.1 and 1.8 against 13.2. On clean input poison falls to about a third, which nothing predicted and nothing yet explains. vm-00075, damage while acting too: no cost could be measured. Lived food over the last 20k is 93 and 97, the final models are level with or above vm-00074's, and clean poison stays at about a third to two fifths of the control's. The predicted early cost did not appear.

### Where it lives

`online.py`: `clog_masks`, `brief_rate`, `damage`, `clog_frames`, `ClogStream`. `lifetime.py`: `damaged` hands each transformer's copy to `update_rollout(inputs=...)`, and `views` hold the acting streams. `tests/test_clog.py`.
