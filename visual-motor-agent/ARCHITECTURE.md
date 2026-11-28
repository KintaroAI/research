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

- **Candidate defaults, not applied yet (the owner, 2026-11-12):** `model.n_layer = 8` (vm-00084, vm-00086, vm-00088: K4D8 89.1 ± 8.2 food over 12 seeds) and `online.instinct_tol = 0.002` (vm-00089: K4D8T 137.3 ± 15.8, +48 ± 21 over K4D8). `dev/configs/U_default.json` still has 2 layers and 0.005; until it changes, experiments set both with `--set`. The best tolerance (0.001, 0, or one that follows the scale) is not yet measured.

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
- **Hindsight before bootstrapping** (for the meal label of task 00108; the owner, 2026-11-17: store it, try the simple
  one-step version first).
  - **The idea:** the replay memory holds what really happened after each stored tick. When a meal really came k
    ticks after a stored tick, its label can be the exact value: 0.95^k, less the penalties met on the way. That
    replaces waiting for the model's one-step predictions to carry the meal back k ticks, one tick per replay.
  - **Only past what is stored** does the model's own prediction fill in (an n-step return). The present hunger label
    already does this for 4 ticks (`online.value_n`).
  - **Gains:** the meal's value reaches far back at once, and the label carries no bias from the model's own errors.
    Try it if the one-step label learns too slowly.
- **Moving costs energy** (the owner, 2026-11-17): hunger should rise with movement, so that standing still becomes an
  option and eating all the time has a price.
  - `env.turn_cost` prices only changes of move; a cost per pixel moved (or per push, with inertia) would be new.
  - With the meal label (task 00108) such a cost needs no change to the label: it is a rise in hunger, and every rise
    is a penalty.
- **Two labels, meal and poison apart** (task 00108): one label holding both means that weighing it by hunger (need)
  also weakens poison avoidance when sated. Two predicted values, the next meal's nearness and the poison ahead, would
  let need weigh the meal alone.
- **Action repeat: decide every k ticks and hold the move** (the owner, 2026-11-17; from DQN, which held each action
  for 4 emulator frames, so that every decision was a short course).
  - **Why:** in the world with inertia (vm-00095) one push moves the body's next position by at most 1 px per axis, so
    the one-move instinct compares candidates that barely differ and overrides most moves. A push held for 4 ticks moves
    the body up to 10 px.
  - **Against the imagined horizon:** vm-00096 (task 00107) holds each candidate for 4 imagined ticks but re-decides
    every tick, at 2.4–3.6 times the acting cost. Action repeat is the cheap counterpart: real commitments, no
    imagination, and a decision only every k ticks (1/k of the acting).
  - **Two forms:**
    - (a) **The agent repeats** (a new option such as `online.act_every = k`): the world and the model still see every
      tick and learn from each (the held move is the move token); only the decision, the policy pass and the instinct,
      comes every k ticks.
    - (b) **The world skips frames** (DQN's form, an `env` option): one agent step is k world ticks; the model sees
      every k-th frame, touches and hunger add up over the k ticks, and a context of 8 steps covers 8k ticks. The time
      scale of prediction changes too.
  - **The test:** k = 4 (the inertia constant) in the world with inertia from birth, against vm-00096's IR and IH.
    - If action repeat matches IH, the commitment helped, not the imagination.
    - If IH wins, imagining the course matters.
    - Then in the world without inertia: does committing cost the agent its fine aim at food?
  - **Run in Breakout, form (a), k = 4 (vm-00118, 2026-11-26): negative.** The label orders held moves better (.688
    against .618 in the last 8 ticks before a landing), but the play falls by .065 ± .053: one decision in 4 ticks
    corrects less than four, and a paddle that never slows down cannot stay under the ball once its move is held. So
    committing does cost the fine aim, in Breakout. The test above, in the world with inertia, has not been run.
  - **A side effect to remember:** the candidates are sampled from the model's own move prediction, and with held
    moves its target is mostly a repeat, so more than half of the candidates at a deciding tick are the move being
    held (.567 against .094). No compute was saved either: the instinct still ran on held ticks.
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

**What comes first since 2026-11-26 (the owner): section 15.** The world model, its dream and its memory, measured by numbers that do not depend on the model's parts. The challenges below keep their order after that.

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

**A bootstrapped value (vm-00085).** K4's label with a tail: 4 real ticks, then γ⁴ times the model's own predicted label of the move taken 4 ticks later (`online.value_gamma`). At γ 0.9 the agent aims further (12–24 px, all six seeds) and walks more steadily, with +17 food per 1k (5 of 6 seeds, not significant) and no chain of moves: its gain over K4 matches the gain of its gait. The value is optimistic, 30–55% below the hunger that follows (K4's label: 14–25%), because the tail is the winner's estimate, a minimum over candidates and transformers, fed back. At γ 0.97 the 4 real ticks carry 11.5% of the label, the candidates' scores collapse (median spread 0 in two seeds) and food falls back to K4's. Next: 8 layers per transformer (vm-00086), then the mean of the four as the tail, refreshed on replay (phase 2).

**Depth with four quadrants (vm-00086).** Each of the four transformers at 8 layers (2.16M parameters each, against 0.97M): with K4's label every measure improves, poison and move CE in all six seeds, and food +24 per 1k (not beyond its interval with six seeds). K4D8 aims as far as V90 did at 2 layers. vm-00085's value at 8 layers stays about 43% optimistic and its candidates' scores collapse (spread 0 in two seeds); it adds nothing over K4D8.

**A fresh, shared tail (vm-00087).** `online.value_refresh`: before every update, each transformer predicts, without gradient, the label of the move actually made n ticks later in every replayed window; their mean is the tail. Long-term memory keeps the n steps past each chunk, and its real parts and tail weights instead of fixed labels. This removes the optimism: abs bias .0043 against V90's .0127 (all six seeds), about −20% of the realized label like K4. The agent aims further than K4, learns faster early and touches less poison than V90, but eats no more by more than the noise of six seeds, and builds no chain of moves. About 11% slower per tick.

**The fresh value at eight layers (vm-00088).** V90RD8 (`online.value_refresh = 1`, `model.n_layer = 8`) against K4D8 over twelve seeds, each seed pair on one card. The fresh tail stays calibrated at depth (about −26% of the realized label, like K4D8's −25%; the stored tail at 8 layers was −43% and its candidates collapsed), and the agent aims further, but it eats no more: +4.2 ± 12.4 food per 1k, poison equal, no chain of moves. K4D8, 89.1 ± 8.2 food over 12 seeds, is the reference. The value is right but its candidates differ by about .001, so the next step is the label's scale.

**The instinct's tolerance (vm-00089).** The instinct keeps the model's first sampled move unless another candidate's predicted label is lower by more than `online.instinct_tol`. Lowered from 0.005 (set in task 00023 without a measurement) to 0.002, at 8 layers and 12 seeds: K4D8 eats 137.3 ± 15.8 food per 1k against 89.1 (+48 ± 21, 11 of 12 seeds), the fresh value 129.3 against 93.3, poison no higher in either. The gain is in seeing (lead over a blind walk with a perfect grab +104 against +65) and grows with learning. A calm agent's predicted drop for a food move one step away is .001–.007, mostly below 0.005, so the old tolerance silenced most food moves. The value gains no more than K4 (−8 ± 17 at 0.002), with a little less poison.

**The value in the big world (vm-00090).** vm-00089's K4D8T and V90RD8T moved unchanged into `configs/U_big.json` (128 x 128, 4 foods, half the time the next food beyond 24 px), four quarter transformers, 6 seeds. There the value pays for the first time: 79.2 food per 1k against K4's 56.4 (+22.9 ± 10.6, 6 of 6 seeds), poison a little lower, still rising at 60k. The gain is aim at far food (past 24 px .240 against .092), which K4's 4-tick label barely ranks. Up close the value still lets food go (eaten when in reach .63 against .85): its food-move drops (.001–.003) often fall inside the tolerance. K4 at 8 layers and 0.002 beats vm-00072's 2-layer K4 at 0.005 by 14, with about one more poison. The value goes forward into a world larger than the view.

**A world larger than the view (vm-00091).** `env.view = 64`: the frame is the 64 x 64 center of the egocentric frame of a 128 x 128 world with 2 foods (`configs/U_view.json`), so food is out of sight 40–50% of the time; the model is the small world's (`img` 64). Only the frame changes. The value pays more here than with the whole world in view: VW 47.6 food per 1k against K4's 21.2 (+26.4 ± 5.9, 12 of 12 seeds), less poison in every seed, still rising at 60k. K4's jittery gait finds little once food leaves the window. Memory of food that left the view lasts as long as the 8-tick context: within it the agent turns away from that food less than chance would (+.062 above a blind baseline for VW, once ticks right after a meal are left out; counting them doubled the figure), and beyond it nothing. A seeing walker eats 96, one that sees the whole world 137: the room for memory and search.

**Two worlds, one set of weights (vm-00117, the first step of challenge 1).** The default agent's weights from a 60k-tick life in one world start a life in the other: the food world and Breakout stage 1, twelve seeds, memory and optimizer new, so the old world is never replayed. Transfer is one-sided: a food-world past helps in Breakout (+.145 ± .031 hit share at 10–30k, .5 reached about 15k ticks earlier in every seed), a Breakout past hurts in the food world (−29.1 ± 3.0 food per 1k at 10–30k, about 30k ticks behind). With learning off the old world's play is at an untrained model's level after 10k ticks of the new one, in both directions, while part of the old world's frame is still predicted. Coming back for 30k ticks, the food world is relearned no faster than a new model learns it but far faster than by an agent with a Breakout past only (+22.7 ± 3.4 food per 1k), and Breakout as fast as a food-world agent that never played it learns it. So the weights do not hold two worlds without help, and what a world leaves behind is not what a frozen test of its play shows. By the rule set beforehand the follow-up is rehearsal (one process, one memory across both worlds), then the flexible layer.

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

## 15. Next Steps: the World Model First (2026-11-26)

**The decision (the owner, 2026-11-26).** Improve the world model, its dream and its memory before concentrating on anything else. The remaining arms of the acting rule (task 00123) and rehearsal for two worlds (vm-00117's follow-up) wait. This is the order of section 1, prediction before behaviour: since vm-00085 most experiments worked on the label and the move, and the last three arms of the acting rule closed negative (vm-00114, vm-00115, vm-00118).

- **The world model:** the prediction of the next tick: the frame, the body, and what a move changes.
- **The dream:** the model fed its own predictions. Its score is noise at 4 ticks (vm-00114; that dream was decoded with the expected value, under which the ball is gone by the fourth tick: the first reference table, below). Lookahead, labels from imagined choices (task 00114) and a critic trained on dreams all need a dream that holds.
- **Memory:** what the model keeps beyond its 8-tick context. Three mechanisms gave none (vm-00092 to vm-00094), judged by foraging, where the agent is still below what needs no memory: in vm-00091 a walker that chases whatever it sees and remembers nothing ate 96 food per 1k where the agent ate 48. Memory is tested in prediction first.

**Watching first (the owner, 2026-11-26).** For now the model's own movement and its decisions are left out: it observes worlds that move by themselves, and the question is only how well it predicts them. The owner's two examples, built as task 00132:
- **Moving food and poison** (`env.kind = watch`, `dev/world/watch_world.py`, `configs/W_discs.json`): the food world's discs on a 64 x 64 torus, 2 food discs at 1.5 px a tick and 2 checker poison discs at .75, with dots under them. They never overlap: two that meet are pushed apart and each one that was closing in has its velocity mirrored, so every disc keeps its speed for ever (an exchange of momentum would mix the speeds, and the two kinds would stop differing). No body is in the frame and nothing is eaten. After the first frame nothing is left to chance. A disc flies about 13 ticks (food) or 20 (poison) between bounces.
- **The court with a script** (`env.auto_lag = 6`, `configs/W_breakout.json`): Breakout stage 1 with the paddle given to a script that follows the ball as it was 6 ticks ago. Nothing in it is random, and each of its moves follows from frames inside the 8-tick context. The delay is what makes the play worth watching: the paddle trails a slanting ball, meets it off-centre and sends it off at an angle, and it returns .80 of the balls (4 ticks: 1.00, 8 ticks: .68). A tracker without a delay returns .99 even at a ninth of the paddle's speed, and its ball goes straight up one column. The score to predict is the body's hunger, as in the court the agent plays: a hit sets it to 0, a miss adds .5.
- **The model only watches** (`online.watch = 1`): the move made is the null move at every tick, whatever the model's own move prediction says; no instinct and no label. The model is the default agent's without the label.

What this sets aside for now: measure 3 (what a move does), measure 12 (the worth to a decision) and the anchor "another move". The contract keeps its moves, which are all null. The reference table needs new lives: the weights of vm-00101 and vm-00103 were raised acting, in other worlds.

**First the measures (the owner).** Before the model is changed we need numbers that say how well a world model performs. They must not depend on the model, because its parts will be swapped all the time: the tokens, the bins, the decoder, the number of transformers, the memory. A number that changes its meaning with a part cannot compare two models. Cross-entropy is such a number: it is counted in the model's own bins, so 32 bins, 256 bins and learned patch codes give values that cannot be set side by side.

**The contract.** A measure uses only what any model can give and what the world gives.
- **From the model, three calls:** `reset()` forgets what is carried (the weights stay); `observe(frame, body, move)` takes one real tick; `dream(moves, n)` returns the next ticks under those moves as frames and body values in the world's own units (grey levels, hunger), in `n` versions if the model can draw them, else its one best guess. How it decodes is its own business.
- **From the world:** recorded stretches of real life; its own state at every tick (for scoring only); copies of itself that can be stepped with other moves or other dice; and a reader that finds its things in any frame, real or dreamed (the body, the ball, a food), written once per world and checked against the world's state on real frames.
- **Never:** the loss, the bins, the tokens, attention, the label. They stay useful to debug one model, not to compare two.
- One adapter per kind of model gives the three calls. DreamerV3's world model (`dev/baselines/`) can give them too, and so can a predictor with no model at all.

**The exam.** The same recorded stretches for every model, so that two models are compared stretch by stretch: made once per world by scripted walkers and by the reference agent, with the world's state kept. There are two sets, one to work with and one opened only for a final comparison: with many swaps, the working set gets fitted by selection. Differences are paired by seed and by stretch, with the 95% interval and the seeds above, as in the write-ups.

**The anchors.** Every number is read between floors and a ceiling that need no model:
- **copy:** the last real frame, unchanged;
- **another move:** the world itself stepped with a different move, which is right about everything but what the move does;
- **other dice:** the world itself stepped with the same moves and other chance, the best possible where the world is random;
- **the world itself:** error 0.

The skill of a prediction is 1 − its error ÷ the copy's error, both summed over the exam: 0 is no better than copying, 1 is exact.

**The measures** (proposed 2026-11-26. The owner's answer was to watch first, above, and then to go on with the plan; the measures were not gone through one by one. Built the same day as `dev/wm_measures.py` for a model that watches: 1, 2, 4, 5, 6, 9, 10 and 11. Not built: 3 and 12 wait for moves, 7 for a model that gives its dream in versions, 8 for a gap. How each is counted: `tasks/00132-watching-worlds.md`. The last column names the older tools that hold a part.)

| # | Measure | Its question | How it is counted | A part is in |
|---|---|---|---|---|
| 1 | Next-frame skill | Is the next tick predicted? | Skill in grey levels over the whole frame, and by kind of thing: the body, what moves by a rule, what appears by chance, the rest | `eval_blind.py`, `eval_clog.py` |
| 2 | Event forecast | Does it know that a meal, a poison, a hit or a miss is coming? | Events are the jumps of the real body signal (the court's score); where the world has no body, its own events (a bounce of two discs), scored by the next-frame skill on the ticks around them against free flight. The predicted body signal's hits against its false alarms (area under the curve, .5 is chance), for the next tick and within h ticks of a dream; and the predicted size of the jump ÷ the real one | the click test (`analyze_hazards.py`), `analyze_lookahead.py` |
| 3 | Move effect (waits while the model only watches) | Does the prediction change with the move as the world does? | Copies of the world stepped with K moves, for 1 tick and held for 4, on pairs of moves whose real outcomes differ. Retrieval: the prediction for move i is nearer to the real outcome of move i than to any other's (chance 1 ÷ K). Size: the predicted difference between two moves ÷ the real one | `analyze_dynamics.py`, `click-real` |
| 4 | Dream curve, dream horizon | How many ticks does the dream hold? | The dream under the moves really made, 1 to 32 ticks: its error against the copy's and against the mean frame's. Horizon: the ticks until it is no better than the better of the two | `eval_dream.py` |
| 5 | Object hold | Are the things still there, and where? | The reader on dreamed and real frames, by kind of thing: the distance in px, lost, doubled. Floor: the thing kept moving as it last moved. Horizon: the ticks while the thing is within reach in half of the dreams | `eval_dream.py`, `dream_measures.py` |
| 6 | Dream realism | Is a dreamed frame a possible frame? | The share of dreamed frames with the world's counts right (one body, one ball); the distance between the grey-level histograms of dreamed and real frames; a free dream of 200 ticks on the model's own moves: when it empties | `dream_measures.py` |
| 7 | Honesty | Does it know what it does not know? | With n versions: their spread ÷ the real error (1 is honest), and the forecast chance of an event against how often it comes | `analyze_lookahead.py` |
| 8 | Memory span | How far back does it reach? | Skill on what must be remembered, by ticks since it was last in the input: what comes back into the view, a thing wholly behind a gap (`online.blind`), a meal on hidden food. Floor: the same for what was never seen. Span: the last age above that floor | `eval_hunger.py chain`, `eval_blind.py`, `eval_hidden.py` |
| 9 | Warm-up | How much of the past does it use? | From a cold start, shown the last k real ticks, k = 1 to 64: measures 1 and 2 against k | none |
| 10 | Learning | How fast does it get there, and what is kept? | Measures 1 to 8 on a life's checkpoints: the ticks to a set level; and the same on an old world after a life in another (vm-00117) | none |
| 11 | Cost | At what price? | Ticks per second alone on the 4090, card memory, parameters | every write-up |
| 12 | Worth to a decision (waits while the model only watches) | What do its errors cost a choice? | One fixed rule choosing among K moves by the dream and by copies of the world: how often the two agree, and what the difference costs | `analyze_lookahead.py` |

- **Pixel error alone must not judge a dream.** A blurred frame scores better than a sharp one with the ball 2 px off, which is counted wrong twice: where it is drawn and where it should be. So measures 5 and 6 stand beside 4.
- **What is remembered must be wholly out of the input.** For the memory span a thing counts only while no pixel of it is in the input. Ages within the 8-tick context show the window; later ages show memory.
- **The text world keeps nats per letter** (`letter_ce.py`): there a letter is the world's own unit, and any model can give each letter a probability.
- **Frozen first, lived after.** Frozen tests repeat exactly. A frozen checkpoint understates a living learner (vm-00005), so the lived form, each tick predicted before it is learned, follows once the frozen board stands.

**The board.** Six numbers per model and world, each beside its floors: next-frame skill, event forecast, move retrieval with the move held for 4 ticks, the dream's horizon by objects, the share of valid dreamed frames at 8 ticks, the memory span; and ticks per second.

**The first reference table (vm-00119, 2026-11-26).** The default agent's model without the label, only watching, 12 seeds a world, 60k ticks; the working set of the exam; the reference decoder `pair`; ± is the 95% interval over the seeds.

| Number | The moving discs | Its floor or ceiling | The scripted court | Its floor or ceiling |
|---|---|---|---|---|
| Next-frame skill | .337 ± .042 (.558 on the life's own discs) | the copy 0, the world 1 | .667 ± .010 | the copy 0, the world 1 |
| Event forecast | the discs that turn are 0.39 px off at the next tick | the straight line 0.40 px | a hit .996, a miss .987 at the next tick; .939 and .913 8 ticks ahead | chance .5 |
| Dream horizon by pixels, ticks | 6.2 | of 32 | 22.7 | of 32 |
| Dream horizon by objects, ticks | food 3.0, poison 3.9 | a straight line through the last 2 frames: 10 and 13 | the ball 9.1, the paddle 22.9 | a straight line through the 8 frames of the window: 11 and 15 |
| Valid frames at 8 ticks | .095 | the world 1 | .752 | the world 1 |
| Real ticks of context that give all of the next-frame skill | 4 of 8 | | 4 of 8 | |
| Real ticks of context that give all of the dream | 4 of 8 | | 2 of 8 | |

- **The model knows what comes next; the dream is weakest in where the things are.** In the disc world a straight line through the last two frames holds a disc three times longer than the dream. In the court the ball is drawn at whole pixels and the dream holds it nearly as long as a line through all 8 frames of the window does, from the last 2 frames alone.
- **Half of the window is not used.** Four of the 8 ticks give all of the next-frame skill and of the dream, in the court's dream two. To hold the ball longer a model must first read the window it has, and then carry the ball's course over more ticks than it sees at once: a first task for a memory, with no gap needed.
- **A floor must know no more than a model can see.** The straight line through the true places, set beforehand, knows the ball's place to a fraction of a pixel; the lines through the frames as seen are the fair floors.
- **The decoder is a part that matters.** The same weights dreamed with the expected value, as the agent's instinct and vm-00114's lookahead dream, hold the ball in .234 of the dreams at tick 2 and .008 at tick 4; with `pair`, in .928 and .818; with the instinct's quick decoding and `pair` (`online.instinct_decode = pair`), in .866 and .654. The pixel error understates this in the court and ranks the blurring decoder first in the disc world. Dream with `pair`. vm-00114's lookahead dreamed with the expected value, so its negative result was read off a dream that, in these lives, has no ball by its fourth tick; its agent acted and had a label, and its weights are not these, so this is a reason to test the lookahead again with `pair`, not a result about it.
- **A life in the disc world learns its own four discs:** +.221 ± .047 in next-frame skill at home.
- **What the board should carry after this table:** the forecast 8 ticks ahead in place of the next tick's (which is at its ceiling), the discs held at tick 4 after a turn in place of the next-tick reading of bounces (too small to see), and the counts by zone beside the valid frames (an empty court is a valid frame).

**One transformer against the four (vm-00120, 2026-11-27).** The owner's step back after the first table: "it is expected that adding transformers will reduce accuracy". The same lives and measures with one transformer on the whole 64 x 64 frame (`configs/W1_discs.json`, `configs/W1_breakout.json`: the W configs without the split; 2.17M parameters against the four's 8.64M), 12 seeds a world, paired by seed with vm-00119's.

| Number | The moving discs: one | four | one − four | The scripted court: one | four | one − four |
|---|---|---|---|---|---|---|
| Next-frame skill | .322 ± .082 | .337 ± .042 | −.015 ± .043 | .759 ± .010 | .667 ± .010 | +.092 ± .010 |
| Next-frame skill on the life's own world | .689 | .558 | +.131 ± .021 | .760 | .699 | +.061 ± .041 |
| Skill at the 8th dreamed tick | .466 | .279 | +.187 ± .031 | .684 | .627 | +.057 ± .013 |
| Dream horizon by pixels, ticks | 8.5 | 6.2 | +2.3 ± 1.6 | 25.6 | 22.7 | +2.9 ± 1.3 |
| Dream horizon by objects, ticks | food 4.8, poison 5.0 | 3.0, 3.9 | +1.8 ± 0.5, +1.1 ± 0.4 | the ball 10.8, the paddle 28.4 | 9.1, 22.9 | +1.8 ± 1.0, +5.5 ± 1.2 |
| Valid frames at 8 ticks | .176 | .095 | +.081 ± .042 | .743 | .752 | −.009 ± .023 |
| Event forecast 8 ticks ahead (a hit, a miss) | | | | .900, .941 | .939, .913 | −.040 ± .035, +.028 ± .028 |

- **The split costs prediction.** One transformer is ahead in the court at the next frame and in the dream in both worlds, with a quarter of the parameters; a life alone runs 54 ticks per second against 40. The four are ahead on the forecast of a hit (−.010 ± .008 one tick ahead, −.040 ± .035 8 ticks ahead; their forecast is the mean of four).
- **Where the four lose:** in the court at the borders between the quarters (a ball that has crossed one); in the disc world everywhere: a disc that stayed inside one quarter, drawn by one transformer alone, is lost as fast. Each quarter transformer learns from a quarter of the frame and sees a thing whole only inside its quarter; the two are not separated.
- **In ticks of life** the four at 60k are where the one transformer was after about 17k ticks in the disc world's dream and after about 27k ticks to about 42k ticks in the court.
- **The one transformer learns its own four discs more by heart** (home − away in next-frame skill +.367 ± .087 against +.221 ± .047): new discs in the watched world matter more for it.
- **The dream's weakness is the same in kind:** a disc held for 4.8 to 5.0 ticks against 10 and 13 for a straight line through two frames.
- **Acting was not run.** The four were adopted because their competing moves forage better (vm-00051); when the model moves again that is to be measured again, and the picture can come from one transformer while several propose the moves.
- **How it was measured:** the part `seam` (the error within 4 px of a border between the quarters and away from it) and `wm_measures.py borders` (the things held by whether they have lain across a border so far; from the kept distances, no model). A band of pixels at one tick does not follow a thing that crossed it and moved on; the second reading does.

**Trained on its own dreamed frames (vm-00121, 2026-11-27).** The first change, by the owner's word: the one transformer with self-forcing (of the 7 replayed windows of an update, 2 get the inputs of their last 1 to 3 or 1 to 7 steps from the model's own predictions, decoded with `pair`; the targets real). Seeds 1–6 against vm-00120's lives by seed; the court's 7-step arm on seeds 1–2.

| Number | The scripted court: none | up to 3 steps | 3 steps − none | The moving discs, new discs: none | up to 3 steps | 3 steps − none | up to 7 steps | 7 steps − none |
|---|---|---|---|---|---|---|---|---|
| Next-frame skill | .757 | .766 | +.009 ± .016 | .253 | .319 | +.066 ± .071 | .190 | −.063 ± .200 |
| Skill at the 8th dreamed tick | .682 | .745 | +.063 ± .033 | .435 | .446 | +.011 ± .080 | .387 | −.049 ± .052 |
| Dream horizon by objects, ticks | the ball 10.8 | 16.2 | +5.3 ± 2.9 | food 4.5 | 4.5 | +0.0 ± 1.3 | 3.5 | −1.0 ± 1.1 |
| The thing held at tick 8 (the ball; the food) | .645 | .826 | +.181 ± .085 | .280 | .290 | +.010 ± .069 | .157 | −.123 ± .070 |
| Skill at the 8th dreamed tick on the life's own world | | | | .516 | .620 | +.104 ± .049 | .606 | +.090 ± .054 |
| Hit forecast within 16 dreamed ticks | .815 | .908 | +.093 ± .072 | | | | | |

- **It works where the life knows the things.** In the court every life and the exam have the same ball, and the dream holds it half as long again, past a straight line through the window's frames (11 ticks) and through its true places (15): a line cannot bounce. The next frame pays nothing.
- **The disc world's exam shows nothing of it,** and the life's own discs do: both arms dream them better. The exam's discs and dots are new to the life.
- **The disc world's exam measures two things at once:** how well a dream holds, and how a life that has seen four discs and eight dots deals with others. The second hides the first. On the next frame the dots are newer than the discs (skill on the dots .264 on the exam against .780 at home).
- **A dreamed disc falls behind on its course** (the part `course`, added here): without self-forcing a poison disc has gone .786 of its true way at tick 4 on own discs and .719 on new ones. The dream's error is a lag and a drift, rarely a loss. Self-forcing reduces the lag on own discs (.856).
- **Seven dreamed steps harm on new discs** (the food is not found in .357 of the dreams at tick 8); why is not settled.
- **Few seeds from here on** (the owner, 2026-11-27): a change is tried on 1 to 2 seeds an arm to see whether it works at all; many seeds wait until something is worth confirming.

**A longer life (vm-00122, 2026-11-27; a look on one seed an arm).** The owner's question: does it improve over time? The same lives for 200k ticks, with and without up to 3 dreamed steps, scored every 20k ticks on the exam and, in the disc world, on the life's own world after its end.

- **What the life knows keeps improving to the end.** The court's ball is held for 16 dreamed ticks at 200k against 9 at 60k, and for 26 against 15 with self-forcing (the dream is 32 ticks long). On its own discs the life holds the food at tick 8 in .516 and .523 of the dreams against .266 and .375, and the poison for 10 and 11 dreamed ticks, where a straight line through two frames holds it for 13.
- **What it has never seen does not.** On new discs the food held at tick 4 is .725 and .818 against .711 and .791 at 60k, and next-frame skill falls a little (−.048 and −.038). The gap between own and new discs widens from both sides: a longer life learns its scene more by heart. More ticks do not replace variety.
- **Self-forcing keeps its gain in the court** (+10 ticks of the ball's horizon at 200k, +6 at 60k): about 65k ticks with it do what 200k do without. On the life's own discs the life without it catches up slowly.
- **One life is a look.** Two runs of the same seed differ at 60k by up to 2 ticks of the ball's horizon and .06 of the food held at tick 8, and seed 1 is the disc world's best seed of vm-00121's six: that self-forcing is ahead on new discs in this seed is not evidence.

**Its own prediction mixed into every frame (vm-00123, 2026-11-27; the owner's idea; a look on one seed an arm).** The owner, after the long dream that lost its ball: noise in what the model sees, so that it learns to recover. Each pixel of every frame the model is shown comes from its own sharp one-step prediction of that frame with probability .5, from the real frame otherwise; the targets are real, and the mix is made again at every update (`online.mix`).

- **In the court it does what self-forcing does.** The ball is held for 15 dreamed ticks at 60k, against 9 and 9 without self-forcing and 15 and 17 with it; the next frame pays a little (.742 against .750 and .775).
- **In the disc world it harms.** The food is held at tick 4 in .555 of the dreams on the life's own discs against .781, and in .564 against .711 on new ones. Its prediction stays wrong at the rims and in the checker, and mixed with the real frame that scrambles what says where a disc is (my reading, not tested).
- **It does not mend a long dream.** In the free dream of 200 ticks the court life with the mix has a ball left in about one dream in eight at tick 200, the others in about one in two, and no dream serves a new ball. The mix trains the model to mend a frame of which half is given; a dream gives it none.
- **The noise is the model's own error.** Where the model predicts well it goes away: in the court 0.05% of the lent pixels are wrong at 60k. Noise that does not shrink (a random grey) and a larger share are vm-00124.

**The order of work.**
1. The worlds to watch and the watching option: built and tested on 2026-11-26 (task 00132, `tests/test_watch.py`).
2. The contract, the exam and the measures, with the adapter for the four transformers and the floors: built on 2026-11-26 (`dev/wm_measures.py`, `tests/test_wm_measures.py`). DreamerV3's adapter is not written.
3. The lives that watch and the reference table: done on 2026-11-26 (vm-00119, above). One transformer set against the four on it: done on 2026-11-27 (vm-00120, above).
4. The changes, each judged on that table (the first done on 2026-11-27, vm-00121; the order is the owner's; they start from one transformer, `configs/W1_discs.json` and `configs/W1_breakout.json`; each on 1 to 2 seeds an arm first, by the owner's word of 2026-11-27):
   - **the dream in the disc world:** training on the model's own dreamed frames (self-forcing; `lifetime.py` refuses it with the four transformers and runs it with one). Done as vm-00121 (above): it works where the life knows the things (the court's ball is held half as long again) and does not show on the disc world's exam, whose discs and dots are new to the life;
   - **the watched disc world:** new discs and new dots now and then, so that a life cannot learn its scene by heart. After vm-00121 this comes before anything else in the disc world: until then its exam measures the meeting with new things together with the dream. vm-00122 adds a reason: 140k more ticks of the same scene did nothing for new discs. Proposed, for the owner's word: with and without up to 3 dreamed steps;
   - **the window, then memory:** the model uses 2 to 4 of its 8 ticks. First the ball held longer by reading the whole window, then by carrying its course over more ticks than the window; then a gap that the discs and the ball pass behind (`online.blind`), with the three mechanisms of vm-00092 to vm-00094 tried again there;
   - **patch codes** in place of pixels predicted one by one (a guess, not yet tested): the numbers it should move are the things held and the valid frames;
   - **an outside reference:** DreamerV3's world model through the same three calls.
5. Then the model's own moves again: measures 3 and 12, the acting rule, rehearsal; and vm-00114's lookahead again with `pair` (its dream was decoded with the expected value, which loses the ball).
