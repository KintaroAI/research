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

## 3. The World

- 64×64 RGB, a continual torus: a new red circle appears after each one eaten; nothing resets (vm-00011).
- Food: clicking the red circle relieves hunger. Poison: clicking a blue circle adds 0.5 to hunger
  (`env.hazard_click_add`, vm-00018); contact alone costs nothing.
- Slow hunger (0.001 per step) so that hunger rarely sits at its cap of 1, where changes leave no trace (vm-00015).

## 4. Tokens: One Format for Everything

Every token carries a **value vector** of V = 192 numbers in [0, 1], with a mask for the numbers it does not use:

| Slot | Per step | Values (in [0, 1]) |
|---|---|---|
| body | 1 | hunger, restlessness (2 used) |
| patch 1..64 | 64 | an 8×8 RGB patch of the 64×64 frame, row by row: 8·8·3 = 192 (pixels / 255) |
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
  Longer lookahead is a later step (§9).

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
| weights | everything learned | about 1.3M parameters (0.8M of them the output head) |

- To begin with, long-term memory keeps every chunk (vm-00007's uniform control). A universal salience rule is the
  model's own surprise: the loss of the chunk's tokens, which counts every kind of token the same way. Gating by
  hunger relief is a special case and is dropped.

## 9. To Do Later

- **Learned discrete tokens (a VQ codebook):** each patch becomes an index into a learned vocabulary, as an LLM's tokens
  are, and the model predicts a softmax over the codebook. It turns the value vectors into true tokens; the risk is that
  small things like the cursor are lost in the codebook. After the universal model works on value vectors.
- **Longer lookahead:** the instinct evaluates a sequence of moves by rolling the model forward (the next body token
  after several moves). Needs the next frames generated patch by patch, or a cheaper latent.
- **Salience-gated memory** by the model's own surprise.
- **Other instincts** (restlessness) through the same one-step lookahead on other body values.

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
| Full-resolution frame prediction (each patch's own 192 values) | new (the current model predicts a 32×32 residual) |
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
