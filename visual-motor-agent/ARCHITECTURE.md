# The Target Architecture: One Transformer, One Lifetime

**Date:** 2026-10-25
**Status:** Design; all parts built (task 00020), first test vm-00016

This is the design we are aiming for, assembled from what vm-00001–00015 taught us. Each decision names the experiment
behind it. Where a part is still a hypothesis, it says so, and §10 lists the experiments that would settle it.

## 1. Principles

1. **One network.** A single GPT-style Transformer reads everything the agent senses and produces everything it
   predicts and does. There is no CNN, no separate vision or motor network, and no planner outside it (§9: later).
2. **Prediction is the teacher.** The main training signal is the error in predicting the next state: the next frame,
   the next internal state, and the agent's own next move. Nothing is rewarded. Hunger is a sensed body variable,
   not a reward (§6).
3. **One lifetime.** One agent lives in one world that never resets, acts on every step, and learns on every step
   from its own experience. There are no separate training and acting phases.
4. **Many bodies early on, one brain throughout.** At the start, the same weights may drive many bodies in parallel
   worlds, so that early learning sees varied, decorrelated experience. The number of bodies shrinks to one (§7).
   It is always one model, never an ensemble.
5. **The model sees only what a body would sense:** the rendered RGB frame, its own drives, and its own previous
   move. True positions are for scoring only (CLAUDE.md, information hygiene).

## 2. The World It Lives In

- 64×64 RGB, a continual torus: a new circle appears after each hit; nothing resets (vm-00011: with walls the agent's
  drift sticks to an edge, and it starves).
- Hazards (blue circles that raise hunger inside), optionally moving circles, distractor dots, an egocentric view
  (vm-00011: the egocentric view made the basic agent hit every test episode by 40k).
- **Hunger rises slowly and rarely reaches its cap** (vm-00014/15). At the cap of 1, a hazard step changes nothing and
  leaves no trace to learn from: `h_rate=0.001`, `hazard_cost=0.02` (82% of hazard steps change hunger, against 28%
  at the old rates).

## 3. Tokens

Every step t becomes one block of tokens. A token is a vector of width d (128). Each kind of token has its own
learned embedding, a generalization of an LLM's embedding table to continuous inputs, plus learned position
embeddings:

```
x = W_kind · input + b_kind  +  type[kind]  +  step[t − window start]  (+ pos2d[patch] for patch tokens)
```

| Token | Count per step | Input | Its output predicts |
|---|---|---|---|
| **patch** | 64 (8×8 px each) | the patch's raw RGB, 8·8·3 = 192 values in [0, 1] | the change of its own cell of the next frame |
| **body** | 1 | the drives: hunger, restlessness | the next drives |
| **move** | 1 | the previous executed move (dx, dy, click) | — |
| **command** | 1 | the hindsight command (§6) | — |
| **act** | 1 | none (a learned query vector) | the next move (the policy), plus auxiliary predictions (§5) |

That is 68 tokens per step and 544 per 8-step window.

- **RGB, not grayscale:** in standard grayscale the red target and the blue hazard are both about 94/255.
- **Separate embeddings per kind,** not one shared projection over zero-padded vectors. With a shared projection, hunger
  enters through the same weights as pixel 0 of every patch (vm-00010's design, kept).
- **8×8 patches:** of 8×8, 16×16 and the whole frame, the token of the patch under the cursor held hazard contact best
  (R² 0.67–0.69, against 0.31–0.36 and about 0, vm-00014). The whole frame as one token loses local detail.
- **pos2d** starts as a smooth 2D sine-cosine code of the patch centre (geometry from step one), then is learned.

## 4. Attention and Order

- **Block-causal:** a token sees every token of its own step and of the earlier steps in its window, never the future.
  Order in time comes from the step embedding, exactly like an LLM's position embedding; order within a step does not
  matter, so the step is attended as a set.
- **The window** is the last K = 8 steps (working memory). It restarts every K steps in training, so acting and
  training compute the same function (CLAUDE.md, "context = block").
- **Size:** 2–4 layers, 4 heads, d = 128, MLP 512, about 0.5M parameters. On the RTX 3090, attention over windows longer
  than 32 tokens uses batched cuBLAS products: one P8 update of 8 windows takes 6.7 ms (task 00018).

## 5. Outputs and Losses

All outputs come from the token outputs of step t; the move just executed, a_t, enters the frame and drives heads,
so they predict the consequence of what the agent actually did:

```
patch token p  ──[z_p, a_t]──► MLP (shared over patches) ──► Δ cell p of frame t+1 (32×32 target, residual)
body token     ──[z_b, a_t]──► MLP ──► drives t+1
act token      ──► policy MLP (detached, hidden 128) ──► μ(dx, dy), σ, p(click)      ← the agent acts on this
act token      ──► auxiliary heads ──► drives t+1, coarse frame t+1 (8×8)            (hypothesis, §10)
```

| Loss | Weight | Why |
|---|---|---|
| frame: squared error per pixel, ×(1 + 10) on pixels that change | 1 | plain error ignores the 1%-of-pixels cursor (vm-00001) |
| drives: squared error, ×(1 + w) on steps where hunger changes unusually | 1 | rare events matter (vm-00013's weight, `drive_changed_weight`) |
| move: −log π(executed move \| window, command), into the policy head only | 1 | the move loss erodes the shared representation when it reaches it (vm-00007/8, vm-00014 P8A) |
| auxiliary on the act token: next drives, coarse next frame | 0.5 | gives the policy's input token a reason to gather the scene (§10) |

- **Why the act token needs auxiliary losses:** in vm-00010 the policy read a token that no loss trained once the move
  loss was detached, and it never aimed. Letting the move loss train the Transformer instead eroded the patch tokens
  (vm-00014). An act token trained to predict the next drives must find the cursor, the circle and the hazards,
  because those decide the next hunger. The policy reads that token through a detached head.
- **Residual frames:** the patch heads predict the change from the current frame, starting near "nothing changes".
- **The policy head is detached and has a hidden layer, with its own gradient clip** (vm-00008: detached plus a hidden
  layer was the first basic agent to aim; vm-00009: `pi_own_clip`).

## 6. How It Chooses Moves Without Reward

The policy is conditional imitation of its own past, steered by a **hindsight command**:

- **In training,** each remembered step carries, as its command token, what actually happened to hunger over the next
  k = 12 steps: `c_t = hunger_t − hunger_(t+k)` (signed: relief positive, hazard contact negative). The move loss
  teaches "moves that led to this outcome look like this".
- **When acting,** the command asks for a good outcome (the current hunger: "relieve all of it"). Moves that earlier led
  to relief are favoured; moves that led into hazards, tagged with negative outcomes, are not.
- **Exploration:** with probability ε (0.2, decaying to 0.05) the agent babbles instead: dx, dy ~ N(0, 0.6), a click
  with probability 0.1.
- vm-00008/11 showed that the one-sided form (relief or nothing) makes the basic agent aim. The signed form is new
  and still to be tested (§10). With it, the agent avoids hazards for the same reason it aims at circles.
- **No reward, no value function, no policy gradient.** Homeostatic RL (reward = drop in hunger) is kept as a
  *control* that measures what reinforcement adds, not as part of the design.

## 7. Learning: One Update per Tick, Many Bodies at First

```
every tick:
  each body b = 1..N(tick):  window_b → act token → move a_b ;  world_b steps ;  store the transition
  one gradient step on:  the newest window of every body  +  R windows replayed from memory
```

- **One update per tick** (AdamW, lr 3e-4, no weight decay). The newest window is always in the batch, so the model
  learns from what just happened; replayed windows keep it stable (vm-00006: with the newest window alone the features
  blew up to norm 10²–10⁵).
- **Parallel bodies at the start ("childhood").** N bodies live in N independent worlds and are driven by the same
  weights, each with its own window and short-term buffer. Every update sees N fresh windows instead of one: about
  N times the experience per update, less correlated, from many situations at once. Early learning is the expensive
  part: vm-00001 needed 0.5M–1M random steps before movement and clicks were predicted.
  - Schedule: N = 32 for the first 10k ticks, halving every 5k ticks down to 1 (one body from tick 30k), then one body
    for the rest of the lifetime. The weights, optimizer state and long-term memory carry straight through: it is the
    same model, now with one body.
  - Cost on the GPU: the forward pass of N windows is one batched call. The update grows with the batch: about 25 ms
    for N = 32 at 8×8 patches (estimate from 6.7 ms for 8 windows), so about 1,300 env steps/s in childhood.
  - This differs from running N independent lifetimes (separate models, as for seeds): those never share weights.
- **Bodies are interchangeable:** the model has no body identity input; body b's window is only its own past.

## 8. Memory

| Kind | What | Size | Where it comes from |
|---|---|---|---|
| **working** | the attention window | 8 steps (544 tokens) | the context |
| **short-term** | ring buffer of each body's recent steps | 256 steps per body | always replayed, recent-first |
| **long-term** | 8-step chunks kept by salience | 100k steps | vm-00007's gated memory |
| **weights** | everything learned | ~0.5M parameters | every update |

- **Long-term memory is selective, like a hippocampus.** A chunk enters when it leads to a change in hunger (relief or
  hazard contact), when it was surprising (high error predicting the whole next state: frame, drives, move), or else
  with a small base probability. In vm-00007, storing relief sped up early learning; a random memory of the same size
  learned more slowly than storing everything, so what is stored matters.
- In childhood all bodies write to one shared long-term memory: the model's memory, not a body's.
- Replayed windows are re-encoded with the current weights; nothing stale is cached.

## 9. Deliberately Left Out (For Now)

- **A planner.** The CEM planner (vm-00002–05, 12) was the best forager (135–142 hits per 1,000 steps with the
  egocentric view), but it imagines in the CNN's latent vector. In this design, imagining means rolling the
  Transformer forward on its own predicted frames: 544 tokens per imagined step, with errors that compound. Planning
  comes back later either as a teacher for the habit (vm-00009) or in a learned latent.
- **Reward, a reward token, a value head** (§6; a control only).
- **A CNN, coordinate channels, U-Net skips:** the patch tokens and pos2d take their place.

## 10. Open Questions, Each an Experiment

| # | Question | Test |
|---|---|---|
| 1 | Is hazard contact learned once it shows in the drives? | vm-00015 (running): slow hunger |
| 2 | Does the act token gather the scene when it must predict the next drives and a coarse frame? | probes of cursor, circle, contact from its output; aim |
| 3 | Does the signed hindsight command produce hazard avoidance? | in a hazard below chance (about 6%), with relief kept |
| 4 | Does childhood (N bodies, shrinking) speed up learning without hurting the single lifetime after it? | N = 32 → 1 against N = 1 for the same number of updates and of world steps |
| 5 | Does the frame loss still teach the cursor at 8×8 patches with distractor dots? | vm-00010's coupling and sensitivity analyses |
| 6 | How much does reinforcement add? | homeostatic RL control on the same model |

## 11. What Exists and What Is New

| Part | Status |
|---|---|
| Patch, body (drives), move and command tokens; block-causal Transformer; per-patch residual frame heads; drives head; detached policy with hidden layer | exists (`model.patch`, vm-00010/14), C++ and CUDA, gradients verified |
| 8×8 patches, attention over 536 tokens on the GPU | exists (task 00018) |
| Continual torus, hazards, slow hunger | exists (`env.respawn`, `env.wrap`, `env.hazards`, `env.h_rate`) |
| Gated long-term memory, short-term buffer, replay with the newest window | exists (`online.memory=gated`) |
| Changed-pixel and changed-drive weights | exists |
| **act token** with auxiliary heads (the policy reads it instead of the move token) | built (task 00020: `model.act_token`, C++, CUDA, reference) |
| **signed hindsight command** | built (`online.goal_mode=delta`) |
| **N parallel bodies with a shrinking schedule** | built (`lifetime.py`) |
| exploration ε decay | built (`online.explore_eps_end`, `explore_decay`) |
