# Architecture Review: the Universal Model as Built

**Date:** 2026-10-29
**Code:** `master` at `ab09c39` (after task 00052). Links are to `dev/` at that commit; line numbers may drift.
**Scope:** what the universal model *is* today: the token format, the summary pyramid, the network, the loss, acting,
learning and memory, every parameter tensor and every buffer the model keeps, with links to the code.
[`ARCHITECTURE.md`](ARCHITECTURE.md) is the design and its reasons; this file is the implementation. §11 lists what
the review found.

## 1. The Whole Loop in One Picture

```
 world (Python)                       model (C++ on the CPU, the same model in CUDA on the GPU)
 ──────────────                       ─────────────────────────────────────────────────────────
 frame 64x64, drives  ─► tokens per step: [body][patch 0 .. patch 63, summaries between][move]
 (hunger, restless.)         │  each token: V values in [0, 1]
                             ▼
                      two-hot over K bins ─► W_in ─► + b_in + slot[s] + wpe[step]       (embedding, d = 128)
                             ▼
                      2 pre-LN Transformer layers, causal over the 8-step window,
                      masked by the allow table (window, pyramid) when set
                             ▼
                      LayerNorm ─► W_out ─► K logits per value of the NEXT token        (one head for all)
                             ▼
                      loss: two-hot cross-entropy, mean over the tokens that have a target

 acting:   the last patch's (or global summary's) prediction of the move = the policy; sample 16 candidates,
           the instinct keeps the one whose predicted next hunger (move token's output) is lowest
 learning: every tick one AdamW step on [newest window of each body] + [windows replayed from long-term memory]
```

Every world (the circle worlds, the text world) goes through this same path. Only the world decides what is drawn in
the frame ([`world/circle_world.py`](dev/world/circle_world.py), [`world/text_world.py`](dev/world/text_world.py)).

## 2. Tokens: the Slot Layout

A **step** is one tick of the world. Each step becomes `nt` tokens, one per **slot**. Slot order comes from one
function, [`vm_uni_layout`](dev/src/vm_agent.h#L91), which writes a table of `VMSlot {kind, idx, lev, r, c}`
([`vm_agent.h:78`](dev/src/vm_agent.h#L78)):

| kind | slot | values (V = 64 for an 8×8 grey patch) | used |
|---|---|---|---|
| 0 body | 0 | hunger, restlessness | 2 |
| 1 patch | 1 … | the patch's pixels / 255, row by row inside the patch (`gray`: one channel) | V |
| 2 summary | after the last patch of its block | none, or its block's thumbnail (`sum_thumb`) | 0 or V |
| 3 move | nt − 1 | (dx + 1)/2, (dy + 1)/2, click (0/1) | 3 |

- **Values** are in [0, 1] by fixed ranges, never by data statistics. Unused entries are 0, and the number used per
  token is known, so the loss masks the rest. The body and the move use the first 2 or 3 entries of the same
  V-vector as a patch.
- **Without summaries:** `nt = 66` (body, 64 patches, move), 528 tokens per 8-step window.
- **The pyramid, `uni_pyr = L`** (task 00048): level ℓ = 1..L has one summary per 2^ℓ × 2^ℓ block of patches. For
  the 8 × 8 grid and L = 3 that is 16 + 4 + 1 = 21 summaries, so `nt = 87` and 696 tokens per window. A summary sits
  right after its block's last patch, finer levels first:

  ```
  … p(1,0) p(1,1) [B1(0,0)] p(1,2) p(1,3) [B1(0,1)] …            row 1 ends two 2x2 blocks
  … p(3,3) [B1(1,1)] [B2(0,0)] …                                  a 2x2 and a 4x4 block end at p(3,3)
  … p(7,7) [B1(3,3)] [B2(1,1)] [B3] [move]                        the global summary B3 predicts the move
  ```

- **Row summaries, `uni_sum = R`** (task 00047): one summary after every R rows, same rules. It is the older
  alternative and excludes the pyramid.
- **Slot counts:** [`vm_uni_nsum`](dev/src/vm_agent.h#L79), [`vm_uni_nt`](dev/src/vm_agent.h#L86). A summary's pixel
  block: [`vm_uni_block`](dev/src/vm_agent.h#L109). The same table in Python: [`agent_lib.uni_layout`](dev/agent_lib.py#L74),
  and [`torch_reference.uni_slot_table`](dev/reference/torch_reference.py#L69). Every tool finds patches through it
  (`layout["patch"]`), never by "slot k + 1 = patch k".
- **Text** ([`configs/T_text_B.json`](dev/configs/T_text_B.json)): a 16 × 16 page of 2 × 2-pixel letters, `patch = 2`,
  so V = 4 and the grid is again 8 × 8 (64 letters per step). The same layout and summary rules apply; a letter's
  value is its brightness, letter k of n at grey k·255/(n − 1).

## 3. The Input: From Values to an Embedding

The token embedding is a sum of four parts, built in [`seq_fwd_uni`](dev/src/vm_agent.cpp#L1362) (C++) and
[`run_uni`](dev/src/vm_cuda.cu#L1856) (CUDA):

```
x[t, s] = W_in · twohot(values[t, s]) + b_in + slot[s] + wpe[t − window start]
```

| part | what it is | shape (8×8 grey image, K = 32) |
|---|---|---|
| `twohot(values)` | each used value split over the two nearest of K bins (bin centres k/(K − 1)), weights summing to 1; unused values all zeros. [`uni_twohot`](dev/src/vm_agent.cpp#L1306), [`k_uni_twohot`](dev/src/vm_cuda.cu#L1568) | V·K = 2,048 |
| `W_in` (`uni_in_w`) | one column per (value position j, bin k). The 2 · (used values) nonzero inputs pick and weight that many columns, so an 8×8 patch sums 128 weighted columns | d × V·K = 128 × 2,048 |
| `b_in` (`uni_in_b`) | the same bias for every token | d |
| `slot[s]` (`uni_slot`) | one learned vector per slot. It is what tells a token's kind and position. At birth: noise (0.02) plus the 2-D sine-cosine code of the slot's centre (0.1), [`vm_uni_center`](dev/src/vm_agent.h#L117): a patch's cell, a summary's block centre, none for body and move (task 00052) | nt × d |
| `wpe[t]` | one learned vector per step position inside the window (0..7) | ctx × d = 8 × 128 |

- `W_in` and `b_in` are **shared by every slot and every world**. The same value at the same position j picks the
  same columns whether it is pixel j of a patch, the hunger (j = 0 of the body) or dx (j = 0 of the move). The slot
  vector separates them.
- `in_bins = 0` (older runs): `W_in` d × V reads the centred values (value − 0.5) directly.
- Init ([`model_init_params`](dev/src/vm_agent.cpp#L887)): `W_in` 0.02 with `in_bins`, `uni_slot` 0.02 plus the
  codes, `W_out` 0.02, the Transformer as GPT-2.

## 4. The Network

Two pre-LN Transformer blocks (`d_model` 128, 4 heads of 32, `d_ff` 512, GELU), then a final LayerNorm:
[`transformer_fwd`](dev/src/vm_cuda.cu#L1448) (CUDA), `transformer_fwd` in [`vm_agent.cpp`](dev/src/vm_agent.cpp)
(C++). Per block: `x += W_o · attn(LN1(x))`, `x += W_2 · GELU(W_1 · LN2(x))`.

**Who attends to whom.** All 8 steps of a window are one causal sequence: token i reads tokens 0..i of its window.
When one of the masks below is set, the table `allow[q, k]` over slots additionally decides whether slot q may read
slot k. It is applied by slot, whatever the step, so the same rule holds for keys in earlier steps. It is built once
by [`vm_attn_allow`](dev/src/vm_agent.h#L146) and copied to the GPU.

| mode | rule (body and move always global) |
|---|---|
| full (default) | no table: every earlier token |
| `attn_split` S (task 00044), `attn_checker` (00046) | regions of patches; tokens meet if their regions do |
| `attn_win` W (00047) | patch ↔ patch only within W rows and W columns (3 × 3 for W = 1) |
| `attn_win` + `uni_sum` | as above; every patch reads every row summary; a summary reads its band's patches |
| `attn_win` + `uni_pyr` (00048) | a token reads a summary of its own or a coarser level if its position, projected to that level, is within W of the summary's: at most 3 × 3 summaries per level. A summary reads every patch of its block directly. No summary reads a finer one |

With the pyramid and two layers, any patch reaches any other in two hops: patch → summary (layer 1) → reader
(layer 2). The cost per window grows as N·log N (5% of full attention's scores at 32 × 32, 44% at 8 × 8; vm-00041).
**On the GPU the table is a mask, not a saving:** windows longer than 32 tokens compute the full Q·Kᵀ with batched
GEMMs ([`attention_fwd`](dev/src/vm_cuda.cu#L1071)) and set the disallowed scores to −∞ in the softmax.

## 5. The Output and the Loss

**One head for every token:** `logits = W_out · LN_f(x) + b_out`, K logits for each of the V values
([`uni_head_row`](dev/src/vm_agent.cpp#L1399)). A value's prediction is a softmax over its K bins. The expected
value is Σ p_k · k/(K − 1).

**What each token predicts:** the values of the next token. The table holds for every step t:

| token | its target | meaning |
|---|---|---|
| body_t | patch 0 of t | the frame begins, from the past and the drives |
| patch / summary i | the next slot (a patch, or a summary's thumbnail) | the frame, token by token |
| the token at slot nt − 2 (last patch; with the pyramid, B3) | move_t | **the policy** |
| move_t | body_{t+1} (the drives after the step) | **the consequence for the body**; the instinct reads it |

- [`k_uni_tokens`](dev/src/vm_cuda.cu#L1553) builds each token's input and target in one pass. The move token's
  target is the stored next drives (`ndrv`).
- **The loss** ([`seq_loss_uni`](dev/src/vm_agent.cpp#L2096), [`k_uni_loss`](dev/src/vm_cuda.cu#L1602)): two-hot
  cross-entropy per used value. A target between two bin centres puts weights on both, so changes smaller than a bin
  (hunger, 0.001 per tick) still move the loss. A token's loss is the mean over its used values. The batch loss is
  the mean over tokens that have a target: `1 / (B · (nt − nnot))`. Every token counts the same; there are no
  per-type weights.
- **Tokens without a target:** without thumbnails a summary has no values, so the token before it has nothing to
  predict (`nnot = nsum`, [`vm_uni_untargeted`](dev/src/vm_agent.h#L127)). It is the one exception to the rule.
  `sum_thumb = 1` removes it.
- **Thumbnails** (`sum_thumb`, task 00049): a summary's values are its block box-averaged to one patch (a level-ℓ
  pixel is the mean of a 2^ℓ × 2^ℓ square). [`uni_values`](dev/src/vm_agent.cpp#L1317) in C++, `uni_value_` in
  CUDA, [`torch_reference.uni_thumbs`](dev/reference/torch_reference.py#L508). The block ends before the summary, so a
  thumbnail holds only pixels already seen.
- **Gradients flow everywhere.** Nothing is detached. One global gradient-norm clip (`max_grad_norm` 0.5), then
  AdamW (lr 3e-4, β 0.9/0.999, no weight decay): [`k_clip`](dev/src/vm_cuda.cu#L682),
  [`k_adamw`](dev/src/vm_cuda.cu#L688), [`vmc_update`](dev/src/vm_cuda.cu#L2345).

## 6. Parameters

The flat parameter array is shared by C++ and CUDA, with the same tensor order (`ParamId` in both files,
[`vm_agent.cpp:502`](dev/src/vm_agent.cpp#L502)). The universal model uses these tensors; the older models' tensors
have size 0 ([`model_layout`](dev/src/vm_agent.cpp#L766)).

| tensor | shape | 8×8 grey image (K 32) | text B (V 4, K 256) | role |
|---|---|---|---|---|
| `uni_in_w` | d × V·K | 262,144 | 131,072 | input map: an embedding per (value position, bin) |
| `uni_in_b` | d | 128 | 128 | input bias |
| `uni_slot` | nt × d | 8,448 (66 slots); 11,136 (87, pyramid) | the same | slot embeddings |
| `wpe` | ctx × d | 1,024 | 1,024 | step embeddings |
| `ln1w/b, ln2w/b` | L × d each | 1,024 | 1,024 | LayerNorms |
| `qkvw/b` | L × 3d × d (+3d) | 99,072 | 99,072 | attention projections |
| `attprojw/b` | L × d × d (+d) | 33,024 | 33,024 | attention output |
| `fcw/b`, `fcprojw/b` | L × d_ff × d, L × d × d_ff (+ biases) | 263,424 | 263,424 | MLP |
| `lnfw/b` | d | 256 | 256 | final LayerNorm |
| `uni_out_w` | V·K × d | 262,144 | 131,072 | output head |
| `uni_out_b` | V·K | 2,048 | 1,024 | output bias |
| `logstd` | 2 | 2 | 2 | left over from the Gaussian policy; unused |
| **total** | | **932,738** (pyramid: 935,426) | | |

56% of the parameters sit in the two maps of the value/bin vocabulary (`uni_in_w`, `uni_out_w`), 43% in the
Transformer.

## 7. Buffers: What Each One Stores

### 7.1 Layout tables (built once per model)

| buffer | where | contents |
|---|---|---|
| `uslot` | C++ `Model` ([`vm_agent.cpp:713`](dev/src/vm_agent.cpp#L713)) | the slot table `VMSlot[nt]`, read by `uni_values`, the init and the loss |
| `uslot` | CUDA `Model` ([`vm_cuda.cu:1041`](dev/src/vm_cuda.cu#L1041)) | the same on the device, 6 ints per slot (kind, index, block origin y, x, pooling fy, fx; fy = 0: no thumbnail) |
| `allow` | both `Model`s | `[nt × nt]` bytes, the attention table of §4; empty / NULL = full attention |

### 7.2 Parameters and optimizer state

| buffer | where | contents |
|---|---|---|
| `params`, `grads` | C++ `Model` | the flat parameter array and its gradient (host) |
| `wt` | C++ `Model` | transposed copies of the matrices (`tlist`), refreshed after each step, so that every product is an NN GEMM |
| `m`, `v` | C++ `Model` | AdamW moments (CPU training only; saved with the optimizer) |
| `P`, `G`, `m`, `v` | CUDA `Model` | the same on the device. During a GPU lifetime the device copy is the live one; `sync_to` copies it back before a checkpoint |

### 7.3 The batch on the device (CUDA `Model`)

| buffer | contents |
|---|---|
| `img` (u8), `drv`, `act`, `ndrv` | the S windows × T steps: frames, drives, the moves made, the drives after each step (the move tokens' targets) |
| `pos`, `start`, `tstart`, `tend` | each step's position in its window (`wpe` index); each token's first and last key |
| `iimg`, `idrv` (`has_in`) | self-forcing only ([`vmc_set_uni_inputs`](dev/src/vm_cuda.cu#L1300)): the inputs to use instead of the batch's, with dreamed last steps; the targets stay the real batch |

### 7.4 Activations of one pass (CUDA `Model`, rows = S · T · nt tokens)

| buffer | shape | contents |
|---|---|---|
| `u_in` | rows × V | each token's centred values (unused 0) |
| `u_hot` | rows × V·K | their two-hot bins: the input of `W_in` |
| `u_tgt` | rows × V | each token's target: the next token's values (the move: the next drives) |
| `xin` | rows × d | the embedding: `W_in` · input + `b_in` + slot + step |
| `ln1`, `ln1m`, `ln1r` | L × rows × d (and × 1) | first LayerNorm per layer: output, mean, 1/std |
| `qkv` | L × rows × 3d | queries, keys, values |
| `att` | L × rows × heads × window | attention weights. **The largest buffer:** it grows with the square of the window (528 or, with the pyramid, 696 keys per row). The 4-step lookahead runs up to 256 candidate windows per pass, which is most of a frozen test's ~10 GB |
| `atty` | L × rows × d | attention output before the projection |
| `res1`, `res2` | L × rows × d | the residual stream after attention and after the MLP |
| `ln2*`, `fch`, `fchg` | L × rows × (d, d_ff) | second LayerNorm; MLP pre-activation and after GELU |
| `lnf`, `lnfm`, `lnfr` | rows × d | final LayerNorm: the outputs the head reads |
| `u_log` | rows × V·K | head logits; in a training pass, overwritten in place by their gradients |
| `u_val` | rows × V | expected values (forward passes for acting and analysis) |
| `u_mode` | S × nt × V | one step's most probable (`mode`) or pair-decoded (`pair`) values: the dream and lookahead decoding ([`vmc_forward_uni_step`](dev/src/vm_cuda.cu#L2164), [`k_uni_mode_step`](dev/src/vm_cuda.cu#L1714)) |
| `u_ce` | rows | each token's cross-entropy in the last update: the surprise signal ([`vmc_uni_token_ce`](dev/src/vm_cuda.cu#L2189)) |
| `zm` | 2 · S × d | acting passes: only each window's last patch and move outputs go through the head (1/33 of the work) |
| `dz`, `dres`, `dln`, `datty`, `dqkv`, `dfch`, `dfchg`, `datt` | as their forward buffers | gradients during the backward pass (reused across layers) |
| `acc` | 34 doubles | loss accumulators: per predicted kind (next body, patch, move) the cross-entropy, the squared error of the expected value, the count |
| `clip` | 1 float | the gradient-clip factor AdamW applies |

### 7.5 The instinct's candidates (CUDA `Model`, rows = S · M candidates)

| buffer | contents |
|---|---|
| `c_in`, `c_hot` | each candidate move as a move token (centred values, two-hot bins) |
| `c_x`, `c_ln*`, `c_qkv`, `c_att`, `c_res`, `c_fch*` | one move-token row per candidate through the layers. Its attention reads the policy pass's cached keys and values (`qkv`), so the window is not recomputed ([`vmc_uni_candidates`](dev/src/vm_cuda.cu#L2198)) |
| `c_lg`, `c_val` | the head on those rows: each candidate's predicted next body; the instinct reads value 0, the hunger |

The C++ path keeps the same quantities per sequence in a `Workspace` ([`vm_agent.cpp:612`](dev/src/vm_agent.cpp#L612):
`u_in`, `u_tgt`, `u_use` (used values per token), `u_row`/`u_prob` (one token's logits and probabilities)). Buffers
are `Buf` objects that grow on demand and free themselves ([`vm_cuda.cu:980`](dev/src/vm_cuda.cu#L980)); an
allocation failure prints "out of device memory".

### 7.6 Experience (host, Python)

| buffer | where | contents | size (defaults) |
|---|---|---|---|
| `Replay` (short-term memory, `stm`) | [`online.py:92`](dev/online.py#L92), one per body | the recent stream, one row per step: `img`, `nimg` (the frame after), `drv`, `ndrv`, `pa` (previous move), `pr`, `es` (episode start), `act`, `rew`, `done`, `hit`, `cmd` | 256 steps |
| `ChunkStore` (long-term memory) | [`online.py:224`](dev/online.py#L224), shared by all bodies | whole 8-step chunks with the same fields, plus why each was stored (`reason`), whether relief followed and when it ended | 100k steps = 12.5k chunks |
| `GatedMemory` | [`online.py:254`](dev/online.py#L254), one per body | the gate between them: per stm step its surprise and which token kind caused it; the recent surprise history; the store counts | – |
| rollout | [`cuda_lib.Rollout`](dev/cuda_lib.py#L89) | the update's windows, [T, N], before the copy to the device | batch × 8 steps |

## 8. One Tick

[`lifetime.py`](dev/lifetime.py#L171), on the GPU:

1. **Act** ([`uni_act`](dev/online.py#L539)). Each body's window is its last ≤ 7 real steps plus the current one
   (the current move slot zeroed). One forward pass returns the bins the move's predictor gives for the current move;
   M = 16 candidates are sampled from them.
   - **Instinct (`instinct_m` 16):** each candidate becomes the move token, and the move token's prediction of the
     next hunger is read (`uni_candidates`). The lowest wins. The first sample is kept unless the best is more than
     `instinct_tol` (0.005) lower.
   - **Lookahead (`instinct_h` H > 1, [`uni_lookahead`](dev/online.py#L456)):** each candidate is held for H steps,
     the steps after the first imagined by the model itself. A dreamed body comes from the move token's prediction, a
     dreamed frame from J Jacobi passes, decoded `mean`, `mode` or `pair`. The score is the hunger after the H-th step.
2. **The world steps.** The transition goes into each body's `stm`. Every 8 steps the gate decides whether the chunk
   that ended goes to long-term memory ([`decide`](dev/online.py#L323)): warmup (the first 1,000 steps), relief
   (`store_relief`), surprise (`store_surprise` q: the chunk's largest token cross-entropy, relative to its slot's
   recent mean, is in the top fraction q; [`note_tokens`](dev/online.py#L306)) or a base draw (`store_base`, 1 =
   keep everything). Once memory is full, a stored chunk replaces the oldest one (`ltm_evict` oldest) or a random one
   (`reservoir`).
3. **Learn.** One update on the newest window of every active body plus replayed chunks, `batch` = 8 windows with one
   body. With self-forcing (`sf_windows`, task 00039), some replayed windows get their last 1..`sf_steps` steps'
   inputs from the model's own dream ([`dream_inputs`](dev/online.py#L508)), while their targets stay real. Then the
   clip and AdamW.

**Parallel bodies** (`online.bodies`, `bodies_hold`, `bodies_halve`, [`bodies_at`](dev/lifetime.py#L59)): the same
weights act for N bodies in N worlds, and N halves on a schedule down to 1. Long-term memory is shared. The current
configs use one body.

## 9. Checkpoints and Equivalence

- **Header** (256 ints): the config in fixed slots. Universal fields: 40 `universal`, 41 `bins`, 42 `gray`, 43
  `in_bins`, 44–48 the regions, 49 `attn_win`, 50 `uni_sum`, 51 `uni_pyr`, 52 `sum_thumb`
  ([`vm_agent.cpp:2543`](dev/src/vm_agent.cpp#L2543)). Then the flat parameters.
- **Three implementations, one model:** C++ (CPU, the reference for training on the CPU), CUDA (lifetimes and
  tests), and float64 PyTorch ([`reference/torch_reference.py`](dev/reference/torch_reference.py#L531), forward and
  loss). Tests compare the loss and every gradient (C++ against the reference about 1e-7; GPU against C++), plus the
  candidates, the lookahead, self-forcing and the dream step, for every layout (full, regions, window, rows,
  pyramid, thumbnails): `tests/test_cuda_model.py`, `test_hier_attention.py`, `test_pyramid_attention.py`,
  `test_summary_thumbs.py`, `test_slot_positions.py` and others (1,449 tests).

## 10. Defaults (`configs/U_touch_ego.json` and the code's defaults)

| | value |
|---|---|
| world | 64 × 64 grey egocentric torus; food solid, poison checkered; 8 grey distractor dots; touch to eat; hunger +0.001 per tick |
| model | `patch` 8, `gray` 1, `bins` 32, `in_bins` 1, `ctx` 8, d 128, 2 layers, 4 heads, d_ff 512 |
| learning | lr 3e-4, clip 0.5, batch 8 windows (newest + 7 replayed), long-term memory 100k steps (keep all) |
| acting | instinct 16 candidates, horizon 1 (H = 4 with `pair` decoding in the lookahead arms) |
| text (`T_text_B`) | the same, with `patch` 2 on a 16 × 16 page (V = 4), `bins` 256 |

## 11. Findings of This Review

**What is universal, as intended:**
- One token format, one input map, one head, one loss, one gradient path for every token and every world. Text and
  images run the same code. The only per-kind facts are in the data (how many values a token uses) and in the slot
  vectors.

**What still treats tokens differently** (by design or not yet removed):
1. **The move token's target** is the next step's drives, not the next slot. The rule "predict the next token"
   wraps around a step: after the move comes the next body. It is the same rule, written out once.
2. **Summaries without thumbnails** leave the token before them without a target (`nnot`). Thumbnails remove it.
3. **The instinct reads one named value**, the hunger (value 0 of the body), outside the network. That is the design
   (goal 3: instincts act through the model's own predictions).
4. **Surprise labels:** `lifetime.py` labels tokens for the gate's counts as [patch …, move, body]
   (`kinds`, [`lifetime.py`](dev/lifetime.py#L135)), so a token that predicts a summary's thumbnail counts as
   "patch". The surprise itself is per slot and unaffected.

**Costs and inefficiencies:**
5. **Masks do not save compute or memory on the GPU.** Every masked mode computes the full Q·Kᵀ and stores the full
   weight matrix (`att`). Real N·log N cost needs a block-sparse kernel. The square-sized `att` is also why a 4-step
   frozen test needs ~10 GB (two per GPU at most).
6. **Long-term memory stores `nimg`** (the frame after each step) for the older models. The universal model never
   reads it (its targets come from the next stored step), so it doubles host memory: 100k × 64 × 64 × 3 bytes =
   1.2 GB, unused.
7. **`logstd`** (2 floats) is still allocated, unused.

**Limits of the current position scheme:**
8. **Slot vectors are absolute,** one per slot. A trained model cannot be run on a larger grid, and a new layout
   needs new slots. Factorized (row + column + level) or computed (2-D RoPE) codes would lift both limits, the same
   for every world.
9. **The 2-D code at birth was misplaced** in every model with summaries until task 00052 (cell k's code on slot
   k + 1): vm-00040 to vm-00044's summary arms. Fixed and tested; vm-00045 measures what it cost.

**Open about the pyramid:**
10. **Summaries and the move predictor.** With the pyramid the move is predicted by the global summary B3 (slot
    nt − 2), not by the last patch. Every summary arm has foraged worse than vm-00036's EB. vm-00045's XB (EB on
    today's code) separates a cost of summaries from a change in the code since vm-00036.
11. **Square blocks follow image layout, not reading order.** In text they barely beat the plain window (vm-00041).
    Blocks shaped like the data's order (1 × 2^ℓ, then rows) would be the universal variant.
