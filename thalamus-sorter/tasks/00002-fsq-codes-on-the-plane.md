# Task 00002: Bottleneck Codes Fed Back and Sorted With the Pixels (experiment ts-00028)

**Date:** 2026-11-14
**Status:** Complete (2026-11-14; results in `../00028-codes-on-the-plane/README.md`)

## Context

- **The owner (2026-11-14):** "I want a separate experiment in thalamus-sorter where we feed bottleneck signals back and try to sort the whole thing - I want to see it on a plane rendered to understand if V1, V2 and so on forms and what it shows visually - V1 obviously the image itself but what other Vn show - will it be bunch of smaller images of the same or something completely different."
- **The design discussion** (a claude.ai chat, saved as `~/Patch-based bottleneck models for sensory fields - Claude.pdf`):
  - Each window gets a small model with a finite scalar quantization (FSQ) bottleneck, trained to predict its inputs a few ticks ahead.
  - The codes are written back as plain signals, with no gradient through them, and sorted together with the pixels.
  - Its small simulation, with codes arriving one tick late and windows overlapping, found that codes form a map of their own instead of mixing into the pixel map. It also found that the sorter's positive-only rule splits codes into two mirror-image maps, because an FSQ number's sign is arbitrary.
  - It suggested sorting in the embedding and projecting to a plane only for viewing. That is what this codebase already does.
- **What the repo already has:**
  - **ts-00021:** feedback neurons, one tick late, with columns' softmax outputs as their signal. They formed a cloud of their own (zero mixed clusters).
  - **ts-00023:** that cloud stacked into layers V1 → V6, depth assigned by BFS on the feedback graph, with V2 receptive fields 2–3 times V1's.
  - **What has never been done:**
    - The layers have never been looked at as images. They were only counted and scattered as cluster centroids (`layer_scatter.py`).
    - The columns' outputs were categories (softmax over prototypes), not graded codes.
    - The fed-back signal was never trained to carry a prediction.

## The question

Rendered on a plane, what do V2, V3, … look like while the eye moves over an image?

1. **Smaller copies of the image:** each layer's map is topographic, a coarser version of the crop.
2. **Several interleaved copies:** one sub-map per code dimension, for example mean brightness, horizontal and vertical structure, and motion.
3. **Something else:** maps that follow the code's meaning rather than its place, for example all "moving left" codes together wherever they are in the crop.

## Design

### The model per window: `--column-type fsq` (new, `column_manager.py`)

- **Separate weights per column, batched.** Columns read their wired inputs; the run wires clusters as windows (existing ring wiring, `--cluster-max-k 2`, so each neuron feeds up to two windows and windows overlap).
- **Encoder:** a 2-layer MLP over the last `window` ticks of the column's input slots (raw values), ending in d = 4 numbers, each `tanh` then rounded to the FSQ levels (8, 5, 5, 5), with a straight-through gradient.
- **Decoder:** a 2-layer MLP from the 4 codes alone to the column's inputs `k` ticks ahead (default k = 2). The loss is MSE over filled slots only. There is no route from input to output other than the codes.
- **Training:** Adam, every `--column-train-every` ticks, on a batch of recent windows from the signal buffer.
- **Outputs:** the 4 code values in [−1, 1] per column, written back by the existing `--column-feedback` path as K = 4·M feedback neurons, one tick late. No gradient flows through them, so each column learns only from its own prediction error.

### Sorting and the input (revised after the motion checks below)

- **The input:** a 64 × 64 grayscale crop moving over `garden.png` (1536 × 1024), so 4,096 sensory neurons.
- **Motion:** the crop drifts smoothly: 2 px per tick, keeping its velocity with probability 0.95, else redrawing it; velocity reverses at the image edge (new `--saccade-momentum`). Under the existing random 50 px steps the next frame cannot be predicted.
- **Neighbour rule:** plain Pearson correlation of the raw (mean-subtracted) signals (`--no-use-deriv-corr`), threshold 0.5.
  - This rule already keeps negative correlations (absolute value), which an FSQ code needs, since its sign is arbitrary.
  - It applies to every neuron alike.
  - New `--corr-sign {auto,pos,abs}` (default `auto` = the original behaviour) lets the control FP keep positive correlations only.
- **Motion checks** (sorter only, no columns, 64 × 64, scratchpad `ts28/`; quality = share of each pixel's 10 nearest embedding neighbours within 3 px of it in the image):

| Motion | Rule | Ticks | Within 3 px |
|---|---|---|---|
| random 50 px steps (the existing presets) | derivative | 5k | 91.7% |
| drift 2 px, momentum 0.95, jump 50 px on 10% or 25% of ticks | derivative | 5k | 0.5%, 0.6% |
| drift 8 or 16 px, momentum 0.9 or 0.7 | derivative | 5k | 0.6–0.7% |
| drift 2 px, momentum 0.95 | plain, threshold 0.5 | 5k | 16.6% |
| the same | plain, threshold 0.8, 0.9, 0.95 | 5k | 0.5–0.6% (too few pairs) |
| random 50 px steps; drift with 10% jumps | plain | 5k | 15.4%; 14.2% |
| **drift 2 px, momentum 0.95** | **plain, threshold 0.5** | **30k** | **98.8%** |

- **What the plain rule may change:** under smooth drift a code one tick late still correlates with its source pixels. The design discussion predicted that codes then join the pixel map near their source patch instead of forming an area of their own. That is part of what the runs show; reading 3 tests it.

### Sizing

- **Rule:** with S sensory neurons, d codes per window and about s inputs per window primary, closing the loop needs about d·S / (s − d) code neurons.
- **The existing formula** `M = S // (neurons_per − d)` gives the same thing. With `--cluster-neurons-per 20`, M = 256 columns and 1,024 code neurons, so each window reads about 20 primary and 40 ring inputs (`--column-max-inputs 48`).

### Rendering: the deliverable (`render_layers.py`, new)

All renders are of the run's final state, replayed on a fresh 2,000-tick stretch of eye movement, with learning off.

1. **The whole plane:**
   - All neurons projected from the 8-D embedding to 2-D (UMAP), then filled as a Voronoi plane (the existing `render()`), each cell showing its neuron's current value (pixels and codes on one gray scale, −1 to 1).
   - It shows a second plane coloured by layer (V1 … Vn, from the existing BFS depth).
   - It is a video of 300 ticks.
2. **Layer by layer, side by side:** the input crop, then V1, V2, V3 …, each layer projected on its own and rendered as an image, as a video.
   - To make the panels comparable, each layer is rotated and flipped (Procrustes) to put its neurons' receptive-field centres where they lie in the crop. The ground-truth layout is used for display only, never in the run.
   - A second version colours each code neuron by its code dimension (1–4), to show sub-maps.
3. **Receptive fields:**
   - For every code neuron, its correlation with each input pixel over the replay, laid out as the crop.
   - A gallery of 16 per layer, plus each layer's mean receptive-field size.

### Runs (100k ticks each; one GPU, after vm-00092 frees the cards)

| Run | What changes | Why |
|---|---|---|
| **F** | the design above | the main run |
| **F0** | the encoders frozen at their random start (`--column-lr 0`) | is the picture from learning, or from the wiring alone? |
| **FP** | `--corr-sign pos` (positive correlations only) | the predicted mirror-image split |
| **F512** | M = 512 (`--cluster-neurons-per 12`) | does depth grow with the number of windows (ts-00023 found it did)? |

## Measures

- **Codes alive:** per code dimension, the number of FSQ levels used, and the share of columns with every dimension using at least 3 levels.
- **Prediction:** the loss against predicting each slot's recent mean.
- **Layers:** clusters per depth (BFS), with sensory and code neurons per cluster (mixed or separate).
- **Topography per layer:**
  - Procrustes disparity between the neurons' plane positions and their receptive-field centres (V1: plane against true pixel positions, the existing `--eval` metric).
  - The same with positions shuffled, as the floor.
- **Copies of the image:** per layer, the correlation between the rendered layer frame and the crop shrunk to the layer's size, after alignment, averaged over the replay.
- **Sub-maps:** among each code neuron's 8 nearest plane neighbours, the share with the same code dimension (chance 1/4) and the share from the same column.
- **Receptive-field size** per layer (area above half the peak).

## The readings, set before the results

1. **Plumbing:** F runs 100k ticks; at least 90% of columns use at least 3 levels in every code dimension; prediction loss is below the slot-mean baseline.
2. **V1 is the image:** at least 90% of F's sensory neurons have their 10 nearest embedding neighbours within 3 px on average (the existing `--eval` K10 measure; 98.8% in the sorter-only check at 30k).
3. **Layers form:** F has at least 3 depths with at least 10 clusters each.
4. **What V2 shows (descriptive, the owner's question).** Two views per layer, both sign-corrected (each code flipped to the sign of its receptive field), each correlated with the crop shrunk to the layer's grid over the replay:
   - **sorted:** the layer's own 2-D projection, rotated/flipped to best match its receptive-field centres. This is what the plane shows.
   - **at receptive-field centres:** each code drawn where in the crop it looks. This is what the codes carry, whatever the sort does with them.
   - *Refined after the 20k pilot (below):* the first version fitted the projection to the receptive-field centres with a free scale, which collapsed V2 to a point. The thresholds are unchanged and apply to the sorted view.
   - The outcomes:
   - **"A smaller copy"** if V2's sorted view correlates with the shrunk crop above 0.5.
   - **"Interleaved sub-maps"** if the same-dimension neighbour share is above 0.5 (chance 0.25).
   - Otherwise **"something else,"** described from the receptive-field gallery.
   - The same for V3 and beyond.
5. **Receptive fields grow:** mean receptive-field size rises with depth, V2 above V1 by at least 1.5 times (ts-00023 found 2–3 times).
6. **Learning matters:** F's code maps are more topographic than F0's (disparity lower) or its codes predict better. If F0 draws the same picture, the layers come from the wiring, not the training.
7. **The sign:** FP splits codes into two mirror maps (same-dimension neighbours divide into two groups of opposite receptive-field sign); F does not.

## Implementation

- [x] `FSQColumn` in `column_manager.py`; `fsq` branch in `cluster_manager.create_column`; CLI `--column-fsq-levels`, `--column-predict-k`, `--column-fsq-hidden`, `--column-fsq-samples`, `--column-fsq-train-every`. `--column-lr 0` now means 0 (it fell back to the default before).
- [x] `--corr-sign` in `solvers/drift_torch.py` (all Pearson paths) and `main.py`.
- [x] `--saccade-momentum` (and `--saccade-jump`, `--saccade-jump-step`, used only in the motion checks) in `main.py`; momentum 0 keeps the original walk and its random draws.
- [x] Motion checks (above).
- [x] `render_layers.py`: replay with learning off (the eye restarts at a random place every 100 ticks so that receptive fields are local), whole-plane and per-layer images and video, receptive fields, the measures above.
- [x] `FSQColumn` logs a fourth loss baseline, each slot's long-run mean (EMA 0.999): the no-information floor for reading 1. Copy-last and the window mean see all 48 inputs, so they are references, not floors.
- [x] Pilot: F for 20k ticks (below).
- [x] Tests (`dev/test_fsq_column.py`, 7 pass): FSQ values on the level grid, straight-through gradient, decoder reads only the codes, a column learns a predictable signal below the window-mean baseline, `--column-lr 0` freezes, save/load round trip, `--corr-sign` (abs keeps a perfectly anticorrelated pair, pos drops it).
- [x] Smoke run (1k ticks) while vm-00092 holds the GPUs.
- [x] The four runs, 100k ticks each, all on the idle 4090 with the owner's OK (F and F0 15:47–16:02, FP and F512 18:55–19:13; Makefile `runs_a`, `runs_b GPU_B=1`).
- [x] Renderer counts only live codes (changing in the replay, field peak at least 0.05) for layers and field sizes, after F0 showed silent codes manufacturing depth; the all-wiring counts are kept in `measures.json`.
- [x] `00028-codes-on-the-plane/` README, Makefile, images.

## Pilot (F, 20k ticks, 2026-11-14 12:01–12:25, on the 4090 beside vm-00092; `exp_00028/001_pilot_F_20k`)

Not a result, a check of the plumbing and the renders. The main runs are 100k ticks.

- **Sort:** 89.9% of pixels have their 10 nearest embedding neighbours within 3 px. Cluster contiguity rose from 0.08 (5k) to 0.95 (20k).
- **Codes:** 95% of columns use at least 3 levels in every dimension. Over the last 10k ticks the loss was above copy-last and the window mean (for example 0.095 against 0.074 and 0.084 at 20k), while the windows were still re-forming (1.7 switches per tick). The no-information baseline was added after this.
- **Layers:** 187 columns at depth 1, 67 at depth 2, 1 at depth 3; neurons V2 748, V3 268, V4 4.
- **Codes form an area of their own:** 0.15% of a code's 10 nearest neighbours are pixels.
- **Receptive fields:** local. V2 mean 236 px (blobs, often with an opposite-sign flank); V3 827 px, 3.5 times larger.
- **What the layers show at 20k:**
  - V2 sorted is a one-dimensional strip (topography 0.11 against 0.00 shuffled; sorted view against the crop −0.01).
  - Drawn where each code looks, V2 is a coarse copy of the crop (0.54); V3 gives 0.24.
  - No sub-maps: same-code-number neighbours 0.26 (chance 0.25).

## Results (summary; full report in the experiment README)

- **Readings:** 2, 4 ("something else"), 5, 6 (on prediction; its topography half is within run-to-run noise) and 7 (as sign grouping) met; 1 a narrow miss for F (87.9% of columns with 3+ levels at tick 100k, bar 90%; loss bar met; F's second run passes at 90.2%); 3 fails (2 column depths; F512 has 7 live columns at depth 3).
- **What the layers show:** codes form their own area; V2 and V3 there are folded sheets of spot detectors, topographic (0.33 and 0.44 in F; 0.22 and 0.24 in F's second run, so the layout varies between runs) but not a picture (sorted view 0.12, while V1 itself scores 0.29 on that measure); drawn at field centres they are a blurred copy (0.51, 0.45); no sub-map per code number (0.24).
- **F8 (added after the write-up, the owner's request, no readings set):** 8 numbers per window, 256 windows kept (`--cluster-neurons-per 24`). It does not help: 22% of codes silent, V2 at field centres 0.39 against F's 0.51, layout the same kind; `make f8`. At 200k ticks (`007_F8_200k`) more codes are silent (28%) and level use is flat from 100k on: not a training-length problem.
- **F16 (the owner's request, no readings set):** 16 numbers per window with the model scaled to keep F's proportions (section below). 42% of codes silent, V2 fainter at field centres (0.31) but the most topographic layout yet (0.51); `make f16`.
- **Deviations from the plan:** topography is Spearman of distances, not Procrustes disparity; reading 4's views refined after the pilot; silent codes excluded after F0.

## F16: 16 numbers per window, the column scaled with them (planned 2026-11-14, the owner's request)

- **The owner:** "lets have not 8 but 16 parameters in bottleneck - scale model appropriately - and lets repeat f8 but as F16 with 16 parameters".
- **Settings:** F's, for 100k ticks on the 4090, except as below. Each part keeps F's proportions:

| Part | F | F8 | F16 | Rule |
|---|---|---|---|---|
| Numbers per window (`--column-outputs`, levels 8,5,5,5 repeated) | 4 | 8 | 16 | the request |
| Neurons per window (`--cluster-neurons-per`) | 20 | 24 | 32 | keeps 256 windows: 4,096 // (32 − 16) |
| Input cap per column (`--column-max-inputs`) | 48 | 48 | 80 | F wires 36 inputs per column on average under a cap of 48 (9% at the cap); a 32-neuron window has 1.6 times as many ring members, so the cap rises by the same factor (77, rounded up) |
| Hidden width, encoder and decoder (`--column-fsq-hidden`) | 64 | 64 | 256 | F's 16 hidden units per number |
| Candidates per neuron (`--k-sample`) | 200 | 200 | 320 | 8,192 neurons instead of 5,120; F's 3.9% (k-sample scales with n) |

- **F8 kept F's cap, hidden width and k-sample,** so F16 against F8 differs in the column's size as well as in the number of codes; F16 against F keeps every proportion.
- **No readings set (as for F8).** Compared with F and F8: live codes and level use, V1 within 3 px, codes per layer, V2 and V3 topography, the sorted and field-centre views, field sizes, same-number neighbours.
- **Run:** `make f16` (`exp_00028/008_F16`, then `render_layers.py`).
- **Result (2026-11-14 23:00–23:20, the 4090):** scaling the model does not rescue the wider bottleneck. 58% of codes live (65 of 256 windows with none), 32% of columns with 3+ levels in every number in the replay, loss about equal to copy-last (1.04 over the last 10 reports; F 0.80). V2 has 1,890 codes, the most topographic layout of any run (0.51), but is fainter at field centres (0.31 against F's 0.51; field peak 0.38). V3 has 492 codes; depths 124, 65, 2. V1 99.0% within 3 px but folded into a bow-tie (topography 0.42). Details in the README's F16 section.

## F2 and a second run of F (planned 2026-11-15, the owner's OK)

- **The owner:** "yes, write the task file and run both checks" (the two cheap checks from the README's Next list, while the GPUs are idle).
- **F2, a narrower bottleneck:** 2 numbers per window at levels (8, 5), the first two of F's. Only the numbers change, as in F8: `--cluster-neurons-per 18` keeps 256 windows (4,096 // (18 − 2)); input cap 48, hidden width 64 and k-sample 200 as in F.
  - **The question:** F8 and F16 suggest a window holds about 4 numbers' worth of predictable structure. If so, F2's two numbers should both be fully used: nearly all codes live, nearly all columns at 3+ levels in both numbers.
  - **Expected cost of fewer numbers, if any:** fewer codes (512), so a smaller V2 and perhaps no V3; a coarser copy at field centres.
- **F again (`010_F_rerun`):** F's exact command. numpy's draws are not seeded (the eye's path, the embedding's start), so the run differs from F; the columns' starting weights are the same (the cluster manager's `RandomState(42)`), and GPU runs do not repeat anyway.
  - **The question:** how much do the single-run measures vary? Every comparison in this experiment is one run against one run. The F8 pair (100k vs 200k) already differed (45% against 52% of columns at 3+ levels).
- **No readings set.** Compared with F (and F8, F16): live codes, level use, V1 within 3 px, codes per layer, column depths, V2 and V3 topography, the sorted and field-centre views, field sizes and peaks, same-number neighbours, loss against copy-last and the no-information baseline (mean of the last 10 reports).
- **Run:** `make checks`: both at once on the 4090 (100k ticks, as F and F0 ran), then `render_layers.py` on each (`exp_00028/009_F2`, `010_F_rerun`).
- **Result (2026-11-15 23:28–23:46, the 4090):**
  - **F again:** what the codes carry repeats, the layout does not.
    - These repeat closely: live codes 99.8% (F 99.9%); 99% of columns at 3+ levels; V2 at field centres 0.49 (0.51); field peak 0.54 (0.55); V2 sorted view 0.10 (0.12); V1 98.3% within 3 px (98.8%); loss 0.64 of the no-information baseline (0.62); two column depths (195, 61 against 184, 72).
    - The layout does not repeat: V2 topography 0.22 (F 0.33) and V3 topography 0.24 (0.44).
    - The loss is below copy-last in only 46 of the 57 reports after 43k, against all 57 in F.
    - So single-run differences in topography of 0.1–0.2 (F0 0.23, FP 0.20) are within F's own spread.
  - **F2:** both numbers are fully used. 100% of codes are live, every column uses all 8 and all 5 levels, and the loss is 0.65 of the no-information baseline.
    - The codes are the sharpest of any run: V2 field peak 0.71, size 187 px, and V2 drawn at field centres matches the crop at 0.68, against F's 0.49–0.51.
    - With half the codes, V2 has 436 and V3 76 (38 depth-2 columns), and V2 topography is 0.39.
    - No sub-maps: same-number neighbours 0.49 (chance 0.5).
  - **The series 2 → 4 → 8 → 16 numbers:**
    - live codes 100% → 99.9% → 78% → 58%;
    - V2 at field centres 0.68 → 0.49–0.51 → 0.39 → 0.31;
    - V2 field peak 0.71 → 0.55 → 0.45 → 0.38.
    - Fewer numbers make fewer, sharper codes. All are used up to 4, and beyond that the extra numbers go silent.

## Verification

- [x] `info.json` of each run records `column_type` fsq, levels, k, `corr_sign`, momentum (checked for all four).
- [ ] Code values written back equal the column outputs of the previous tick (spot check): not done.
- [ ] The renders' crop panel matches the saccade position logged for the same tick: not done (the replay draws its own path).
