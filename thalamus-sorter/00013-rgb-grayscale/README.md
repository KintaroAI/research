# ts-00013: RGB + Grayscale Multi-Channel Sorting

**Date:** 2026-03-12
**Status:** Complete
**Source:** `exp/ts-00013`

## Goal

Feed 4 channels per pixel position (R, G, B, grayscale) into the correlation-based sorter and analyze the resulting embedding structure. Key questions:

1. Does spatial proximity dominate (same-pixel neurons cluster tightly regardless of channel)?
2. Does channel identity dominate (R neurons form one cluster, G another, etc.)?
3. Or does a hybrid structure emerge (same-pixel clusters with channel-based sub-structure)?

## Motivation

All experiments so far used single-channel grayscale signals. Real neural systems process multi-modal signals (color, motion, etc.). Understanding how the sorter handles multi-channel input reveals whether it discovers both spatial and feature-type structure.

### Correlation structure prediction

At the same pixel position, R/G/B channels see the same spatial content (edges, textures). Their temporal correlation should be very high — they all change together as the saccade window moves. The grayscale channel is a weighted sum (0.299R + 0.587G + 0.114B), so it correlates with all three.

Cross-pixel same-channel correlation depends on the image's spatial statistics. For natural images, nearby pixels of the same channel are moderately correlated.

Expected hierarchy: **same-pixel cross-channel > nearby-pixel same-channel > far-pixel**.

### Setup

- Source: `saccades.png` (1536x1024 RGB), grayscale computed on-the-fly
- Grid: 80x80 pixel positions × 4 channels = 25,600 neurons, treated as flat pixel grid
- Layout: pixel-first via `.ravel()` on `(80, 80, 4)` — same-pixel channels are at consecutive indices (distance 1-3 in the flat grid)
- Effective render grid: 320×80 (width = crop_w × channels)
- k_sample: 800 (matching 3.1% fraction from ts-00012, since n=25,600)
- Signal buffer: (25600, T=1000), rolling saccade walker

## Approach

1. Add `--signal-channels 4` flag + PNG source loading
2. Run 80x80 rgbg with k_sample=800 for 50k ticks
3. Analyze K=10 embedding neighbor composition with `analyze_channels.py`
4. Compare with grayscale-only baseline (ts-00012: 160x160, k=800, 50k → 97.2% <5px)

## Results

### 50k ticks, k_sample=800

Training: 50000 ticks in 401s, 3.36B pairs, std=0.3415.

Standard eval (against 320x80 flat grid): PCA=0.9996, K10 mean=83.29, <3px=2.2%, <5px=5.5%.
The flat grid eval is meaningless — the true structure isn't a 320x80 rectangle.

### Channel structure analysis

**Channel identity dominates completely. Zero spatial mixing.**

| Neighbor type | Fraction |
|---------------|----------|
| Same pixel, diff channel | 0.1% |
| Diff pixel, same channel | 75.4% |
| Diff pixel, diff channel | 24.4% |

Per-channel neighbor composition:

| Channel | Neighbors: R | G | B | GS | Same-ch pixel dist | <5px |
|---------|-------------|---|---|----|--------------------|------|
| R | **100%** | 0 | 0 | 0 | 35.75 | 3.8% |
| G | 0 | **50.8%** | 0 | 49.2% | 22.85 | 19.3% |
| B | 0 | 0 | **100%** | 0 | 20.07 | 27.0% |
| GS | 0 | 49.0% | 0 | **51.0%** | 23.00 | 18.8% |

**Key observations:**

1. **R is isolated.** R neighbors are 100% R, zero mixing with any other channel. Within-R spatial quality is worst (3.8% <5px), suggesting R has weak spatial correlation in this image.

2. **G and GS merge.** Their K=10 neighbors are ~50/50 G and GS. This is expected: GS = 0.299R + 0.587G + 0.114B, so GS is 59% green. G and GS are nearly the same signal, so the solver correctly groups them together.

3. **B is isolated.** 100% same-channel, best spatial quality (27% <5px). Blue has the strongest spatial correlation structure — likely because natural images have smooth blue regions (sky, shadows).

4. **No same-pixel clustering.** Only 0.1% of neighbors share a pixel position. The solver treats each channel as an independent spatial map, never discovering that R/G/B/GS at the same (x,y) see the same region.

### Interpretation

The 8-dimensional embedding space separates into 3 channel clusters (R, G+GS, B). Within each cluster, the solver attempts topographic sorting, with varying success (B > G/GS > R).

This makes sense mechanically: the correlation between same-channel nearby pixels (spatial autocorrelation) is strong within each channel. But the correlation between different channels at the same pixel, while high in absolute terms, manifests as a *uniform offset* — all R values shift together, all G values shift together. The MSE scoring sees nearby same-channel pixels as more similar than co-located cross-channel pixels because the saccade walk produces correlated *spatial gradients* within a channel.

**Why no cross-channel clustering despite high same-pixel correlation:** When the saccade window shifts by 1 pixel, R(x,y) changes to R(x+1,y) — a small change if the image is locally smooth. But R(x,y) and G(x,y) both change — R goes to R(x+1,y) and G goes to G(x+1,y). The MSE between R(x,y) and G(x,y) over time is dominated by their different mean values (R and G channels have different intensities), not their co-variation. The solver correctly identifies this as "different signal, different cluster."

### Convergence comparison (grayscale vs RGBG)

| Run | n | Ticks | PCA disp | <3px | <5px |
|-----|---|-------|----------|------|------|
| Gray 80x80 | 6,400 | 1k | 0.67 | 50.0% | 73.3% |
| RGBG 80x80 | 25,600 | 1k | 0.999 | 0.3% | 0.7% |
| RGBG 80x80 | 25,600 | 50k | 0.999 | 2.2% | 5.5% |
| RGBG 80x80 | 25,600 | 1M | 0.999 | 1.2% | 3.1% |

Grayscale at 1k ticks is already 73% sorted. RGBG hasn't started — the solver spends its first ticks discovering channel identity (strong signal) before sorting spatially within channels (weaker signal). Even at 1M ticks (58.9B pairs, 7878s), flat-grid eval shows minimal improvement.

### 1M within-channel spatial quality

| Channel | Self-neighbors | Spatial <5px | Mean dist |
|---------|---------------|-------------|-----------|
| R | 96.5% | 7.5% | 31.01 |
| G | 53.1% (+ 44.2% GS) | 14.1% | 26.35 |
| B | 100% | 13.9% | 25.62 |
| GS | 55.0% (+ 44.2% G) | 13.1% | 26.56 |

Compared to 50k: R spatial quality improved 3.8%→7.5%, G/GS similar ~14% vs ~19%, B dropped 27%→14%. Channel separation remains total — 0% same-pixel clustering. 20x more training gives modest within-channel spatial improvement but the channel-dominates-space structure is unchanged.

### Color-tinted rendering

Added color-tinted pixel values so channels are visually distinguishable in UMAP renders:
- R neurons → red tint, G → green, B → blue, GS → gray

Confirmed visually: 3 distinct colored blobs (red, green+gray mixed, blue), each with internal spatial structure. Exactly matches the quantitative analysis.

### Pure RGB (no grayscale channel)

Removing the GS channel eliminates the G/GS merge noise. `--signal-channels 3` gives 80×80×3 = 19,200 neurons on a 240×80 render grid. k_sample=600 (3.1% fraction).

#### saccades.png — high inter-channel correlation (R-G=0.95, R-B=0.78, G-B=0.92)

50k ticks, 8 dims: PCA=0.9997, flat eval <5px=6.1%.

Channel separation is total (99.9-100% self-neighbors). Without GS bridging G, all three channels are completely isolated (only 65 cross-channel neighbor pairs out of 192k).

| Channel | <3px | <5px | Mean dist |
|---------|------|------|-----------|
| R | 1.6% | 3.4% | 40.83 |
| G | 17.1% | 34.0% | 9.09 |
| B | 6.2% | 12.7% | 20.07 |

G sorts best without GS noise (34% <5px vs 19% in RGBG). R remains worst — weakest spatial autocorrelation in this warm-toned image.

#### garden.png — low inter-channel correlation (R-G=0.64, R-B=0.48, G-B=0.74)

Switched to a color-diverse garden image (flowers, sky, foliage) to reduce inter-channel correlation.

50k ticks, 8 dims: PCA=0.9972, flat eval <5px=0.4%.

**Channels do NOT separate.** Neighbor composition is roughly uniform (~35% each), with 117,935 cross-channel neighbor pairs (61% of all neighbors). The lower inter-channel correlation means channel identity is no longer the dominant signal. But spatial quality is near-random (<5px ~1%, mean dist ~52) — the solver hasn't found spatial structure yet either.

| Channel | Self-neighbors | <3px | <5px | Mean dist |
|---------|---------------|------|------|-----------|
| R | 37.4% | 0.4% | 0.9% | 53.02 |
| G | 43.2% | 0.7% | 1.6% | 47.35 |
| B | 35.0% | 0.5% | 1.1% | 53.36 |

#### garden.png — dims and ticks comparison

| Run | Dims | Flat <5px | R self | G self | B self | R <5px | G <5px | B <5px |
|-----|------|-----------|--------|--------|--------|--------|--------|--------|
| 50k | 8 | 0.4% | 37% | 43% | 35% | 0.9% | 1.6% | 1.1% |
| 50k | 16 | 0.3% | 33% | 33% | 33% | 1.0% | 1.1% | 1.0% |
| 500k | 8 | 46.2% | 98% | 99% | 98% | 23.2% | 93.2% | 97.1% |
| 500k | 16 | 63.8% | 98% | 99% | 100% | 51.0% | 99.2% | 100% |

**Key findings:**

1. **At 50k, nothing has happened.** No channel separation, no spatial sorting, regardless of dims. With lower inter-channel correlation, the solver needs more ticks to discover channel identity.

2. **At 500k, channels fully separate and spatial sorting is excellent.** B and G reach near-perfect within-channel sorting (93-100% <5px). The garden's color diversity just delays separation — it doesn't prevent it. Given enough ticks, channel identity still dominates.

3. **16D helps R significantly (23→51% <5px).** Red in garden.png has a complex spatial distribution (scattered flowers, mixed regions), needing more embedding capacity. G and B have simpler spatial structure (contiguous sky, foliage) and sort well even in 8D.

4. **Cross-channel spatial proximity remains near-zero.** At 500k only 2-3k cross-channel neighbor pairs exist (out of 192k total), with mean pixel distance ~63. The solver never discovers that R/G/B at the same (x,y) see the same spatial region.

5. **Ticks matter more than dims.** The 50k→500k jump (10x ticks) transforms random embeddings into well-sorted channels. The 8D→16D jump helps the hardest channel (R) but G and B are already saturated at 8D.

## Commands

```bash
# RGBG 4-channel (saccades.png)
python main.py word2vec --mode correlation \
  -W 80 -H 80 --dims 8 --k-neg 5 --lr 0.001 \
  --normalize-every 100 --k-sample 800 --threshold 0.02 \
  --signal-T 1000 --signal-source saccades.png \
  --signal-channels 4 \
  --saccade-step 50 --use-mse \
  -i K_80_g.png \
  -f 50000 --save-every 1000 --eval --save-model \
  -o output_13_rgbg_50k

# Pure RGB 3-channel (garden.png)
python main.py word2vec --mode correlation \
  -W 80 -H 80 --dims 8 --k-neg 5 --lr 0.001 \
  --normalize-every 100 --k-sample 600 --threshold 0.02 \
  --signal-T 1000 --signal-source garden.png \
  --signal-channels 3 \
  --saccade-step 50 --use-mse \
  -i K_80_g.png \
  -f 50000 --save-every 1000 --eval --save-model \
  -o output_13_rgb_garden_50k
```

## Files

- `main.py` — added `--signal-channels`, PNG source loading (RGB/RGBG)
- `analyze_channels.py` — post-hoc channel structure analysis of embeddings
- `render_embeddings.py` — extended `render()` to handle (n,3) color pixel values
- `test_umap_compare.py` — CPU vs GPU UMAP parameter sweep (grayscale)
- `test_umap_compare_color.py` — CPU vs GPU UMAP parameter sweep (color, 320×80)
