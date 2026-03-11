# ts-00008: Async Display Pipeline

**Date:** 2026-03-11
**Status:** In Progress
**Source:** *tagged on completion as `exp/ts-00008`*

## Goal

Decouple training from visualization so the GPU solver runs at full speed while rendering (UMAP projection, Procrustes alignment, frame saving) happens in a separate process/thread.

## Motivation

Currently in sentence mode, each frame render blocks training:
- UMAP warm projection: ~0.9s per frame
- Procrustes alignment: ~1ms
- Frame save (cv2.imwrite): ~1ms
- Training tick: ~1ms

Rendering dominates — training is 1000x faster than visualization. With 100 frames over 1000 ticks, we spend ~90s rendering and ~1s training. Decoupling lets training run continuously while renders happen in the background.

## Method

### Push vs pull

Two approaches considered:

1. **Push (queue):** Training pushes snapshots to a queue, renderer consumes in order. Simple, but needs backpressure policy — if renderer is slow, queue grows or frames must be dropped explicitly.

2. **Pull (shared memory):** Renderer periodically grabs the latest available snapshot. If rendering finishes fast, it waits for the next update. If rendering took longer than the interval, it skips to whatever's current. Natural best-effort — no queue, no drop policy.

**Chose pull.** The renderer works at its own pace and always sees the most recent state. No frame ordering guarantees, but for visualization that's fine — we want to see the current state, not replay every intermediate.

### Torn read problem

With shared memory, the renderer could read while training is mid-write, getting a mix of old and new embedding values (a "torn read"). Solved with **double buffering**:

- Two embedding slots in shared memory (slot 0 and slot 1)
- Training writes to the **inactive** slot, then atomically flips the active index
- Renderer always reads the **active** slot — always a complete, consistent snapshot

```
Training:                    Renderer:
  write to slot 1              read slot 0 (safe)
  flip active → 1              ...rendering...
  write to slot 0              read slot 1 (safe)
  flip active → 0              ...rendering...
```

### Implementation

- `multiprocessing.Process` for the render worker (UMAP is CPU-bound, avoids GIL)
- `multiprocessing.Array('f', n * dims)` × 2 for double-buffered embeddings
- `multiprocessing.Value('i')` for active slot index, tick counter, done flag
- Each snapshot is n × dims × 4 bytes (e.g. 6400 × 8 × 4 = 200KB) — negligible memory
- `prev_2d` (warm projection state) lives in the render worker, maintaining projection continuity
- Renderer polls with 1ms sleep when no new data available
- Training signals completion via done flag; renderer drains the final snapshot and exits

Enabled with `--async-render` flag. Falls back to synchronous render without it.

## Results

### Test 1: 1k ticks, async vs sync (80x80, D8, UMAP + Procrustes)

| Mode | Training time | Frames rendered | Total time |
|------|--------------|-----------------|------------|
| Sync | 125.5s | 100 (every 10th tick) | 125.5s |
| Async | 8.9s | 2 | 32.6s |

Training is **14x faster** when decoupled. Renderer only captured 2 frames because 1k ticks finishes in <9s while UMAP cold start alone takes ~15s. The renderer grabbed tick ~10 (random, disparity 0.999) and the final state (disparity 0.956).

### Test 2: 10k ticks, async (80x80, D8, UMAP + Procrustes)

With 10x more ticks, the renderer has time to capture many frames:

| Metric | Value |
|--------|-------|
| Training time | 88.2s |
| Frames rendered | 71 |
| Total time | 89.6s |
| Render drain after training | 1.4s |

Renderer nearly kept up — only 1.4s of drain after training finished. 71 frames at ~0.9s/frame warm UMAP ≈ 64s of render work, overlapping with 88s of training. Disparity curve drops smoothly from 1.0 → 0.24, sampling non-uniformly across ticks (denser at start when UMAP is slower cold, sparser mid-training as warm UMAP accelerates).

### Comparison: sync vs async at 10k ticks

| | Sync (estimated) | Async |
|--|-------------------|-------|
| Training wall time | ~88s + ~85s render = ~173s | 88.2s |
| Frames | 1000 (every tick) | 71 (best-effort) |
| Total wall time | ~173s | 89.6s |
| Speedup | 1x | **~2x** |

Async is ~2x faster at 10k ticks. The speedup grows with shorter training (at 1k ticks, 14x). For very long runs the renderer keeps up and overhead approaches zero.

## Files

- `main.py` — Updated sentence mode with `--async-render` flag and double-buffered shared memory
