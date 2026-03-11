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

1. Training loop runs on main thread/process, producing embeddings at full GPU speed.
2. At save intervals, snapshot current embeddings and push to a render queue.
3. Render worker (separate process) pulls snapshots, runs UMAP + Procrustes + save.
4. Training never waits for rendering to complete.

### Design considerations

- **Thread vs process:** UMAP is CPU-bound (numpy/numba), so a separate process avoids GIL. `multiprocessing.Process` with a `Queue`.
- **Memory:** Each snapshot is 6400 × 8 × 4 bytes = 200KB — negligible. Queue can hold many frames.
- **Warm projection state:** `prev_2d` must live in the render worker (it owns the projection continuity).
- **Backpressure:** If renders can't keep up, either drop frames or let the queue grow (bounded queue with drop policy).
- **Shutdown:** Sentinel value in queue signals render worker to flush remaining frames and exit.

## Results

*(to be filled)*

## Files

- `main.py` — Updated sentence mode with async render pipeline
