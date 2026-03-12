#!/usr/bin/env python3
"""Thalamus sorter: topographic map formation via correlation-based
neuron sorting. Explores multiple algorithms for arranging neurons
so that temporally correlated units become spatial neighbors.

Grid mode (no camera):
    python main.py greedy --width 80 --height 80 --k 24
    python main.py continuous --width 80 --height 80 --k 24 --lr 0.05
    python main.py mst --width 20 --height 20
    python main.py sa --width 20 --height 20

Camera mode (live video input):
    python main.py camera-sa --width 32 --height 24
    python main.py camera-spatial --width 16 --height 12
"""

import argparse
import os
import sys
import numpy as np
import cv2

from utils.weights import (inverse_distance_1d, decay_distance_2d,
                            topk_decay2d, topk_inv1d, OnlineWeightMatrix)
from utils.correlation import temporal_correlation
from utils.graph import build_mst, tree_to_adjacency, dfs_order
from utils.camera import Camera
from utils.display import show_grid, show_vector, wait, poll_quit
from solvers.greedy_drift import GreedyDrift
from solvers.continuous_drift import ContinuousDrift
from solvers.temporal_correlation import TemporalCorrelation
from solvers.word2vec_drift import Word2vecDrift
from solvers.drift_torch import DriftSolver
from solvers.simulated_annealing import SimulatedAnnealing
from solvers.spatial_coherence import SpatialCoherence


# ---------------------------------------------------------------------------
# Grid-mode runners (pre-defined weight matrix, no camera)
# ---------------------------------------------------------------------------

def _load_image(path, width, height):
    """Load an image, resize to (width, height), convert to grayscale."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not load image '{path}'")
        sys.exit(1)
    resized = cv2.resize(img, (width, height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def run_greedy(args):
    w, h = args.width, args.height

    # Load image if provided
    image = None
    if args.image:
        image = _load_image(args.image, w, h)
        print(f"Greedy drift: {w}x{h} grid, k={args.k}, image={args.image}")
    else:
        print(f"Greedy drift: {w}x{h} grid, k={args.k}, "
              f"weights={args.weight_type}, decay={args.decay}")

    # Matrix-free top-K computation (O(nK) instead of O(n²))
    n = w * h
    k = min(args.k, n - 1)
    if args.weight_type == "decay2d":
        top_k = topk_decay2d(w, h, k)
    else:
        top_k = topk_inv1d(n, k)

    solver = GreedyDrift(w, h, k=k, move_fraction=args.move_fraction,
                         image=image, gpu=args.gpu, top_k=top_k)

    # Output directory for saving frames
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    max_frames = args.frames  # 0 = unlimited

    # Determine sync interval for GPU batching:
    # Sync only when we need CPU data — for saving or display.
    if args.gpu:
        if output_dir:
            sync_every = args.save_every
        else:
            sync_every = 10  # display interval

    tick = 0
    saved = 0
    while True:
        if args.gpu:
            # Full-GPU path: batch ticks, sync only when needed
            solver.run_gpu(sync_every)
            tick += sync_every
        else:
            # CPU reference path
            solver.tick()
            tick += 1

        if tick % 10 == 0:
            if solver.output is not None:
                show_grid("Restored image", solver.output)
            show_grid("Sorted neurons", solver.neurons_matrix)
            wait()

        # Save frame
        if output_dir and tick % args.save_every == 0:
            frame = solver.output if solver.output is not None else solver.neurons_matrix
            normalized = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            path = os.path.join(output_dir, f"frame_{saved:06d}.png")
            cv2.imwrite(path, normalized)
            saved += 1
            if saved % 100 == 0:
                print(f"  tick {tick}, saved {saved} frames")

        # Stop condition
        if max_frames > 0 and tick >= max_frames:
            print(f"Done: {tick} frames")
            break
        if poll_quit():
            break


def run_continuous(args):
    w, h = args.width, args.height

    image = None
    if args.image:
        image = _load_image(args.image, w, h)
        print(f"Continuous drift: {w}x{h} grid, k={args.k}, lr={args.lr}, "
              f"dims={args.dims}, image={args.image}")
    else:
        print(f"Continuous drift: {w}x{h} grid, k={args.k}, lr={args.lr}, "
              f"dims={args.dims}, weights={args.weight_type}")

    n = w * h
    k = min(args.k, n - 1)
    if args.weight_type == "decay2d":
        top_k = topk_decay2d(w, h, k)
    else:
        top_k = topk_inv1d(n, k)

    solver = ContinuousDrift(w, h, top_k, k=k, lr=args.lr, dims=args.dims,
                             margin=args.margin, image=image, gpu=args.gpu)

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    max_frames = args.frames
    sync_every = args.save_every if (args.gpu and output_dir) else 10

    tick = 0
    saved = 0
    while True:
        if args.gpu:
            solver.run_gpu(sync_every)
            tick += sync_every
        else:
            solver.tick()
            tick += 1
            if tick % 10 == 0:
                solver.render()

        if tick % 10 == 0:
            if solver.output is not None:
                show_grid("Continuous drift", solver.output)
            elif solver.neurons_matrix is not None:
                show_grid("Continuous drift", solver.neurons_matrix)
            wait()

        if output_dir and tick % args.save_every == 0:
            if solver.output is None:
                solver.render()
            frame = solver.output if solver.output is not None else solver.neurons_matrix
            normalized = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            path = os.path.join(output_dir, f"frame_{saved:06d}.png")
            cv2.imwrite(path, normalized)
            saved += 1
            if saved % 100 == 0:
                stats = solver.position_stats()
                print(f"  tick {tick}, saved {saved} frames, "
                      f"mean_norm={stats['mean_norm']:.4f} std={stats['std_mean']:.4f}")

        if max_frames > 0 and tick >= max_frames:
            if solver.output is None:
                solver.render()
            stats = solver.position_stats()
            print(f"Done: {tick} ticks, "
                  f"mean_norm={stats['mean_norm']:.4f} std={stats['std_mean']:.4f}")
            break
        if poll_quit():
            break


def run_word2vec(args):
    w, h = args.width, args.height

    image = None
    if args.image:
        image = _load_image(args.image, w, h)

    mode = args.mode
    norm_every = args.normalize_every
    if mode == "sentence":
        from render_embeddings import project, align_to_grid, render as render_emb
        import time
        import torch

        n = w * h
        k = min(args.k, n - 1)
        top_k = topk_decay2d(w, h, k)
        dsolver = DriftSolver(n, top_k=top_k, dims=args.dims, lr=args.lr,
                              mode='dot', k_neg=args.k_neg,
                              normalize_every=norm_every, device='cuda')

        # Warm start: load previous embeddings
        if args.warm_start:
            warm = np.load(args.warm_start)
            dsolver.positions = torch.from_numpy(warm).to(dsolver.device)
            print(f"Warm start from {args.warm_start} ({warm.shape})")

        print(f"Word2vec drift (sentence): {w}x{h} grid, k={args.k}, "
              f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}, "
              f"window={args.window}, normalize_every={norm_every}, "
              f"align={args.align}, async={args.async_render}")

        # Pixel values for image rendering
        pixel_values = None
        if image is not None:
            pixel_values = np.zeros(n, dtype=np.uint8)
            for i in range(n):
                pixel_values[i] = image[i // w, i % w]

        output_dir = args.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Pick projection method
        render_method = args.render
        if render_method == 'euclidean':
            render_method = 'pca'

        if getattr(args, 'sync_render', False):
            args.async_render = False

        if args.async_render and output_dir:
            # --- Async render: training and rendering decoupled ---
            # Double-buffered shared memory prevents torn reads.
            # Writer fills inactive slot, then flips the index atomically.
            # Reader always reads the active slot — always consistent.
            import multiprocessing as mp

            buf_size = n * args.dims
            shm_buf0 = mp.Array('f', buf_size)       # slot 0
            shm_buf1 = mp.Array('f', buf_size)       # slot 1
            shm_active = mp.Value('i', 0)             # which slot is readable
            shm_tick = mp.Value('i', 0)               # tick of active slot
            shm_done = mp.Value('i', 0)               # 1 = training finished

            def _render_worker(shm_buf0, shm_buf1, shm_active, shm_tick,
                               shm_done, n, dims, w, h, pixel_values,
                               render_method, do_align, cold_proj, output_dir):
                """Pull-based render worker. Reads from active double-buffer slot."""
                from render_embeddings import (project, align_to_grid,
                                               render as render_emb)
                bufs = [shm_buf0, shm_buf1]
                prev_2d = None
                last_rendered_tick = 0
                frame_idx = 0

                while True:
                    cur_tick = shm_tick.value
                    done = shm_done.value

                    if cur_tick > last_rendered_tick:
                        # Read from the active (complete) slot
                        slot = shm_active.value
                        emb = np.frombuffer(bufs[slot].get_obj(),
                                            dtype=np.float32).reshape(n, dims).copy()
                        last_rendered_tick = cur_tick

                        warm = None if cold_proj else prev_2d
                        pos_2d = project(emb, w, h, render_method, prev_2d=warm)
                        if do_align:
                            pos_2d = align_to_grid(pos_2d, w, h)
                        if not cold_proj:
                            prev_2d = pos_2d.copy()
                        frame = render_emb(pos_2d, w, h, pixel_values)

                        if frame is not None:
                            normalized = cv2.normalize(
                                frame, None, 0, 255, cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
                            path = os.path.join(output_dir,
                                                f"frame_{frame_idx:06d}.png")
                            cv2.imwrite(path, normalized)
                            frame_idx += 1

                    elif done:
                        break
                    else:
                        time.sleep(0.001)

                print(f"  render worker: {frame_idx} frames saved, "
                      f"last tick={last_rendered_tick}")

            bufs = [shm_buf0, shm_buf1]
            worker = mp.Process(target=_render_worker,
                                args=(shm_buf0, shm_buf1, shm_active, shm_tick,
                                      shm_done, n, args.dims,
                                      w, h, pixel_values, render_method,
                                      args.align, args.cold_projection,
                                      output_dir))
            worker.start()

            t0 = time.time()
            max_frames = args.frames
            write_slot = 1  # start writing to inactive slot
            for tick in range(1, max_frames + 1):
                dsolver.tick_sentence(window=args.window)
                if tick % args.save_every == 0:
                    # Write to inactive slot, then flip
                    emb = dsolver.get_positions()
                    dst = np.frombuffer(bufs[write_slot].get_obj(),
                                        dtype=np.float32)
                    np.copyto(dst, emb.ravel())
                    shm_active.value = write_slot
                    shm_tick.value = tick
                    write_slot = 1 - write_slot

            # Final snapshot
            emb = dsolver.get_positions()
            dst = np.frombuffer(bufs[write_slot].get_obj(), dtype=np.float32)
            np.copyto(dst, emb.ravel())
            shm_active.value = write_slot
            shm_tick.value = max_frames + 1
            time.sleep(0.01)
            shm_done.value = 1

            elapsed_train = time.time() - t0
            s = dsolver.stats()
            print(f"Training done: {max_frames} ticks in {elapsed_train:.1f}s, "
                  f"std={s['std']:.4f}")

            worker.join()
            elapsed_total = time.time() - t0
            print(f"Total (train + render drain): {elapsed_total:.1f}s")

        else:
            # --- Synchronous render (original path) ---
            prev_2d = None

            def render_frame():
                nonlocal prev_2d
                emb = dsolver.get_positions()
                warm = None if args.cold_projection else prev_2d
                pos_2d = project(emb, w, h, render_method, prev_2d=warm)
                if args.align:
                    pos_2d = align_to_grid(pos_2d, w, h)
                if not args.cold_projection:
                    prev_2d = pos_2d.copy()
                return render_emb(pos_2d, w, h, pixel_values)

            t0 = time.time()
            max_frames = args.frames
            saved = 0
            for tick in range(1, max_frames + 1):
                dsolver.tick_sentence(window=args.window)
                if output_dir and tick % args.save_every == 0:
                    frame = render_frame()
                    if frame is not None:
                        normalized = cv2.normalize(
                            frame, None, 0, 255, cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U)
                        path = os.path.join(output_dir,
                                            f"frame_{saved:06d}.png")
                        cv2.imwrite(path, normalized)
                        saved += 1
                        if saved % 100 == 0:
                            elapsed = time.time() - t0
                            s = dsolver.stats()
                            print(f"  tick {tick}, saved {saved} frames, "
                                  f"std={s['std']:.4f}, elapsed={elapsed:.1f}s")
                elif not output_dir and tick % 10 == 0:
                    frame = render_frame()
                    if frame is not None:
                        show_grid("Sentence skip-gram", frame)
                    wait()
                if poll_quit():
                    break

            elapsed = time.time() - t0
            s = dsolver.stats()
            print(f"Done: {max_frames} ticks in {elapsed:.1f}s, "
                  f"std={s['std']:.4f}")

            # Save final frame
            if output_dir:
                frame = render_frame()
                if frame is not None:
                    normalized = cv2.normalize(
                        frame, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
                    path = os.path.join(output_dir,
                                        f"frame_{saved:06d}.png")
                    cv2.imwrite(path, normalized)
                    print(f"  final frame saved: {path}")

        # Save model (only if explicitly requested)
        if args.save_model or args.save_model_path:
            save_path = args.save_model_path
            if not save_path and output_dir:
                save_path = os.path.join(output_dir, "model.npy")
            if save_path:
                np.save(save_path, dsolver.get_positions())
                print(f"  model saved: {save_path}")
        return

    if mode == "correlation":
        from render_embeddings import project, align_to_grid, render as render_emb
        import time
        import torch

        n = w * h
        dsolver = DriftSolver(n, top_k=None, k=args.k, dims=args.dims,
                              lr=args.lr, mode='dot', k_neg=args.k_neg,
                              normalize_every=norm_every, device='cuda')

        if args.warm_start:
            warm = np.load(args.warm_start)
            dsolver.positions = torch.from_numpy(warm).to(dsolver.device)
            print(f"Warm start from {args.warm_start} ({warm.shape})")

        # Build signal buffer: Gaussian-smoothed random fields
        # Each frame is a spatially-correlated 2D signal — nearby pixels
        # get similar values, giving correlation structure to discover.
        T = args.signal_T
        sigma = args.signal_sigma
        print(f"Word2vec drift (correlation): {w}x{h} grid, "
              f"k_sample={args.k_sample}, threshold={args.threshold}, "
              f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}, "
              f"signal: T={T}, sigma={sigma}, "
              f"normalize_every={norm_every}, align={args.align}")

        # Generate spatially-correlated signals
        from scipy.ndimage import gaussian_filter
        signals_np = np.zeros((n, T), dtype=np.float32)
        for t in range(T):
            noise = np.random.randn(h, w).astype(np.float32)
            smoothed = gaussian_filter(noise, sigma=sigma)
            signals_np[:, t] = smoothed.ravel()
        signals = torch.from_numpy(signals_np).to(dsolver.device)
        print(f"  signal buffer: ({n}, {T}), spatial sigma={sigma}")

        # Pixel values for rendering
        pixel_values = None
        if image is not None:
            pixel_values = np.zeros(n, dtype=np.uint8)
            for i in range(n):
                pixel_values[i] = image[i // w, i % w]

        output_dir = args.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        render_method = args.render
        if render_method == 'euclidean':
            render_method = 'pca'

        prev_2d = None

        def render_frame():
            nonlocal prev_2d
            emb = dsolver.get_positions()
            warm = None if args.cold_projection else prev_2d
            pos_2d = project(emb, w, h, render_method, prev_2d=warm)
            if args.align:
                pos_2d = align_to_grid(pos_2d, w, h)
            if not args.cold_projection:
                prev_2d = pos_2d.copy()
            return render_emb(pos_2d, w, h, pixel_values)

        t0 = time.time()
        max_frames = args.frames
        saved = 0
        total_pairs = 0
        for tick in range(1, max_frames + 1):
            pairs = dsolver.tick_correlation(
                signals, k_sample=args.k_sample,
                threshold=args.threshold, window=args.window,
                anchor_only=args.anchor_only)
            total_pairs += pairs

            if output_dir and tick % args.save_every == 0:
                frame = render_frame()
                if frame is not None:
                    normalized = cv2.normalize(
                        frame, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
                    path = os.path.join(output_dir, f"frame_{saved:06d}.png")
                    cv2.imwrite(path, normalized)
                    saved += 1
                    if saved % 100 == 0:
                        elapsed = time.time() - t0
                        s = dsolver.stats()
                        print(f"  tick {tick}, saved {saved} frames, "
                              f"std={s['std']:.4f}, pairs={total_pairs}, "
                              f"elapsed={elapsed:.1f}s")
            elif not output_dir and tick % 10 == 0:
                frame = render_frame()
                if frame is not None:
                    show_grid("Correlation skip-gram", frame)
                wait()
            if poll_quit():
                break

        elapsed = time.time() - t0
        s = dsolver.stats()
        print(f"Done: {max_frames} ticks in {elapsed:.1f}s, "
              f"std={s['std']:.4f}, total_pairs={total_pairs}")

        if output_dir:
            frame = render_frame()
            if frame is not None:
                normalized = cv2.normalize(
                    frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                path = os.path.join(output_dir, f"frame_{saved:06d}.png")
                cv2.imwrite(path, normalized)
                print(f"  final frame saved: {path}")

        if args.save_model or args.save_model_path:
            save_path = args.save_model_path
            if not save_path and output_dir:
                save_path = os.path.join(output_dir, "model.npy")
            if save_path:
                np.save(save_path, dsolver.get_positions())
                print(f"  model saved: {save_path}")
        return

    if mode == "similarity":
        print(f"Word2vec drift (similarity): {w}x{h} grid, P={args.P}, "
              f"sigma={args.sigma}, threshold={args.threshold}, "
              f"lr={args.lr}, dims={args.dims}")
        solver = Word2vecDrift(w, h, top_k=None, lr=args.lr, dims=args.dims,
                               P=args.P, sigma=args.sigma,
                               threshold=args.threshold,
                               normalize_every=norm_every,
                               image=image, gpu=args.gpu)
    elif mode == "dual-xy":
        print(f"Word2vec drift (dual-xy): {w}x{h} grid, k={args.k}, "
              f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}, "
              f"normalize_every={norm_every}")
        n = w * h
        k = min(args.k, n - 1)
        # top_k will be built internally by tick_dual_xy
        solver = Word2vecDrift(w, h, top_k=None, k=k, lr=args.lr, dims=args.dims,
                               k_neg=args.k_neg, normalize_every=norm_every,
                               image=image, gpu=args.gpu)
    else:
        print(f"Word2vec drift ({mode}): {w}x{h} grid, k={args.k}, "
              f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}")
        n = w * h
        k = min(args.k, n - 1)
        top_k = topk_decay2d(w, h, k)
        solver = Word2vecDrift(w, h, top_k, k=k, lr=args.lr, dims=args.dims,
                               k_neg=args.k_neg, normalize_every=norm_every,
                               image=image, gpu=args.gpu)

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    max_frames = args.frames
    sync_every = args.save_every if (args.gpu and output_dir) else 10

    gpu_mode = mode if mode == "similarity" else "skipgram"
    tick_fns = {"similarity": solver.tick_similarity, "skipgram": solver.tick,
                "dual": solver.tick_dual, "dual-xy": solver.tick_dual_xy}
    tick_fn = tick_fns[mode]
    render_fns = {"euclidean": solver.render, "angular": solver.render_angular,
                  "bestpc": solver.render_bestpc}
    render_fn = render_fns[args.render]

    tick = 0
    saved = 0
    while True:
        if args.gpu:
            solver.run_gpu(sync_every, mode=gpu_mode)
            tick += sync_every
        else:
            tick_fn()
            tick += 1
            if tick % 10 == 0:
                render_fn()

        if tick % 10 == 0:
            if solver.output is not None:
                show_grid("Word2vec drift", solver.output)
            elif solver.neurons_matrix is not None:
                show_grid("Word2vec drift", solver.neurons_matrix)
            wait()

        if output_dir and tick % args.save_every == 0:
            if solver.output is None:
                render_fn()
            frame = solver.output if solver.output is not None else solver.neurons_matrix
            normalized = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            path = os.path.join(output_dir, f"frame_{saved:06d}.png")
            cv2.imwrite(path, normalized)
            saved += 1
            if saved % 100 == 0:
                stats = solver.position_stats()
                print(f"  tick {tick}, saved {saved} frames, "
                      f"mean_norm={stats['mean_norm']:.4f} std={stats['std_mean']:.4f}")

        if max_frames > 0 and tick >= max_frames:
            if solver.output is None:
                render_fn()
            stats = solver.position_stats()
            print(f"Done: {tick} ticks, "
                  f"mean_norm={stats['mean_norm']:.4f} std={stats['std_mean']:.4f}")
            break
        if poll_quit():
            break


def run_temporal(args):
    w, h = args.width, args.height

    image = None
    if args.image:
        image = _load_image(args.image, w, h)

    print(f"Temporal correlation: {w}x{h} grid, buf={args.buf_source}, P={args.P}, "
          f"lr={args.lr}, dims={args.dims}")

    # Precompute top_k if using embeddings source
    top_k = None
    if args.buf_source == "embeddings":
        n = w * h
        k = min(args.emb_k, n - 1)
        top_k = topk_decay2d(w, h, k)

    solver = TemporalCorrelation(w, h, P=args.P, lr=args.lr, dims=args.dims,
                                 buf_source=args.buf_source,
                                 T=args.T, sigma=args.sigma,
                                 threshold=args.threshold,
                                 top_k=top_k, emb_dims=args.emb_dims,
                                 emb_ticks=args.emb_ticks,
                                 image=image, gpu=args.gpu)

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    max_frames = args.frames
    sync_every = args.save_every if (args.gpu and output_dir) else 10

    tick = 0
    saved = 0
    while True:
        if args.gpu:
            solver.run_gpu(sync_every)
            tick += sync_every
        else:
            solver.tick()
            tick += 1
            if tick % 10 == 0:
                solver.render()

        if tick % 10 == 0:
            if solver.output is not None:
                show_grid("Temporal correlation", solver.output)
            elif solver.neurons_matrix is not None:
                show_grid("Temporal correlation", solver.neurons_matrix)
            wait()

        if output_dir and tick % args.save_every == 0:
            if solver.output is None:
                solver.render()
            frame = solver.output if solver.output is not None else solver.neurons_matrix
            normalized = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            path = os.path.join(output_dir, f"frame_{saved:06d}.png")
            cv2.imwrite(path, normalized)
            saved += 1
            if saved % 100 == 0:
                stats = solver.position_stats()
                print(f"  tick {tick}, saved {saved} frames, "
                      f"mean_norm={stats['mean_norm']:.4f} std={stats['std_mean']:.4f}")

        if max_frames > 0 and tick >= max_frames:
            if solver.output is None:
                solver.render()
            stats = solver.position_stats()
            print(f"Done: {tick} ticks, "
                  f"mean_norm={stats['mean_norm']:.4f} std={stats['std_mean']:.4f}")
            break
        if poll_quit():
            break


def run_mst(args):
    w, h = args.width, args.height
    print(f"MST sort: {w}x{h} grid, weights={args.weight_type}")

    if args.weight_type == "decay2d":
        W = decay_distance_2d(w, h, decay_rate=args.decay)
    else:
        W = inverse_distance_1d(w * h)

    # MST sort is a one-shot operation
    sorted_matrix = _mst_sort(W, w, h)

    # Show input (random) and output (sorted)
    random_matrix = np.random.permutation(w * h).reshape(h, w)
    show_grid("Random neurons", random_matrix)
    show_grid("MST sorted", sorted_matrix)
    print("Press 'q' to quit.")
    while True:
        if poll_quit():
            break


def _mst_sort(weight_matrix, width, height, start=0):
    """Build max spanning tree and DFS-traverse for ordering."""
    tree_edges = build_mst(weight_matrix, maximize=True)
    graph = tree_to_adjacency(tree_edges)
    ordering = dfs_order(start, graph)
    return np.array(ordering).reshape(height, width)


def run_sa(args):
    w, h = args.width, args.height
    print(f"Simulated annealing: {w}x{h} grid, temp={args.temp}, "
          f"cooling={args.cooling}, weights={args.weight_type}")

    if args.weight_type == "decay2d":
        W = decay_distance_2d(w, h, decay_rate=args.decay)
    else:
        W = inverse_distance_1d(w * h)

    solver = SimulatedAnnealing(W, w, h,
                                init_temp=args.temp,
                                cooling_rate=args.cooling)

    tick = 0
    while True:
        solver.tick(iterations=args.sa_iterations)
        tick += 1
        if tick % 5 == 0:
            show_grid("SA sorted", solver.neurons_matrix)
            wait()
            print(f"  tick {tick}, cost={solver.cost:.1f}, temp={solver.temp:.4f}")
        if poll_quit():
            break


# ---------------------------------------------------------------------------
# Camera-mode runners (online weight learning from live video)
# ---------------------------------------------------------------------------

def run_camera_sa(args):
    """Learn weight matrix from camera temporal correlations,
    then sort with MST on learned weights."""
    w, h = args.width, args.height
    n = w * h
    steps = args.steps
    print(f"Camera SA: {w}x{h}, temporal window={steps}")

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("Error: sklearn required for camera-sa mode. "
              "Install with: pip install scikit-learn")
        sys.exit(1)

    online_W = OnlineWeightMatrix(n, tail=10)
    ideal_W = inverse_distance_1d(n)

    temporal_input = np.zeros((n, steps))
    temporal_output = np.zeros((n, steps))

    # Permutation matrix (starts as identity = no sorting)
    perm = np.eye(n)

    with Camera(w, h) as cam:
        tick = 0
        neurons_matrix = np.arange(n).reshape(h, w)

        while True:
            step = tick % steps
            tick += 1

            temporal_input[:, step] = cam.read_gray()
            # Forward pass through permutation
            temporal_output[:, step] = np.dot(temporal_input[:, step], perm)

            if tick >= steps:
                # Find nearest neighbors in temporal space
                nn = NearestNeighbors(n_neighbors=min(8, n - 1),
                                     algorithm='ball_tree')
                nn.fit(temporal_input)
                _, neighbors_indices = nn.kneighbors(return_distance=True)

                # Update weight matrix from observed correlations
                for i, neighbors in enumerate(neighbors_indices):
                    for j in neighbors:
                        if i != j:
                            corr = temporal_correlation(
                                temporal_output, temporal_output, i, j)
                            # Find neuron identities at positions i, j
                            ni = np.where(perm[:, i] == 1)[0]
                            nj = np.where(perm[:, j] == 1)[0]
                            if len(ni) > 0 and len(nj) > 0:
                                online_W.update(ni[0], nj[0], abs(corr))

                # Sort using learned weights via MST
                sorted_matrix = _mst_sort(online_W.weights, w, h)
                # Rebuild output using sorted ordering
                output = temporal_output[:, step]
                rebuilt = np.zeros(n)
                for idx, neuron_id in enumerate(sorted_matrix.ravel()):
                    if neuron_id < n:
                        rebuilt[idx] = output[neuron_id]
                show_vector("Sorted output", rebuilt, w, h)

            # Show variance and learned weights
            if tick > steps:
                variance = np.var(temporal_input, axis=1)
                show_vector("Variance", variance, w, h)
                show_grid("Learned weights", online_W.weights)

            wait()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def run_camera_spatial(args):
    """Sort camera output using total variation loss gradient descent."""
    w, h = args.width, args.height
    print(f"Camera spatial coherence: {w}x{h}, lr={args.lr}, epochs={args.epochs}")

    solver = SpatialCoherence(w, h, lr=args.lr)

    with Camera(w, h) as cam:
        while True:
            input_signal = cam.read_gray()
            output = solver.optimize(input_signal, epochs=args.epochs)
            show_vector("Spatial coherence output", output, w, h)
            wait()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Thalamus sorter: topographic map formation algorithms")
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args added to each subparser
    def add_common(p):
        p.add_argument("--width", "-W", type=int, default=40)
        p.add_argument("--height", "-H", type=int, default=40)
        p.add_argument("--weight-type", choices=["inv1d", "decay2d"],
                        default="decay2d",
                        help="Weight matrix type (default: decay2d)")
        p.add_argument("--decay", type=float, default=0.1,
                        help="Decay rate for decay2d weights (default: 0.1)")

    # --- greedy ---
    p_greedy = sub.add_parser("greedy", help="Greedy drift sorting")
    add_common(p_greedy)
    p_greedy.add_argument("--k", type=int, default=24,
                          help="Number of nearest neighbors (default: 24)")
    p_greedy.add_argument("--move-fraction", type=float, default=0.9,
                          help="Fraction of neurons to move per tick (default: 0.9)")
    p_greedy.add_argument("--image", "-i", type=str, default=None,
                          help="Input image to scramble and reconstruct")
    p_greedy.add_argument("--frames", "-f", type=int, default=0,
                          help="Number of frames to run (0 = unlimited)")
    p_greedy.add_argument("--output-dir", "-o", type=str, default=None,
                          help="Directory to save output frames as PNGs")
    p_greedy.add_argument("--save-every", type=int, default=1,
                          help="Save every Nth frame (default: 1)")
    p_greedy.add_argument("--gpu", action="store_true",
                          help="Use GPU acceleration via CuPy")
    p_greedy.set_defaults(func=run_greedy)

    # --- continuous ---
    p_cont = sub.add_parser("continuous",
                            help="Continuous position drift (embedding-style)")
    add_common(p_cont)
    p_cont.add_argument("--k", type=int, default=24,
                        help="Number of nearest neighbors (default: 24)")
    p_cont.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate / step fraction (default: 0.05)")
    p_cont.add_argument("--dims", type=int, default=2,
                        help="Position vector dimensionality (default: 2)")
    p_cont.add_argument("--margin", type=float, default=0.1,
                        help="Dead zone radius around centroid (default: 0.1)")
    p_cont.add_argument("--image", "-i", type=str, default=None,
                        help="Input image to scramble and reconstruct")
    p_cont.add_argument("--frames", "-f", type=int, default=0,
                        help="Number of frames to run (0 = unlimited)")
    p_cont.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Directory to save output frames as PNGs")
    p_cont.add_argument("--save-every", type=int, default=1,
                        help="Save every Nth frame (default: 1)")
    p_cont.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration via CuPy")
    p_cont.set_defaults(func=run_continuous)

    # --- word2vec ---
    p_w2v = sub.add_parser("word2vec",
                           help="Word2vec-style drift (skip-gram or similarity mode)")
    p_w2v.add_argument("--width", "-W", type=int, default=40,
                       help="Grid width (default: 40)")
    p_w2v.add_argument("--height", "-H", type=int, default=40,
                       help="Grid height (default: 40)")
    p_w2v.add_argument("--mode", choices=["skipgram", "similarity", "dual", "dual-xy", "sentence", "correlation"],
                       default="similarity",
                       help="Update mode: skipgram, similarity, dual, dual-xy, sentence (precomputed neighbors), correlation (online neighbor discovery from signals) (default: similarity)")
    p_w2v.add_argument("--window", type=int, default=5,
                       help="Sliding window size for sentence mode (default: 5)")
    p_w2v.add_argument("--normalize-every", type=int, default=0,
                       help="Normalize W/C vectors every N ticks (0 = disabled, default: 0)")
    # skipgram mode args
    p_w2v.add_argument("--k", type=int, default=24,
                       help="Number of nearest neighbors for positive sampling (skipgram mode, default: 24)")
    p_w2v.add_argument("--k-neg", type=int, default=5,
                       help="Number of negative samples per positive (skipgram mode, default: 5)")
    # similarity mode args
    p_w2v.add_argument("--P", type=int, default=10,
                       help="Random peers per neuron per tick (similarity mode, default: 10)")
    p_w2v.add_argument("--sigma", type=float, default=5.0,
                       help="Gaussian RBF kernel width (similarity mode, default: 5.0)")
    p_w2v.add_argument("--threshold", type=float, default=0.0,
                       help="Similarity/correlation threshold (default: 0.0)")
    # correlation mode args
    p_w2v.add_argument("--k-sample", type=int, default=50,
                       help="Random candidates to check per neuron (correlation mode, default: 50)")
    p_w2v.add_argument("--signal-T", type=int, default=100,
                       help="Temporal signal buffer length (correlation mode, default: 100)")
    p_w2v.add_argument("--signal-sigma", type=float, default=3.0,
                       help="Gaussian smoothing sigma for signal generation (correlation mode, default: 3.0)")
    p_w2v.add_argument("--anchor-only", action="store_true",
                       help="Correlation mode: only (anchor, neighbor) pairs, no transitive sliding window")
    p_w2v.add_argument("--render", choices=["euclidean", "angular", "bestpc",
                                            "direct", "procrustes", "lstsq",
                                            "umap", "tsne", "spectral", "mds"],
                       default="euclidean",
                       help="Render projection: euclidean/pca (top-2 PCs), "
                            "bestpc (grid-correlated PCs), angular (unit norm + PCA), "
                            "direct (first 2 dims), procrustes/lstsq (supervised linear), "
                            "umap/tsne/spectral/mds (nonlinear) (default: euclidean)")
    p_w2v.add_argument("--align", action="store_true",
                       help="Procrustes-align rendered output to grid (fixes rotation/flip)")
    p_w2v.add_argument("--warm-start", type=str, default=None,
                       help="Load .npy embeddings as initial positions (warm start)")
    p_w2v.add_argument("--cold-projection", action="store_true",
                       help="Disable projection warm start (run UMAP/t-SNE from scratch each frame)")
    p_w2v.add_argument("--async-render", action="store_true", default=True,
                       help="Render in separate process (default: on)")
    p_w2v.add_argument("--sync-render", action="store_true",
                       help="Force synchronous rendering")
    p_w2v.add_argument("--save-model", action="store_true",
                       help="Save final embeddings to .npy file")
    p_w2v.add_argument("--save-model-path", type=str, default=None,
                       help="Path for saved model (default: output_dir/model.npy)")
    # common args
    p_w2v.add_argument("--lr", type=float, default=0.05,
                       help="Learning rate (default: 0.05)")
    p_w2v.add_argument("--dims", type=int, default=2,
                       help="Position vector dimensionality (default: 2)")
    p_w2v.add_argument("--image", "-i", type=str, default=None,
                       help="Input image to scramble and reconstruct")
    p_w2v.add_argument("--frames", "-f", type=int, default=0,
                       help="Number of frames to run (0 = unlimited)")
    p_w2v.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Directory to save output frames as PNGs")
    p_w2v.add_argument("--save-every", type=int, default=1,
                       help="Save every Nth frame (default: 1)")
    p_w2v.add_argument("--gpu", action="store_true",
                       help="Use GPU acceleration via CuPy")
    p_w2v.set_defaults(func=run_word2vec)

    # --- temporal ---
    p_temp = sub.add_parser("temporal",
                            help="Temporal correlation sorting (no precomputed neighbors)")
    p_temp.add_argument("--width", "-W", type=int, default=40,
                        help="Grid width (default: 40)")
    p_temp.add_argument("--height", "-H", type=int, default=40,
                        help="Grid height (default: 40)")
    p_temp.add_argument("--buf-source", choices=["synthetic", "gaussian", "embeddings"],
                        default="synthetic",
                        help="Buffer source: gaussian fields or converged embeddings (default: embeddings)")
    p_temp.add_argument("--T", type=int, default=200,
                        help="Buffer length for gaussian source (default: 200)")
    p_temp.add_argument("--P", type=int, default=1,
                        help="Random peers per neuron per tick (default: 1)")
    p_temp.add_argument("--sigma", type=float, default=5.0,
                        help="Spatial smoothing sigma / RBF kernel width (default: 5.0)")
    p_temp.add_argument("--threshold", type=float, default=0.0,
                        help="Similarity threshold: sim > threshold pulls, < repels (default: 0.0)")
    p_temp.add_argument("--emb-k", type=int, default=25,
                        help="K neighbors for embedding generation (default: 25)")
    p_temp.add_argument("--emb-dims", type=int, default=16,
                        help="Dimensionality of source embeddings (default: 16)")
    p_temp.add_argument("--emb-ticks", type=int, default=100000,
                        help="Convergence ticks for embedding generation (default: 100000)")
    p_temp.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (default: 0.05)")
    p_temp.add_argument("--dims", type=int, default=2,
                        help="Position vector dimensionality (default: 2)")
    p_temp.add_argument("--image", "-i", type=str, default=None,
                        help="Input image to scramble and reconstruct")
    p_temp.add_argument("--frames", "-f", type=int, default=0,
                        help="Number of frames to run (0 = unlimited)")
    p_temp.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Directory to save output frames as PNGs")
    p_temp.add_argument("--save-every", type=int, default=1,
                        help="Save every Nth frame (default: 1)")
    p_temp.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration via CuPy")
    p_temp.set_defaults(func=run_temporal)

    # --- mst ---
    p_mst = sub.add_parser("mst", help="MST-based one-shot sorting")
    add_common(p_mst)
    p_mst.set_defaults(func=run_mst)

    # --- sa ---
    p_sa = sub.add_parser("sa", help="Simulated annealing sorting")
    add_common(p_sa)
    p_sa.add_argument("--temp", type=float, default=100.0,
                      help="Initial temperature (default: 100)")
    p_sa.add_argument("--cooling", type=float, default=0.99,
                      help="Cooling rate (default: 0.99)")
    p_sa.add_argument("--sa-iterations", type=int, default=100,
                      help="SA iterations per tick (default: 100)")
    p_sa.set_defaults(func=run_sa)

    # --- camera-sa ---
    p_csa = sub.add_parser("camera-sa",
                           help="Camera: learn weights online, sort with MST")
    p_csa.add_argument("--width", "-W", type=int, default=32)
    p_csa.add_argument("--height", "-H", type=int, default=24)
    p_csa.add_argument("--steps", type=int, default=10,
                       help="Temporal window size (default: 10)")
    p_csa.set_defaults(func=run_camera_sa)

    # --- camera-spatial ---
    p_csp = sub.add_parser("camera-spatial",
                           help="Camera: spatial coherence via TV loss")
    p_csp.add_argument("--width", "-W", type=int, default=16)
    p_csp.add_argument("--height", "-H", type=int, default=12)
    p_csp.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    p_csp.add_argument("--epochs", type=int, default=1000,
                       help="Optimization epochs per frame (default: 1000)")
    p_csp.set_defaults(func=run_camera_spatial)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
