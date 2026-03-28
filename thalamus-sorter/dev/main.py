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

from utils.weights import topk_decay2d
from utils.display import show_grid, wait, poll_quit
from solvers.drift_torch import DriftSolver
from legacy_runners import (run_greedy, run_continuous, run_temporal,
                             run_mst, run_sa, run_camera_sa,
                             run_camera_spatial, run_word2vec_legacy)


# ---------------------------------------------------------------------------
# Helpers
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


def _eval_embeddings(emb, w, h):
    """Evaluate embedding quality: PCA Procrustes disparity + K-neighbor metrics."""
    from scipy.spatial import procrustes as scipy_procrustes, cKDTree
    n = emb.shape[0]
    grid = np.column_stack([np.arange(n) % w, np.arange(n) // w]).astype(np.float64)

    # PCA Procrustes
    _, _, Vt = np.linalg.svd(emb, full_matrices=False)
    pca_2d = (emb @ Vt[:2].T).astype(np.float64)
    _, _, pca_disp = scipy_procrustes(grid, pca_2d)

    # K=10 neighbors in embedding space → grid distance
    tree = cKDTree(emb)
    _, idx = tree.query(emb, k=11)
    idx = idx[:, 1:]  # remove self
    gx = np.arange(n) % w
    gy = np.arange(n) // w
    dists = np.abs(gx[idx] - gx[:, None]) + np.abs(gy[idx] - gy[:, None])
    mean_dist = float(dists.mean())
    within_3 = float((dists <= 3).mean())
    within_5 = float((dists <= 5).mean())

    result = {
        "pca_disparity": round(pca_disp, 4),
        "k10_mean_dist": round(mean_dist, 2),
        "k10_within_3px": round(within_3 * 100, 1),
        "k10_within_5px": round(within_5 * 100, 1),
    }
    print(f"  eval: PCA={pca_disp:.4f} K10: mean={mean_dist:.2f} "
          f"<3px={within_3*100:.1f}% <5px={within_5*100:.1f}%")
    return result


def _save_results_and_model(output_dir, args, dsolver, w, h, t0, max_frames,
                            total_pairs=None, wlog=None, n_sensory=None):
    """Common end-of-run: eval, info.json, model save."""
    import time
    if output_dir:
        s = dsolver.stats()
        results = {
            "ticks": max_frames,
            "std": round(s['std'], 4),
            "elapsed": round(time.time() - t0, 1),
        }
        if total_pairs is not None:
            results["total_pairs"] = total_pairs
        if getattr(args, 'eval', False):
            emb = dsolver.get_positions()
            if n_sensory is not None:
                emb = emb[:n_sensory]
            results["eval"] = _eval_embeddings(emb, w, h)
            if wlog:
                e = results["eval"]
                wlog.log_eval(e["pca_disparity"], e["k10_mean_dist"],
                              e["k10_within_3px"], e["k10_within_5px"])
        if dsolver.knn_k > 0:
            results["knn"] = {
                "K": dsolver.knn_k,
                "history": dsolver.get_knn_history(),
            }
        _save_run_info(output_dir, args, results=results, wlog=wlog)

    # Save KNN lists alongside model
    if output_dir and dsolver.knn_k > 0:
        knn_path = os.path.join(output_dir, "knn_lists.npy")
        np.save(knn_path, dsolver.get_knn_lists())
        print(f"  knn saved: {knn_path}")

    if args.save_model or args.save_model_path:
        save_path = args.save_model_path
        if not save_path and output_dir:
            save_path = os.path.join(output_dir, "model.npy")
        if save_path:
            np.save(save_path, dsolver.get_positions())
            print(f"  model saved: {save_path}")


def _save_run_info(output_dir, args, results=None, wlog=None):
    """Save/update info.json with command, parameters, and results."""
    import json, subprocess, datetime
    info_path = os.path.join(output_dir, "info.json")

    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        # Git hash for reproducibility
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_hash = None

        info = {
            "command": " ".join(sys.argv),
            "timestamp": datetime.datetime.now().isoformat(),
            "git_hash": git_hash,
            "args": {k: v for k, v in vars(args).items()
                     if not k.startswith("_") and k != "func"},
        }

    if results:
        info["results"] = results

    if wlog and wlog.run_url:
        info["wandb_url"] = wlog.run_url

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)


from cluster_manager import ClusterManager
from training_loop import run_training_loop


def run_word2vec(args):
    from utils.wandb_logger import WandbLogger
    # Parse comma-separated tags
    if getattr(args, 'wandb_tags', None):
        args.wandb_tags = [t.strip() for t in args.wandb_tags.split(',')]
    wlog = WandbLogger(args)

    w, h = args.width, args.height

    image = None
    if args.image:
        image = _load_image(args.image, w, h)

    mode = args.mode
    norm_every = args.normalize_every

    assert not getattr(args, 'use_mse', False), \
        "--use-mse is deprecated. Use --use-deriv-corr with threshold=0.5 instead."

    # Resolve anchor_sample from --anchor-sample or --anchor-batches
    if getattr(args, 'anchor_sample', None) is not None:
        anchor_sample = args.anchor_sample
    elif getattr(args, 'anchor_batches', None) is not None:
        anchor_sample = args.anchor_batches * args.batch_size
    else:
        anchor_sample = args.batch_size  # default: 1 batch

    if mode in ("sentence", "correlation"):
        import time
        import torch

        sig_channels = getattr(args, 'signal_channels', 1)
        n = w * h * sig_channels
        n_sensory = n

        # --- Feedback loop: column outputs → feedback neurons ---
        column_outputs = getattr(args, 'column_outputs', 0)
        column_feedback = getattr(args, 'column_feedback', False)
        cluster_m = getattr(args, 'cluster_m', 0)
        neurons_per = getattr(args, 'cluster_neurons_per', 0)

        # Auto-compute M from neurons-per-cluster target
        if cluster_m == 0 and neurons_per > 0 and column_outputs > 0:
            cluster_m = n_sensory // (neurons_per - column_outputs)
            args.cluster_m = cluster_m

        K = 0
        if column_feedback and cluster_m > 0 and column_outputs > 0:
            K = cluster_m * column_outputs
            n = n_sensory + K
            print(f"Column feedback: K={K} feedback neurons, "
                  f"n_sensory={n_sensory}, n_total={n}")

        # --- Pixel values for rendering (no model dependency) ---
        pixel_values = None
        if image is not None:
            if sig_channels > 1:
                channel_tints = {
                    0: np.array([0.3, 0.3, 1.0]),  # R -> red (BGR)
                    1: np.array([0.3, 1.0, 0.3]),  # G -> green (BGR)
                    2: np.array([1.0, 0.3, 0.3]),  # B -> blue (BGR)
                    3: np.array([0.8, 0.8, 0.8]),  # GS -> gray (BGR)
                }
                pixel_values = np.zeros((n_sensory, 3), dtype=np.uint8)
                for i in range(n_sensory):
                    px = i // sig_channels
                    ch = i % sig_channels
                    gray = float(image[px // w, px % w])
                    tint = channel_tints.get(ch, np.array([1.0, 1.0, 1.0]))
                    pixel_values[i] = np.clip(gray * tint, 0, 255).astype(np.uint8)
            else:
                pixel_values = np.zeros(n_sensory, dtype=np.uint8)
                for i in range(n_sensory):
                    pixel_values[i] = image[i // w, i % w]
        elif args.signal_source and (
                args.signal_source.endswith('.png') or
                args.signal_source.endswith('.jpg') or
                args.signal_source.endswith('.npy')):
            # Center crop from signal source as static pixel values
            if args.signal_source.endswith('.npy'):
                src = np.load(args.signal_source)
            else:
                img_bgr = cv2.imread(args.signal_source)
                src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if src.ndim == 3:
                src_h, src_w = src.shape[:2]
            else:
                src_h, src_w = src.shape
            cy, cx = (src_h - h) // 2, (src_w - w) // 2
            crop = src[cy:cy+h, cx:cx+w]
            raw = crop.ravel()[:n_sensory]
            vmin, vmax = raw.min(), raw.max()
            if vmax > vmin:
                pixel_values = ((raw - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                pixel_values = np.full(n_sensory, 128, dtype=np.uint8)

        render_w = w * sig_channels
        render_h = h

        output_dir = args.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            _save_run_info(output_dir, args)

        render_method = args.render
        if render_method == 'euclidean':
            render_method = 'pca'

        # --- Create renderer ---
        from render_server import Renderer
        viz_address = getattr(args, 'viz_address', None)
        field_address = getattr(args, 'field_address', None)
        renderer = Renderer(output_dir, render_w, render_h,
                            sig_channels=sig_channels,
                            viz_address=viz_address,
                            field_address=field_address) if output_dir else None

        # --- Create solver (imports torch/cuml in main process) ---
        if mode == "sentence":
            k = min(args.k, n - 1)
            top_k = topk_decay2d(w, h, k)
            device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
            dsolver = DriftSolver(n, top_k=top_k, dims=args.dims, lr=args.lr,
                                  mode='dot', k_neg=args.k_neg,
                                  normalize_every=norm_every, device=device)
        else:
            knn_k = getattr(args, 'knn_track', 0)
            lr_decay = getattr(args, 'lr_decay', 1.0)
            knn_nofn = getattr(args, 'knn_nofn', False)
            device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
            dsolver = DriftSolver(n, top_k=None, k=args.k, dims=args.dims,
                                  lr=args.lr, mode='dot', k_neg=args.k_neg,
                                  normalize_every=norm_every, device=device,
                                  knn_k=knn_k, lr_decay=lr_decay,
                                  knn_nofn=knn_nofn)

        if args.warm_start:
            warm = np.load(args.warm_start)
            if K > 0 and warm.shape[0] == n_sensory:
                fb_init = np.random.randn(K, args.dims).astype(np.float32) * 0.01
                warm = np.concatenate([warm, fb_init], axis=0)
            dsolver.positions = torch.from_numpy(warm).to(dsolver.device)
            print(f"Warm start from {args.warm_start} ({warm.shape})")

        # --- Correlation mode: build signal buffer ---
        saccade_source = None
        if mode == "correlation":
            T = args.signal_T
            sigma = args.signal_sigma
            print(f"Word2vec drift (correlation): {w}x{h} grid, "
                  f"k_sample={args.k_sample}, threshold={args.threshold}, "
                  f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}, "
                  f"signal: T={T}, sigma={sigma}, "
                  f"normalize_every={norm_every}, align={args.align}")

            signals_np = np.random.rand(n, T).astype(np.float32) if K > 0 \
                else np.zeros((n, T), dtype=np.float32)

            # Synthetic benchmark signals
            from benchmarks import get_benchmark
            bench = get_benchmark(args.signal_source) if args.signal_source else None
            bench_signal = None
            bench_metadata = None
            if bench is not None:
                bench_signal, bench_metadata = bench.make_signal(w, h, args)
                if bench_signal is not None:
                    if '_refs' in bench_metadata and renderer is not None:
                        bench_metadata['_refs']['renderer'] = renderer
                    for t in range(T):
                        signals_np[:n_sensory, t] = bench_signal(t)

            if (bench is None or bench_signal is None) and args.signal_source:
                if args.signal_source.endswith('.png') or args.signal_source.endswith('.jpg'):
                    img_bgr = cv2.imread(args.signal_source)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    if sig_channels == 4:
                        gs = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]
                        source = np.concatenate([img_rgb, gs[:,:,np.newaxis]], axis=2)
                    elif sig_channels == 3:
                        source = img_rgb
                    else:
                        source = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                else:
                    source = np.load(args.signal_source)

                if source.ndim == 3:
                    src_h, src_w, src_c = source.shape
                    assert src_c == sig_channels, \
                        f"Source has {src_c} channels but --signal-channels={sig_channels}"
                    crop_h, crop_w = h, w
                    print(f"  multi-channel: {sig_channels}ch, "
                          f"crop={crop_w}x{crop_h} pixels -> {n} neurons")
                else:
                    src_h, src_w = source.shape
                    crop_h, crop_w = h, w

                max_dy = src_h - crop_h
                max_dx = src_w - crop_w
                assert max_dy > 0 and max_dx > 0, \
                    f"Source {src_w}x{src_h} too small for {crop_w}x{crop_h} crop"
                saccade_step = args.saccade_step
                use_raw = args.use_mse or args.use_deriv_corr
                walk_dy = np.random.randint(0, max_dy + 1)
                walk_dx = np.random.randint(0, max_dx + 1)
                for t in range(T):
                    walk_dy = np.clip(walk_dy + np.random.randint(-saccade_step, saccade_step + 1),
                                      0, max_dy)
                    walk_dx = np.clip(walk_dx + np.random.randint(-saccade_step, saccade_step + 1),
                                      0, max_dx)
                    crop = source[walk_dy:walk_dy+crop_h, walk_dx:walk_dx+crop_w].ravel()
                    if use_raw:
                        signals_np[:n_sensory, t] = crop
                    else:
                        signals_np[:n_sensory, t] = crop - crop.mean()
                src_desc = f"{src_w}x{src_h}" + (f"x{sig_channels}" if source.ndim == 3 else "")
                crop_desc = f"{crop_w}x{crop_h}" + (f"x{sig_channels}" if source.ndim == 3 else "")
                mean_sub = "raw" if use_raw else "mean-subtracted"
                print(f"  signal buffer: ({n}, {T}), rolling saccades from "
                      f"{args.signal_source} ({src_desc}), "
                      f"crop={crop_desc}, step={saccade_step}, {mean_sub}")
                saccade_source = torch.from_numpy(source).to(dsolver.device)
            elif bench_signal is None:
                from scipy.ndimage import gaussian_filter
                for t in range(T):
                    noise = np.random.randn(h, w).astype(np.float32)
                    smoothed = gaussian_filter(noise, sigma=sigma)
                    signals_np[:n_sensory, t] = smoothed.ravel()
                print(f"  signal buffer: ({n}, {T}), spatial sigma={sigma}")

            signals = torch.from_numpy(signals_np).to(dsolver.device)
            tick_counter = [0]
        else:
            print(f"Word2vec drift (sentence): {w}x{h} grid, k={args.k}, "
                  f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}, "
                  f"window={args.window}, normalize_every={norm_every}, "
                  f"align={args.align}, async={args.async_render}")

        # --- Motor control setup ---
        motor_col_id = getattr(args, 'motor_column', -1)
        motor_scale = getattr(args, 'motor_scale', 5.0)
        motor_log = [] if motor_col_id >= 0 else None
        # Proprioception: 6 override neurons (last 6 sensory neurons)
        # [0,1] = position_x, position_y normalized
        # [2,3,4,5] = urgency for dx+, dx-, dy+, dy-
        motor_proprio = motor_col_id >= 0
        if motor_proprio:
            proprio_idx = list(range(n_sensory - 6, n_sensory))
            urgency = np.zeros(4, dtype=np.float32)  # dx+, dx-, dy+, dy-
            urgency_rate = 0.005  # ramp per tick when not moving (~200 ticks to 1.0)
            prev_walk = [walk_dx if saccade_source is not None else 0,
                         walk_dy if saccade_source is not None else 0]
            print(f"Motor proprioception: neurons {proprio_idx}, "
                  f"urgency_rate={urgency_rate}")

        # --- Predictive shift mixing ---
        _pred_shift_base = getattr(args, 'predictive_shift', 0)
        _pred_mix = getattr(args, 'predictive_mix', 0.0)
        def _tick_predictive_shift():
            if _pred_mix > 0 and _pred_shift_base > 0:
                return _pred_shift_base if np.random.random() < _pred_mix else 0
            return _pred_shift_base

        # --- Tick function ---
        if mode == "correlation":
            def do_tick():
                if bench_signal is not None:
                    col = tick_counter[0] % T
                    signals[:n_sensory, col] = torch.from_numpy(
                        bench_signal(tick_counter[0] + T)).to(signals.device)
                    tick_counter[0] += 1
                elif saccade_source is not None:
                    nonlocal walk_dy, walk_dx
                    # Motor bias from designated column
                    motor_dx, motor_dy = 0.0, 0.0
                    if (motor_col_id >= 0 and cluster_mgr is not None
                            and cluster_mgr.column_mgr is not None):
                        out = cluster_mgr.column_mgr.get_outputs()[motor_col_id]
                        motor_dx = (out[0] - out[1]) * motor_scale
                        motor_dy = (out[2] - out[3]) * motor_scale
                        if motor_log is not None:
                            motor_log.append((tick_counter[0], walk_dx, walk_dy,
                                              float(motor_dx), float(motor_dy),
                                              out.tolist()))
                    old_dx, old_dy = walk_dx, walk_dy
                    rand_dy = np.random.randint(-saccade_step, saccade_step + 1)
                    rand_dx = np.random.randint(-saccade_step, saccade_step + 1)
                    # Motor confidence suppresses random walk
                    if motor_proprio:
                        motor_mag = np.sqrt(motor_dx**2 + motor_dy**2)
                        confidence = min(1.0, motor_mag / motor_scale)
                        rand_scale = 1.0 - confidence
                        rand_dy = int(round(rand_dy * rand_scale))
                        rand_dx = int(round(rand_dx * rand_scale))
                    walk_dy = np.clip(walk_dy + rand_dy + int(round(motor_dy)),
                                      0, max_dy)
                    walk_dx = np.clip(walk_dx + rand_dx + int(round(motor_dx)),
                                      0, max_dx)
                    crop = saccade_source[walk_dy:walk_dy+crop_h, walk_dx:walk_dx+crop_w].reshape(-1)
                    col = tick_counter[0] % T
                    if use_raw:
                        signals[:n_sensory, col] = crop
                    else:
                        signals[:n_sensory, col] = crop - crop.mean()
                    # Override proprioception neurons
                    if motor_proprio:
                        dx_moved = walk_dx - old_dx
                        dy_moved = walk_dy - old_dy
                        # Update urgency: ramp up when not moving, reset on move
                        # [0]=dx+, [1]=dx-, [2]=dy+, [3]=dy-
                        moves = [dx_moved > 0, dx_moved < 0,
                                 dy_moved > 0, dy_moved < 0]
                        for i in range(4):
                            if moves[i]:
                                urgency[i] = 0.0
                            else:
                                urgency[i] = min(1.0, urgency[i] + urgency_rate)
                        # Write to signal buffer
                        pos_x = walk_dx / max(max_dx, 1)
                        pos_y = walk_dy / max(max_dy, 1)
                        proprio_vals = torch.tensor(
                            [pos_x, pos_y] + urgency.tolist(),
                            dtype=torch.float32, device=signals.device)
                        signals[proprio_idx, col] = proprio_vals
                    tick_counter[0] += 1
                return dsolver.tick_correlation(
                    signals, k_sample=args.k_sample,
                    threshold=args.threshold, window=args.window,
                    anchor_only=args.anchor_only,
                    use_covariance=args.use_covariance,
                    use_mse=args.use_mse,
                    use_deriv_corr=args.use_deriv_corr,
                    max_hit_ratio=args.max_hit_ratio,
                    batch_size=args.batch_size,
                    anchor_sample=anchor_sample,
                    fp16=getattr(args, 'fp16', False),
                    matmul_corr=getattr(args, 'matmul_corr', True),
                    predictive_shift=_tick_predictive_shift())
        else:
            def do_tick():
                dsolver.tick_sentence(window=args.window)
                return 0

        # --- Live clustering ---
        embed_render_mode = getattr(args, 'render_mode', 'grid') == 'embed'
        cluster_mgr = None
        cluster_m = getattr(args, 'cluster_m', 0)
        if cluster_m > 0:
            cluster_k2 = getattr(args, 'cluster_k2', 16)
            cluster_hyst = getattr(args, 'cluster_hysteresis', 0.0)
            knn2_mode = getattr(args, 'cluster_knn2_mode', 'incremental')
            centroid_mode = getattr(args, 'cluster_centroid_mode', 'nudge')
            cluster_max_k = getattr(args, 'cluster_max_k', 1)
            column_outputs = getattr(args, 'column_outputs', 0)
            column_config = {
                'type': getattr(args, 'column_type', 'default'),
                'n_outputs': column_outputs,
                'max_inputs': getattr(args, 'column_max_inputs', 20),
                'window': getattr(args, 'column_window', 4),
                'lr': getattr(args, 'column_lr', 0.05),
                'temperature': getattr(args, 'column_temperature', 0.5),
                'alpha': getattr(args, 'column_alpha', 0.01),
                'reseed_after': getattr(args, 'column_reseed_after', 1000),
                'match_threshold': getattr(args, 'column_match_threshold', 0.1),
                'streaming_decay': getattr(args, 'column_streaming_decay', 0.5),
                'lateral': getattr(args, 'column_lateral', False),
                'lateral_k': getattr(args, 'lateral_k', 6),
                'lateral_inputs': getattr(args, 'lateral_inputs', False),
                'lateral_input_k': getattr(args, 'lateral_input_k', 4),
                'eligibility': getattr(args, 'eligibility', False),
                'trace_decay': getattr(args, 'trace_decay', 0.95),
            } if column_outputs > 0 else None
            cluster_mgr = ClusterManager(
                n, cluster_m, w, h, k2=cluster_k2,
                lr=getattr(args, 'cluster_lr', 1.0),
                split_every=getattr(args, 'cluster_split_every', 10),
                max_k=cluster_max_k,
                knn2_mode=knn2_mode,
                centroid_mode=centroid_mode,
                hysteresis=cluster_hyst,
                track_history=getattr(args, 'cluster_track_history', False),
                render_mode=getattr(args, 'cluster_render_mode', 'color'),
                n_sensory=n_sensory,
                embed_render=embed_render_mode,
                embed_method=render_method,
                column_config=column_config,
                renderer=renderer,
                output_dir=output_dir, wlog=wlog)
            render_mode = getattr(args, 'cluster_render_mode', 'color')
            if render_mode in ('signal', 'both') or column_outputs > 0:
                cluster_mgr.set_signals(signals, sig_channels, T)
            cluster_mgr._pixel_values = pixel_values
            cluster_mgr._dsolver = dsolver
            # Give benchmark access to column manager for motor control
            if bench_metadata is not None and '_refs' in bench_metadata:
                bench_metadata['_refs']['column_mgr'] = cluster_mgr.column_mgr
                bench_metadata['_refs']['renderer'] = renderer
                bench_metadata['_refs']['dsolver'] = dsolver
            col_str = f", columns={column_outputs}out" if column_outputs > 0 else ""
            print(f"Live clustering enabled: m={cluster_m}, k2={cluster_k2}, "
                  f"max_k={cluster_max_k}, "
                  f"hysteresis={cluster_hyst}, knn2={knn2_mode}, "
                  f"centroid={centroid_mode}, render={render_mode}{col_str}, "
                  f"report_every={getattr(args, 'cluster_report_every', 1000)}")
            if motor_col_id >= 0:
                print(f"Motor control: column {motor_col_id}, scale={motor_scale}")
            # Warm-start cluster + column state
            warm_clusters = getattr(args, 'warm_start_clusters', None)
            if warm_clusters and os.path.isdir(warm_clusters):
                cluster_mgr.load_state(warm_clusters, dsolver.positions)

        # --- Training + rendering ---
        def on_save(tick, total_pairs):
            if renderer is not None and pixel_values is not None:
                emb = dsolver.get_positions()[:n_sensory]
                renderer.grid(tick, emb, pixel_values,
                              method=render_method,
                              align=args.align, gpu=args.render_gpu)

        t0, total_pairs = run_training_loop(
            do_tick, dsolver, max_frames=args.frames,
            sig_channels=sig_channels, wlog=wlog,
            log_every=getattr(args, 'log_every', 1000),
            knn_report_every=getattr(args, 'knn_report_every', 1000),
            cluster_report_every=getattr(args, 'cluster_report_every', 1000),
            save_every=getattr(args, 'save_every', 1000),
            n_sensory=n_sensory if K > 0 else None,
            w=w,
            on_save=on_save if output_dir else None,
            can_break=poll_quit, cluster_mgr=cluster_mgr,
            bench_metadata=bench_metadata,
            viz_every=getattr(args, 'viz_every', 0))

        elapsed = time.time() - t0
        s = dsolver.stats()
        print(f"Done: {args.frames} ticks in {elapsed:.1f}s, "
              f"std={s['std']:.4f}, total_pairs={total_pairs}")
        wlog.log_done(args.frames, elapsed, s['std'], total_pairs)

        # Final frame
        if renderer is not None and pixel_values is not None:
            emb = dsolver.get_positions()[:n_sensory]
            renderer.grid(args.frames, emb, pixel_values,
                          method=render_method,
                          align=args.align, gpu=args.render_gpu)

        if cluster_mgr is not None and cluster_mgr.initialized:
            cluster_mgr.report(args.frames)
            if output_dir:
                cluster_mgr.save(output_dir)

        # Benchmark analysis
        if bench is not None and bench_metadata is not None:
            bench_metadata['_tick_fn'] = bench_signal
            bench_metadata['_total_ticks'] = args.frames
            bench.analyze(bench_metadata, cluster_mgr, signals,
                          tick_counter, T, output_dir)

        # Motor log analysis
        if motor_log and output_dir:
            log_arr = np.array([(t, x, y, mdx, mdy) for t, x, y, mdx, mdy, _ in motor_log],
                               dtype=np.float32)
            np.save(os.path.join(output_dir, "motor_log.npy"), log_arr)
            positions = log_arr[:, 1:3].astype(int)
            if len(positions) > 0:
                motor_mag = np.sqrt(log_arr[:, 3]**2 + log_arr[:, 4]**2)
                hist_flat = np.histogram2d(positions[:, 1], positions[:, 0],
                                           bins=[int(positions[:, 1].max()) + 1,
                                                 int(positions[:, 0].max()) + 1])[0].ravel()
                uniformity = hist_flat.std() / max(hist_flat.mean(), 1e-8)
                print(f"  Motor: {len(motor_log)} ticks logged, "
                      f"mean|motor|={motor_mag.mean():.2f}, "
                      f"position uniformity={uniformity:.3f} "
                      f"(0=uniform, higher=concentrated)")
                if renderer is not None:
                    renderer.heatmap(positions)

        _save_results_and_model(output_dir, args, dsolver, render_w, render_h,
                               t0, args.frames, total_pairs=total_pairs,
                               wlog=wlog, n_sensory=n_sensory if K > 0 else None)
        wlog.finish()
        return

    # Legacy modes: similarity, dual, dual-xy, skipgram
    run_word2vec_legacy(args)


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
    p_greedy.add_argument("--save-every", type=int, default=1000,
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
    p_cont.add_argument("--save-every", type=int, default=1000,
                        help="Save every Nth frame (default: 1)")
    p_cont.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration via CuPy")
    p_cont.set_defaults(func=run_continuous)

    # --- word2vec ---
    p_w2v = sub.add_parser("word2vec",
                           help="Word2vec-style drift (skip-gram or similarity mode)")
    p_w2v.add_argument("--preset", type=str, default=None,
                       help="Load parameter preset from presets/ directory (e.g. 'gray_80x80'). "
                            "CLI args override preset values.")
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
    p_w2v.add_argument("--threshold", type=float, default=0.5,
                       help="Similarity/correlation threshold (default: 0.0)")
    # correlation mode args
    p_w2v.add_argument("--k-sample", type=int, default=200,
                       help="Random candidates to check per neuron (correlation mode, default: 50)")
    p_w2v.add_argument("--batch-size", type=int, default=256,
                       help="GPU batch size for anchor processing (default: 256)")
    anchor_group = p_w2v.add_mutually_exclusive_group()
    anchor_group.add_argument("--anchor-sample", type=int, default=None,
                       help="Total unique anchor neurons per tick (split into batch_size chunks)")
    anchor_group.add_argument("--anchor-batches", type=int, default=None,
                       help="Number of batches per tick (anchor_sample = anchor_batches * batch_size)")
    p_w2v.add_argument("--signal-T", type=int, default=1000,
                       help="Temporal signal buffer length (correlation mode, default: 100)")
    p_w2v.add_argument("--signal-sigma", type=float, default=3.0,
                       help="Gaussian smoothing sigma for signal generation (correlation mode, default: 3.0)")
    p_w2v.add_argument("--signal-source", type=str, default=None,
                       help="Path to signal source: .npy/.png/.jpg, or benchmark name (e.g. 'xor')")
    # Register benchmark-specific args
    from benchmarks import list_benchmarks, get_benchmark
    for bname in list_benchmarks():
        bmod = get_benchmark(bname)
        if bmod and hasattr(bmod, 'add_args'):
            bmod.add_args(p_w2v)
    p_w2v.add_argument("--signal-channels", type=int, default=1,
                       help="Channels per pixel from source: 1=gray, 3=RGB, 4=RGBG (default: 1). "
                            "n = W*H*channels. PNG auto-loads as RGB.")
    p_w2v.add_argument("--saccade-step", type=int, default=5,
                       help="Max pixels to shift per timestep in saccade mode (default: 5)")
    p_w2v.add_argument("--use-covariance", action="store_true",
                       help="Use covariance (corr×std1×std2) instead of Pearson correlation; downweights flat regions")
    p_w2v.add_argument("--use-mse", action="store_true",
                       help="Use MSE as distance metric (lower=more similar). "
                            "No per-frame global mean needed. Threshold ~0.02.")
    p_w2v.add_argument("--use-deriv-corr", action=argparse.BooleanOptionalAction, default=True,
                       help="Pearson correlation on temporal derivatives. "
                            "Dead neurons get score=0 (no variance gate needed). "
                            "Threshold ~0.3-0.5 (higher=more similar).")
    p_w2v.add_argument("--max-hit-ratio", type=float, default=None,
                       help="Discard anchors where neighbors/k_sample exceeds this ratio (e.g. 0.1). "
                            "Filters out global signals — if a neuron correlates with everyone, skip it.")
    p_w2v.add_argument("--anchor-only", action="store_true",
                       help="Correlation mode: only (anchor, neighbor) pairs, no transitive sliding window")
    p_w2v.add_argument("--knn-track", type=int, default=0,
                       help="Track per-neuron KNN list of this size (0=off). "
                            "Monitors embedding convergence via neighbor stability.")
    p_w2v.add_argument("--knn-report-every", type=int, default=1000,
                       help="Report KNN stability every N ticks (default: 1000)")
    p_w2v.add_argument("--log-every", type=int, default=1000,
                       help="Print tick progress every N ticks (default: 1000)")
    p_w2v.add_argument("--lr-decay", type=float, default=1.0,
                       help="Multiply lr by this factor at each normalization event (default: 1.0 = no decay)")
    p_w2v.add_argument("--knn-nofn", action="store_true",
                       help="Add neighbor-of-neighbor candidates to correlation probing. "
                            "Requires --knn-track. Breaks O(n²) scaling.")
    p_w2v.add_argument("--render", choices=["euclidean", "angular", "bestpc",
                                            "direct", "procrustes", "lstsq",
                                            "umap", "tsne", "spectral", "mds"],
                       default="euclidean",
                       help="Render projection: euclidean/pca (top-2 PCs), "
                            "bestpc (grid-correlated PCs), angular (unit norm + PCA), "
                            "direct (first 2 dims), procrustes/lstsq (supervised linear), "
                            "umap/tsne/spectral/mds (nonlinear) (default: euclidean)")
    p_w2v.add_argument("--render-mode", choices=["grid", "embed"],
                       default="grid",
                       help="'grid' (default), 'embed' saves additional embed_NNNNNN.png "
                            "scatter plots at cluster_report_every intervals")
    p_w2v.add_argument("--viz-address", type=str, default=None,
                       help="host:port for live graph visualization app (e.g., 192.168.1.5:9100)")
    p_w2v.add_argument("--viz-every", type=int, default=0,
                       help="Send graph to viz app every N ticks (0=disabled, 1=every tick)")
    p_w2v.add_argument("--field-address", type=str, default=None,
                       help="host:port for live field visualization (e.g., 192.168.1.5:9101)")
    p_w2v.add_argument("--align", action="store_true",
                       help="Procrustes-align rendered output to grid (fixes rotation/flip)")
    p_w2v.add_argument("--warm-start", type=str, default=None,
                       help="Load .npy embeddings as initial positions (warm start)")
    p_w2v.add_argument("--warm-start-clusters", type=str, default=None,
                       help="Load cluster+column state from this directory (full resume)")
    p_w2v.add_argument("--cold-projection", action=argparse.BooleanOptionalAction, default=True,
                       help="Run UMAP/t-SNE from scratch each frame (--no-cold-projection for warm start)")
    p_w2v.add_argument("--async-render", action="store_true", default=True,
                       help="Render in separate process (default: on)")
    p_w2v.add_argument("--sync-render", action="store_true",
                       help="Force synchronous rendering")
    p_w2v.add_argument("--eval", action="store_true",
                       help="Evaluate embeddings (PCA Procrustes + K-neighbor) and save to info.json")
    p_w2v.add_argument("--save-model", action="store_true",
                       help="Save final embeddings to .npy file")
    p_w2v.add_argument("--save-model-path", type=str, default=None,
                       help="Path for saved model (default: output_dir/model.npy)")
    # common args
    p_w2v.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    p_w2v.add_argument("--dims", type=int, default=8,
                       help="Position vector dimensionality (default: 8)")
    p_w2v.add_argument("--image", "-i", type=str, default=None,
                       help="Input image to scramble and reconstruct")
    p_w2v.add_argument("--frames", "-f", type=int, default=0,
                       help="Number of frames to run (0 = unlimited)")
    p_w2v.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Directory to save output frames as PNGs")
    p_w2v.add_argument("--save-every", type=int, default=1000,
                       help="Save every Nth frame (default: 1000)")
    p_w2v.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True,
                       help="Use GPU for solver (--no-gpu for CPU)")
    p_w2v.add_argument("--render-gpu", action=argparse.BooleanOptionalAction, default=True,
                       help="Use CuPy GPU for rendering (--no-render-gpu for CPU, default: CPU)")
    p_w2v.add_argument("--fp16", action="store_true",
                       help="Use float16 for correlation computation (faster on GPU, slight precision loss)")
    p_w2v.add_argument("--matmul-corr", action=argparse.BooleanOptionalAction, default=True,
                       help="Use matmul for correlation (default: on). "
                            "--no-matmul-corr uses gather path (less memory, better on CPU)")
    p_w2v.add_argument("--predictive-shift", type=int, default=0,
                       help="Predictive correlation: shift anchor signal by N ticks "
                            "(0=co-occurrence, 1=causal prediction)")
    p_w2v.add_argument("--predictive-mix", type=float, default=0.0,
                       help="Probability of using predictive shift per tick "
                            "(0.0=always co-occurrence, 0.1=10%% predictive, 1.0=always predictive)")
    # live clustering
    p_w2v.add_argument("--cluster-m", type=int, default=0,
                       help="Number of clusters (0=disabled)")
    p_w2v.add_argument("--cluster-k2", type=int, default=16,
                       help="Cluster-level KNN size (default: 16)")
    p_w2v.add_argument("--cluster-lr", type=float, default=1.0,
                       help="Centroid nudge learning rate (default: 1.0)")
    p_w2v.add_argument("--cluster-report-every", type=int, default=1000,
                       help="Save cluster visualization every N ticks (default: 1000)")
    p_w2v.add_argument("--cluster-split-every", type=int, default=10,
                       help="Attempt dead cluster recovery every N ticks (default: 10)")
    p_w2v.add_argument("--cluster-hysteresis", type=float, default=0.0,
                       help="Reassignment resistance: neuron must be (1-h)*dist closer to jump (default: 0.0)")
    p_w2v.add_argument("--cluster-knn2-mode", type=str, default='incremental',
                       choices=['incremental', 'knn'],
                       help="knn2 update strategy: 'incremental' (from pairs, no --knn-track needed) "
                            "or 'knn' (from neuron-level KNN lists, requires --knn-track)")
    p_w2v.add_argument("--cluster-centroid-mode", type=str, default='nudge',
                       choices=['exact', 'nudge'],
                       help="Centroid update: 'nudge' (lr-based drift toward member mean, default) "
                            "or 'exact' (incremental arithmetic, immediate — causes churn)")
    p_w2v.add_argument("--cluster-max-k", type=int, default=2,
                       help="Ring buffer depth for multi-cluster membership (default: 2)")
    p_w2v.add_argument("--cluster-track-history", action="store_true",
                       help="Save per-neuron cluster ID at each report interval")
    p_w2v.add_argument("--cluster-render-mode", type=str, default='color',
                       choices=['color', 'signal', 'both'],
                       help="Cluster visualization: 'color' (ID-based), 'signal' (mean neuron signal), 'both'")
    # column wiring (thalamus-to-cortex)
    p_w2v.add_argument("--column-type", type=str, default="default",
                       help="Column type: 'default' (softmax WTA) or 'conscience' (hard WTA + homeostatic)")
    p_w2v.add_argument("--column-outputs", type=int, default=4,
                       help="Column outputs per cluster (0=disabled, 4=enable with 4 outputs)")
    p_w2v.add_argument("--column-max-inputs", type=int, default=20,
                       help="Pre-allocated input slots per column (default: 20)")
    p_w2v.add_argument("--column-window", type=int, default=10,
                       help="Sliding window size for streaming columns (default: 10)")
    p_w2v.add_argument("--column-lr", type=float, default=0.05,
                       help="Column learning rate (default: 0.05)")
    p_w2v.add_argument("--column-temperature", type=float, default=0.2,
                       help="Column softmax temperature (default: 0.2)")
    p_w2v.add_argument("--column-match-threshold", type=float, default=0.1,
                       help="Column match threshold for dormant reassignment (default: 0.1)")
    p_w2v.add_argument("--column-streaming-decay", type=float, default=0.8,
                       help="Column streaming EMA decay (default: 0.8, rule of thumb: 1-2/window)")
    p_w2v.add_argument("--column-alpha", type=float, default=0.01,
                       help="Conscience threshold learning rate (default: 0.01)")
    p_w2v.add_argument("--column-reseed-after", type=int, default=1000,
                       help="Reseed dead units after N ticks without winning (default: 1000)")
    p_w2v.add_argument("--column-feedback", action="store_true",
                       help="Feed column outputs back as signal for feedback neurons")
    p_w2v.add_argument("--column-lateral", action="store_true",
                       help="Enable lateral connections between columns")
    p_w2v.add_argument("--lateral-k", type=int, default=2,
                       help="Lateral connections per column (default: 2)")
    p_w2v.add_argument("--lateral-sparsity", type=float, default=1.0,
                       help="Fraction of lateral connections to keep (1.0=full, 0.1=10%%)")
    p_w2v.add_argument("--lateral-inputs", action="store_true",
                       help="Enable lateral input connections between columns")
    p_w2v.add_argument("--lateral-input-k", type=int, default=4,
                       help="Lateral input connections per column (default: 4)")
    p_w2v.add_argument("--eligibility", action="store_true",
                       help="Enable eligibility traces on columns (reward-gated learning)")
    p_w2v.add_argument("--trace-decay", type=float, default=0.95,
                       help="Eligibility trace decay per tick (default: 0.95, ~20 tick window)")
    p_w2v.add_argument("--cluster-neurons-per", type=int, default=0,
                       help="Target neurons per cluster (auto-computes M from formula)")
    p_w2v.add_argument("--motor-column", type=int, default=-1,
                       help="Cluster whose column outputs steer saccade (-1=disabled, 0=first cluster)")
    p_w2v.add_argument("--motor-scale", type=float, default=5.0,
                       help="Motor output scale in pixels (default: 5.0)")
    # wandb logging
    p_w2v.add_argument("--wandb", action="store_true",
                       help="Log metrics to Weights & Biases")
    p_w2v.add_argument("--wandb-project", type=str, default="thalamus-sorter",
                       help="W&B project name (default: thalamus-sorter)")
    p_w2v.add_argument("--wandb-name", type=str, default=None,
                       help="W&B run name")
    p_w2v.add_argument("--wandb-group", type=str, default=None,
                       help="W&B run group")
    p_w2v.add_argument("--wandb-tags", type=str, default=None,
                       help="Comma-separated W&B tags")
    p_w2v.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity (team/user)")
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
    p_temp.add_argument("--save-every", type=int, default=1000,
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

    # Two-pass parsing: first extract --preset, then apply defaults from
    # preset file, then re-parse so CLI args override preset values.
    args, remaining = parser.parse_known_args()

    if hasattr(args, 'preset') and args.preset:
        import json
        preset_path = args.preset
        if not os.path.isabs(preset_path) and not os.path.exists(preset_path):
            # Look in presets/ subdirectory
            preset_path = os.path.join(os.path.dirname(__file__), 'presets', preset_path)
            if not preset_path.endswith('.json'):
                preset_path += '.json'
        with open(preset_path) as f:
            preset = json.load(f)
        print(f"Preset: {args.preset} -> {preset}")
        # Apply preset as defaults, CLI args will override
        # Find the subparser that matches the command
        subparser = sub.choices.get(args.command)
        if subparser:
            subparser.set_defaults(**preset)
        # Re-parse with preset defaults applied
        args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
