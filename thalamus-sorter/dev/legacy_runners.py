"""Legacy solver runners — greedy, continuous, temporal, MST, SA, camera modes,
and old word2vec modes (similarity/dual/dual-xy/skipgram).

These are kept for reference but are not actively maintained. The active
algorithm is word2vec correlation/sentence mode in main.py.
"""

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
from solvers.simulated_annealing import SimulatedAnnealing
from solvers.spatial_coherence import SpatialCoherence


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


def _mst_sort(weight_matrix, width, height, start=0):
    """Build max spanning tree and DFS-traverse for ordering."""
    tree_edges = build_mst(weight_matrix, maximize=True)
    graph = tree_to_adjacency(tree_edges)
    ordering = dfs_order(start, graph)
    return np.array(ordering).reshape(height, width)


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


def run_word2vec_legacy(args):
    """Legacy word2vec modes: similarity, dual, dual-xy, skipgram.
    Called from run_word2vec when mode is not correlation/sentence."""
    w, h = args.width, args.height
    mode = args.mode
    norm_every = args.normalize_every

    image = None
    if args.image:
        image = _load_image(args.image, w, h)

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
