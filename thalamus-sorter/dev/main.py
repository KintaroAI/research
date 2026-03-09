#!/usr/bin/env python3
"""Thalamus sorter: topographic map formation via correlation-based
neuron sorting. Explores multiple algorithms for arranging neurons
so that temporally correlated units become spatial neighbors.

Grid mode (no camera):
    python main.py greedy --width 80 --height 80 --k 24
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
