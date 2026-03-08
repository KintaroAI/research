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
import sys
import numpy as np
import cv2

from utils.weights import inverse_distance_1d, decay_distance_2d, OnlineWeightMatrix
from utils.correlation import temporal_correlation
from utils.graph import build_mst, tree_to_adjacency, dfs_order
from utils.camera import Camera
from utils.display import show_grid, show_vector, wait
from solvers.greedy_drift import GreedyDrift
from solvers.simulated_annealing import SimulatedAnnealing
from solvers.spatial_coherence import SpatialCoherence


# ---------------------------------------------------------------------------
# Grid-mode runners (pre-defined weight matrix, no camera)
# ---------------------------------------------------------------------------

def run_greedy(args):
    w, h = args.width, args.height
    print(f"Greedy drift: {w}x{h} grid, k={args.k}, "
          f"weights={args.weight_type}, decay={args.decay}")

    if args.weight_type == "decay2d":
        W = decay_distance_2d(w, h, decay_rate=args.decay)
    else:
        W = inverse_distance_1d(w * h)

    solver = GreedyDrift(w, h, W, k=args.k, move_fraction=args.move_fraction)

    tick = 0
    while True:
        solver.tick()
        tick += 1
        if tick % 10 == 0:
            show_grid("Sorted neurons", solver.neurons_matrix)
            wait()
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
        if cv2.waitKey(100) & 0xFF == ord('q'):
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
