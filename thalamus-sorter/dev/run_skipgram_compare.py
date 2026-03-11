"""Compare gensim skip-gram (reference) vs our PyTorch skip-gram.

Both are fed the same neighbor structure from a 2D grid.
We compare: embedding quality, top-K overlap with true neighbors,
PC correlations with grid x/y, and reconstruction quality.

Usage:
    python run_skipgram_compare.py --size 40 --dims 16 --k 24
"""

import numpy as np
import time
import argparse
from scipy.spatial import cKDTree
from gensim.models import Word2Vec


def build_topk(size, k):
    """Compute top-K 2D neighbors by proximity."""
    n = size * size
    coords = np.zeros((n, 2), dtype=np.float32)
    coords[:, 0] = np.arange(n) % size
    coords[:, 1] = np.arange(n) // size
    tree = cKDTree(coords)
    _, top_k = tree.query(coords, k=k + 1)
    top_k = top_k[:, 1:].astype(np.int32)
    return coords, top_k


def build_sentences(top_k, n_epochs=10):
    """Build sentences from neighbor lists.
    Each sentence = [neuron_id, neighbor1, neighbor2, ...].
    Repeat n_epochs times with shuffling for better coverage."""
    n, k = top_k.shape
    sentences = []
    for epoch in range(n_epochs):
        order = np.random.permutation(n)
        for i in order:
            # Shuffle neighbors for variety
            neighbors = top_k[i].copy()
            np.random.shuffle(neighbors)
            sentence = [str(i)] + [str(j) for j in neighbors]
            sentences.append(sentence)
    return sentences


def analyze_embeddings(embeddings, size, label):
    """Analyze embedding quality: PC correlations, variance, top-K overlap."""
    n = size * size
    coords = np.zeros((n, 2), dtype=np.float32)
    coords[:, 0] = np.arange(n) % size
    coords[:, 1] = np.arange(n) // size

    grid_x = coords[:, 0]
    grid_y = coords[:, 1]

    # PCA
    _, _, Vt = np.linalg.svd(embeddings, full_matrices=False)
    pcs = embeddings @ Vt.T

    # Variance explained
    variances = np.var(pcs, axis=0)
    total_var = variances.sum()
    var_pct = variances / total_var * 100

    print(f"\n=== {label} ===")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Std: {np.std(embeddings):.4f}")
    print(f"Top-5 PC variance: {var_pct[:5].round(1)}")

    # PC correlations with grid x/y
    best_x_corr, best_x_pc = 0, 0
    best_y_corr, best_y_pc = 0, 0
    for i in range(min(embeddings.shape[1], 16)):
        cx = abs(np.corrcoef(pcs[:, i], grid_x)[0, 1])
        cy = abs(np.corrcoef(pcs[:, i], grid_y)[0, 1])
        if cx > best_x_corr:
            best_x_corr = cx
            best_x_pc = i
        if cy > best_y_corr:
            best_y_corr = cy
            best_y_pc = i

    print(f"Best PC→X: PC{best_x_pc} r={best_x_corr:.4f}")
    print(f"Best PC→Y: PC{best_y_pc} r={best_y_corr:.4f}")

    # Top-K overlap: do learned embedding neighbors match true grid neighbors?
    true_tree = cKDTree(coords)
    _, true_topk = true_tree.query(coords, k=25)
    true_topk = true_topk[:, 1:]

    emb_tree = cKDTree(embeddings)
    _, emb_topk = emb_tree.query(embeddings, k=25)
    emb_topk = emb_topk[:, 1:]

    overlap = np.mean([
        len(set(true_topk[i]) & set(emb_topk[i])) / 24
        for i in range(n)
    ])
    print(f"Top-24 neighbor overlap: {overlap*100:.1f}%")

    return {
        'best_x_corr': best_x_corr, 'best_y_corr': best_y_corr,
        'overlap': overlap, 'var_pct': var_pct,
    }


def run_gensim(sentences, n, dims, k, window, epochs, workers=4):
    """Run gensim Word2Vec skip-gram."""
    print(f"\nTraining gensim skip-gram: dims={dims}, window={window}, "
          f"epochs={epochs}, {len(sentences)} sentences")
    t0 = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=dims,
        window=window,
        min_count=1,
        sg=1,           # skip-gram
        hs=0,           # negative sampling
        negative=5,     # 5 negative samples
        workers=workers,
        epochs=epochs,
        seed=42,
    )
    elapsed = time.time() - t0
    print(f"Gensim done in {elapsed:.1f}s")

    # Extract embeddings in neuron order
    embeddings = np.zeros((n, dims), dtype=np.float32)
    for i in range(n):
        embeddings[i] = model.wv[str(i)]
    return embeddings


def run_ours(top_k, n, dims, k_neg, normalize_every, n_ticks, mode='dot',
             window=5, lr=0.05):
    """Run our PyTorch skip-gram."""
    from solvers.drift_torch import DriftSolver
    import torch

    label = f"sentence(w={window})" if mode == 'sentence' else "single-pair"
    print(f"\nTraining our skip-gram ({label}): dims={dims}, k_neg={k_neg}, "
          f"normalize_every={normalize_every}, lr={lr}, {n_ticks} ticks")
    t0 = time.time()

    solver = DriftSolver(n, top_k=top_k, dims=dims, lr=lr,
                         mode='dot', k_neg=k_neg,
                         normalize_every=normalize_every, device='cuda')
    for _ in range(n_ticks):
        if mode == 'sentence':
            solver.tick_sentence(window=window)
        else:
            solver.tick()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Our solver done in {elapsed:.1f}s")

    return solver.get_positions()


def render_comparison(embeddings_list, size, image_path, output_prefix):
    """Render embeddings as 2D images for visual comparison.

    Args:
        embeddings_list: list of (embedding_array, label) tuples
    """
    import cv2
    from scipy.spatial import cKDTree

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: could not load {image_path}, skipping render")
        return

    n = size * size
    pixel_values = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        pixel_values[i] = img[i // size, i % size]

    for emb, label in embeddings_list:
        # PCA to 2D
        _, _, Vt = np.linalg.svd(emb, full_matrices=False)
        pos_2d = (emb @ Vt[:2].T).astype(np.float64)

        for d in range(2):
            mn, mx = pos_2d[:, d].min(), pos_2d[:, d].max()
            span = mx - mn if mx - mn > 1e-8 else 1.0
            pos_2d[:, d] = (pos_2d[:, d] - mn) / span * (size - 1)

        grid_y, grid_x = np.mgrid[0:size, 0:size]
        grid_points = np.column_stack([
            grid_x.ravel().astype(np.float64),
            grid_y.ravel().astype(np.float64)
        ])
        tree = cKDTree(pos_2d)
        _, nearest = tree.query(grid_points)
        output = pixel_values[nearest].reshape(size, size)

        path = f"{output_prefix}_{label}.png"
        cv2.imwrite(path, output)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--dims", type=int, default=16)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--k-neg", type=int, default=5)
    parser.add_argument("--gensim-epochs", type=int, default=20)
    parser.add_argument("--sentence-epochs", type=int, default=10)
    parser.add_argument("--our-ticks", type=int, default=50000)
    parser.add_argument("--normalize-every", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("-i", "--image", default=None)
    parser.add_argument("-o", "--output-prefix", default="output_7_compare")
    args = parser.parse_args()

    np.random.seed(42)
    n = args.size ** 2

    # Build neighbor structure
    print(f"Grid: {args.size}x{args.size} = {n} neurons, k={args.k}, dims={args.dims}")
    coords, top_k = build_topk(args.size, args.k)

    # Build sentences from neighbor lists
    sentences = build_sentences(top_k, n_epochs=args.sentence_epochs)
    print(f"Built {len(sentences)} sentences (len={args.k+1} each)")

    # Run gensim
    emb_gensim = run_gensim(sentences, n, args.dims, args.k,
                            window=args.window, epochs=args.gensim_epochs)
    stats_gensim = analyze_embeddings(emb_gensim, args.size, "Gensim skip-gram")

    # Run ours (single-pair, original) — needs normalization to avoid blow-up
    norm_single = args.normalize_every if args.normalize_every > 0 else 100
    emb_single = run_ours(top_k, n, args.dims, args.k_neg,
                          norm_single, args.our_ticks, mode='dot')
    stats_single = analyze_embeddings(emb_single, args.size,
                                      "Our skip-gram (single-pair)")

    # Run ours (sentence mode, mimicking gensim)
    # Each tick_sentence processes ~250 pairs (25 positions × ~10 context each)
    # So fewer ticks needed to match gensim's training volume
    sentence_ticks = max(200, args.our_ticks // 50)
    # Use smaller lr for sentence mode — many more updates per tick
    emb_sentence = run_ours(top_k, n, args.dims, args.k_neg,
                            args.normalize_every, sentence_ticks,
                            mode='sentence', window=args.window, lr=0.005)
    stats_sentence = analyze_embeddings(emb_sentence, args.size,
                                        "Our skip-gram (sentence)")

    # Summary
    print("\n=== COMPARISON ===")
    print(f"{'':20s} {'Gensim':>10s} {'Single':>10s} {'Sentence':>10s}")
    print(f"{'PC→X correlation':20s} {stats_gensim['best_x_corr']:10.4f} "
          f"{stats_single['best_x_corr']:10.4f} {stats_sentence['best_x_corr']:10.4f}")
    print(f"{'PC→Y correlation':20s} {stats_gensim['best_y_corr']:10.4f} "
          f"{stats_single['best_y_corr']:10.4f} {stats_sentence['best_y_corr']:10.4f}")
    print(f"{'Top-24 overlap':20s} {stats_gensim['overlap']*100:9.1f}% "
          f"{stats_single['overlap']*100:9.1f}% {stats_sentence['overlap']*100:9.1f}%")

    # Render if image provided
    if args.image:
        render_comparison(
            [(emb_gensim, 'gensim'), (emb_single, 'single'),
             (emb_sentence, 'sentence')],
            args.size, args.image, args.output_prefix)


if __name__ == "__main__":
    main()
