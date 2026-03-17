"""GPU-accelerated drift solvers using PyTorch.

Supports two solver modes:
  - 'euclidean': ContinuousDrift — positions lerp toward neighbor centroids
  - 'dot': Word2vec dual-vector — W·C skip-gram with cross-updates

And tick variants:
  - tick():         standard (top-K positive + random negative)
  - tick_dual_xy(): alternating x/y neighbor signals for dimension specialization

Usage:
    from solvers.drift_torch import DriftSolver

    solver = DriftSolver(n=6400, top_k=top_k, dims=16, mode='euclidean')
    solver.tick()
    positions_np = solver.get_positions()  # back to numpy
"""

import torch
import numpy as np

# When True, KNN updates preserve existing entry positions (tenure-ordered).
# New candidates only replace the worst slot. When False, topk re-sorts by
# current cosine similarity (original behavior).
KNN_STABLE_INSERT = True


class DriftSolver:
    """Unified GPU drift solver for topographic map formation.

    Args:
        n: number of neurons
        top_k: (n, k) int32 array of neighbor indices (numpy or torch)
        k: number of neighbors (inferred from top_k if provided)
        dims: embedding dimensionality
        lr: learning rate
        mode: 'euclidean' or 'dot'
        k_neg: negative samples per tick (dot mode only)
        margin: dead zone for euclidean mode
        normalize_every: periodic L2 normalization interval (dot mode, 0=off)
        device: 'cuda', 'cpu', or specific device
    """

    def __init__(self, n, top_k=None, k=24, dims=3, lr=0.05,
                 mode='euclidean', k_neg=5, margin=0.1,
                 normalize_every=0, device='cuda', knn_k=0,
                 lr_decay=1.0, knn_nofn=False):
        self.n = n
        self.dims = dims
        self.lr = lr
        self.lr_decay = lr_decay
        self.knn_nofn = knn_nofn
        self.mode = mode
        self.k_neg = k_neg
        self.margin = margin
        self.normalize_every = normalize_every
        self._tick_count = 0
        self.device = torch.device(device)

        # Positions (W vectors)
        if mode == 'euclidean':
            self.positions = torch.randn(n, dims, device=self.device)
        else:
            scale = 0.5 / dims
            self.positions = torch.empty(n, dims, device=self.device).uniform_(-scale, scale)

        # Context vectors (dot mode only)
        self.contexts = torch.zeros(n, dims, device=self.device) if mode == 'dot' else None

        # Top-K neighbors
        if top_k is not None:
            if isinstance(top_k, np.ndarray):
                top_k = torch.from_numpy(top_k.astype(np.int64))
            self.top_k = top_k.to(self.device)
            self.k = self.top_k.shape[1]
        else:
            self.top_k = None
            self.k = k

        # Helper: arange for indexing
        self._arange = torch.arange(n, device=self.device)

        # Online KNN tracking
        self.knn_k = knn_k
        if knn_k > 0:
            self.knn_lists = torch.randint(0, n, (n, knn_k), device=self.device)
            self.knn_dists = torch.full((n, knn_k), -float('inf'), device=self.device)
            self._knn_prev = self.knn_lists.clone()
            self._knn_overlap_history = []
        else:
            self.knn_lists = None

    # ---- Euclidean mode ----

    def tick_euclidean(self):
        """Lerp toward neighbor centroid with dead zone + LayerNorm."""
        neighbor_pos = self.positions[self.top_k]       # (n, k, dims)
        centroids = neighbor_pos.mean(dim=1)             # (n, dims)
        delta = centroids - self.positions
        if self.margin > 0:
            dist = delta.norm(dim=1, keepdim=True)
            scale = torch.tanh(dist / self.margin)
            delta = delta * scale
        self.positions.add_(self.lr * delta)
        self._layernorm()

    def _layernorm(self):
        """Center + unit variance per dimension."""
        self.positions.sub_(self.positions.mean(dim=0))
        std = self.positions.std(dim=0)
        std = std.clamp(min=1e-8)
        self.positions.div_(std)

    # ---- Dot product (word2vec) mode ----

    def tick_dot(self):
        """Two-vector skip-gram: W←C and C←W cross-updates."""
        # Positive: sample one random neighbor from top_k
        pos_idx = torch.randint(0, self.k, (self.n,), device=self.device)
        j_pos = self.top_k[self._arange, pos_idx]

        c_pos = self.contexts[j_pos]                          # (n, dims)
        dot_pos = (self.positions * c_pos).sum(dim=1)          # (n,)
        sig_pos = torch.sigmoid(-dot_pos)                      # σ(-dot)

        # W[i] += lr * σ(-dot) * C[j]
        grad_w_pos = sig_pos.unsqueeze(1) * c_pos
        # C[j] += lr * σ(-dot) * W[i]  (use pre-update W)
        grad_c_pos = sig_pos.unsqueeze(1) * self.positions

        self.positions.add_(self.lr * grad_w_pos)
        # Scatter-add for C updates (multiple neurons may share same j)
        self.contexts.scatter_add_(0, j_pos.unsqueeze(1).expand_as(grad_c_pos),
                                   self.lr * grad_c_pos)

        # Negative: sample k_neg random neurons
        j_neg = torch.randint(0, self.n, (self.n, self.k_neg), device=self.device)
        c_neg = self.contexts[j_neg]                           # (n, k_neg, dims)
        dot_neg = (self.positions.unsqueeze(1) * c_neg).sum(dim=2)  # (n, k_neg)
        sig_neg = torch.sigmoid(dot_neg)                       # σ(dot)

        # W[i] -= lr * σ(dot) * C[neg]
        push_w = (sig_neg.unsqueeze(2) * c_neg).sum(dim=1)    # (n, dims)
        self.positions.sub_(self.lr * push_w)

        # C[neg] -= lr * σ(dot) * W[i]
        push_c = sig_neg.unsqueeze(2) * self.positions.unsqueeze(1)  # (n, k_neg, dims)
        # Flatten and scatter
        j_neg_flat = j_neg.reshape(-1)
        push_c_flat = (self.lr * push_c).reshape(-1, self.dims)
        self.contexts.scatter_add_(0, j_neg_flat.unsqueeze(1).expand_as(push_c_flat),
                                   -push_c_flat)

        self._maybe_normalize()

    def tick_sentence(self, window=5):
        """Skip-gram with sliding window over neighbor sentences, like gensim.

        Each neuron's sentence = [self, shuffled_neighbor1, ..., shuffled_neighbor_k].
        All (center, context) pairs within the window are batched into a single
        GPU operation — no Python loops over pairs.
        """
        seq_len = self.k + 1

        # Build sentences: (n, k+1) — neuron followed by shuffled neighbors
        perm = torch.argsort(torch.rand(self.n, self.k, device=self.device), dim=1)
        shuffled_nb = torch.gather(self.top_k, 1, perm)
        sentences = torch.cat([self._arange.unsqueeze(1), shuffled_nb], dim=1)  # (n, seq_len)

        # Build all valid (center_offset, context_offset) pairs
        if not hasattr(self, '_sentence_offsets'):
            offsets = []
            for c in range(seq_len):
                for ctx in range(max(0, c - window), min(seq_len, c + window + 1)):
                    if ctx != c:
                        offsets.append((c, ctx))
            self._sentence_offsets = torch.tensor(offsets, device=self.device)  # (P, 2)

        P = self._sentence_offsets.shape[0]  # number of pairs per sentence

        # Gather all center and context neuron IDs: (n, P)
        c_offs = self._sentence_offsets[:, 0]  # (P,)
        x_offs = self._sentence_offsets[:, 1]  # (P,)
        center_ids = sentences[:, c_offs]  # (n, P)
        ctx_ids = sentences[:, x_offs]     # (n, P)

        # Flatten to (n*P,) for batched processing
        center_flat = center_ids.reshape(-1)  # (n*P,)
        ctx_flat = ctx_ids.reshape(-1)        # (n*P,)
        B = center_flat.shape[0]

        # --- Positive updates ---
        w_center = self.positions[center_flat]     # (B, dims)
        c_ctx = self.contexts[ctx_flat]            # (B, dims)
        dot = (w_center * c_ctx).sum(dim=1)        # (B,)
        sig = torch.sigmoid(-dot)                  # (B,)

        grad_w = self.lr * sig.unsqueeze(1) * c_ctx    # (B, dims)
        grad_c = self.lr * sig.unsqueeze(1) * w_center  # (B, dims)

        self.positions.scatter_add_(
            0, center_flat.unsqueeze(1).expand_as(grad_w), grad_w)
        self.contexts.scatter_add_(
            0, ctx_flat.unsqueeze(1).expand_as(grad_c), grad_c)

        # --- Negative sampling: k_neg negatives per pair ---
        j_neg = torch.randint(0, self.n, (B, self.k_neg), device=self.device)  # (B, k_neg)
        c_neg = self.contexts[j_neg]                                   # (B, k_neg, dims)
        dot_neg = (w_center.unsqueeze(1) * c_neg).sum(dim=2)          # (B, k_neg)
        sig_neg = torch.sigmoid(dot_neg)                               # (B, k_neg)

        # W[center] -= lr * sum(σ(dot) * C[neg])
        push_w = (sig_neg.unsqueeze(2) * c_neg).sum(dim=1)            # (B, dims)
        self.positions.scatter_add_(
            0, center_flat.unsqueeze(1).expand_as(push_w), -self.lr * push_w)

        # C[neg] -= lr * σ(dot) * W[center]
        push_c = sig_neg.unsqueeze(2) * w_center.unsqueeze(1)         # (B, k_neg, dims)
        j_neg_flat = j_neg.reshape(-1)                                 # (B*k_neg,)
        push_c_flat = (self.lr * push_c).reshape(-1, self.dims)       # (B*k_neg, dims)
        self.contexts.scatter_add_(
            0, j_neg_flat.unsqueeze(1).expand_as(push_c_flat), -push_c_flat)

        self._maybe_normalize()

    def tick_correlation(self, signals, k_sample=50, threshold=0.5, window=5,
                         anchor_only=False, use_covariance=False,
                         use_mse=False, use_deriv_corr=False,
                         max_hit_ratio=None, batch_size=256,
                         anchor_sample=256, fp16=False,
                         matmul_corr=True):
        """Skip-gram from correlation-based neighbor discovery.

        Instead of precomputed top-K, each tick:
        1. Pick anchor neurons (all n split into anchor_batches chunks)
        2. For each, sample k_sample random candidates
        3. Compute similarity score (Pearson, covariance, or MSE-based)
        4. Keep only pairs above threshold
        5. Filter out anchors with too many hits (global signal, not local)
        6. Form variable-length sentences and run skip-gram

        Args:
            signals: (n, T) tensor of temporal signals on device
            k_sample: candidates to check per neuron
            threshold: minimum score to include as neighbor
            window: sliding window for skip-gram pairs
            anchor_only: if True, only generate (anchor, neighbor) pairs
                         instead of sliding window over full sentence
            use_covariance: if True, use covariance (corr × std1 × std2)
                           instead of Pearson correlation. Downweights
                           low-variance neurons (uniform regions).
            use_mse: if True, use MSE as distance metric.
                     Keep pairs where MSE < threshold.
                     No per-frame global mean needed.
            max_hit_ratio: if set, discard anchors where
                          good_neighbors/k_sample > this value. Filters out
                          neurons seeing global signals (correlated with
                          everyone) rather than local spatial structure.
            anchor_sample: total unique anchor neurons per tick.
                          Split into sequential chunks of batch_size.
                          Default 256 = one batch.
            fp16: if True, run correlation computation in float16 for speed.
            matmul_corr: if True (default), precompute normalized signals and
                use matmul for O(n²) but fast correlation. If False, gather
                candidate signals directly for O(batch*k_sample*T) with less
                total work but slower random memory access.
        """
        n = self.n
        compute_dtype = torch.float16 if fp16 else torch.float32

        # Matmul path: precompute normalized signals for all n neurons once,
        # then anchor @ normed.T gives (batch, n) — all pairwise correlations
        # via a single cuBLAS matmul. This is O(n²) — we compute correlations
        # against every neuron even though we only need k_sample of them. But
        # GPU matmul throughput >> random memory gather bandwidth, so the
        # "wasted" computation is free.
        #
        # Gather path (matmul_corr=False): gather (batch, k_sample, T) candidate
        # signals and compute correlations element-wise. O(batch * k_sample * T)
        # — less total work but slower due to random memory access. Better on
        # CPU or when k_sample << n and memory is tight.
        # TODO: incremental sig_normed update. Currently rebuilds the full
        # (n, T-1) normalized signal matrix every tick. Only one column of
        # `signals` changes per tick (the new saccade frame), which affects
        # exactly two derivative entries (col and col-1). Could patch those
        # two columns and incrementally update mean/norm instead of recomputing
        # from scratch. Negligible at 80x80 but ~100M elements at 320x320.
        if matmul_corr:
            if use_deriv_corr:
                sig = signals.to(compute_dtype)
                deriv = sig[:, 1:] - sig[:, :-1]
                centered = deriv - deriv.mean(dim=1, keepdim=True)
                norms = centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
                sig_normed = centered / norms
            elif use_mse:
                sig = signals.to(compute_dtype)
                T_len = sig.shape[1]
                sig_sq_mean = (sig * sig).mean(dim=1)
                sig_normed = sig
            elif use_covariance:
                sig = signals.to(compute_dtype)
                T_len = sig.shape[1]
                sig_normed = sig - sig.mean(dim=1, keepdim=True)
            else:
                sig = signals.to(compute_dtype)
                centered = sig - sig.mean(dim=1, keepdim=True)
                norms = centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
                sig_normed = centered / norms

        # Generate anchor chunks: anchor_sample unique anchors,
        # processed sequentially in chunks of batch_size
        total_anchors = min(anchor_sample, n)
        perm = torch.randperm(n, device=self.device)[:total_anchors]
        self._last_anchors = perm
        all_anchors = list(perm.split(batch_size))

        total_pairs = 0
        all_centers = []
        all_contexts = []
        for anchors in all_anchors:
            batch = anchors.shape[0]

            # Sample random candidates for each anchor
            candidates = torch.randint(0, n, (batch, k_sample), device=self.device)
            k_random = k_sample  # remember original random count for max_hit_ratio

            # Neighbor-of-neighbor sampling: add KNN-guided candidates
            if self.knn_k > 0 and self.knn_nofn:
                nofn = self._sample_nofn(anchors)  # (batch, nofn_count)
                candidates = torch.cat([candidates, nofn], dim=1)

            if matmul_corr:
                # Matmul path: (batch, T') @ (T', n) → (batch, n), then index
                anchor_normed = sig_normed[anchors]

                if use_mse:
                    cross = (anchor_normed @ sig_normed.T) / T_len
                    score_all = sig_sq_mean[anchors].unsqueeze(1) + sig_sq_mean.unsqueeze(0) - 2 * cross
                    score = score_all.gather(1, candidates).float()
                    mask = score < threshold
                elif use_covariance:
                    score_all = (anchor_normed @ sig_normed.T) / T_len
                    score = score_all.gather(1, candidates).float()
                    mask = score.abs() > threshold
                else:
                    score_all = anchor_normed @ sig_normed.T
                    score = score_all.gather(1, candidates).float()
                    if use_deriv_corr:
                        mask = score > threshold
                    else:
                        mask = score.abs() > threshold
            else:
                # Gather path: fetch candidate signals, compute element-wise
                anchor_sig = signals[anchors]
                cand_sig = signals[candidates]
                if fp16:
                    anchor_sig = anchor_sig.half()
                    cand_sig = cand_sig.half()

                if use_deriv_corr:
                    anchor_d = anchor_sig[:, 1:] - anchor_sig[:, :-1]
                    cand_d = cand_sig[:, :, 1:] - cand_sig[:, :, :-1]
                    anchor_dc = anchor_d - anchor_d.mean(dim=1, keepdim=True)
                    cand_dc = cand_d - cand_d.mean(dim=2, keepdim=True)
                    a_norm = anchor_dc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    c_norm = cand_dc.norm(dim=2, keepdim=True).clamp(min=1e-8)
                    score = (anchor_dc.unsqueeze(1) / a_norm.unsqueeze(1) *
                             cand_dc / c_norm).sum(dim=2).float()
                    mask = score > threshold
                elif use_mse:
                    diff = anchor_sig.unsqueeze(1) - cand_sig
                    score = (diff * diff).mean(dim=2).float()
                    mask = score < threshold
                elif use_covariance:
                    anchor_centered = anchor_sig - anchor_sig.mean(dim=1, keepdim=True)
                    cand_centered = cand_sig - cand_sig.mean(dim=2, keepdim=True)
                    T_len = anchor_sig.shape[1]
                    score = ((anchor_centered.unsqueeze(1) * cand_centered).sum(dim=2) / T_len).float()
                    mask = score.abs() > threshold
                else:
                    anchor_centered = anchor_sig - anchor_sig.mean(dim=1, keepdim=True)
                    cand_centered = cand_sig - cand_sig.mean(dim=2, keepdim=True)
                    a_norm = anchor_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    c_norm = cand_centered.norm(dim=2, keepdim=True).clamp(min=1e-8)
                    score = ((anchor_centered / a_norm).unsqueeze(1) *
                             (cand_centered / c_norm)).sum(dim=2).float()
                    mask = score.abs() > threshold

            # Per-anchor good neighbor counts
            good_counts = mask.sum(dim=1)  # (batch,)

            # Filter: discard anchors with too many hits (global signal detection)
            # Only count hits in the random portion — nofn candidates are expected
            # to pass threshold at high rates.
            if max_hit_ratio is not None:
                random_hits = mask[:, :k_random].sum(dim=1)
                max_hits = int(k_random * max_hit_ratio)
                too_popular = random_hits > max_hits
                good_counts[too_popular] = 0
                mask[too_popular] = False

            # Update online KNN with correlation-passing candidates
            self._update_knn(anchors, candidates, mask)

            max_good = int(good_counts.max().item())
            if max_good == 0:
                continue

            # Pack good candidates: sort mask so True values come first
            sorted_mask, sort_idx = mask.int().sort(dim=1, descending=True)
            sorted_cands = torch.gather(candidates, 1, sort_idx)
            trimmed = sorted_cands[:, :max_good]

            if anchor_only:
                pos_idx = torch.arange(max_good, device=self.device).unsqueeze(0)
                valid = pos_idx < good_counts.unsqueeze(1)
                valid_flat = valid.reshape(-1)
                anchor_exp = anchors.unsqueeze(1).expand_as(trimmed).reshape(-1)
                neighbor_flat = trimmed.reshape(-1)
                center_flat = anchor_exp[valid_flat]
                ctx_flat = neighbor_flat[valid_flat]
            else:
                # Sliding window over sentences: [anchor, nb1, nb2, ...]
                sentences = torch.cat([anchors.unsqueeze(1), trimmed], dim=1)
                seq_len = max_good + 1

                offsets = []
                for c in range(seq_len):
                    for ctx in range(max(0, c - window), min(seq_len, c + window + 1)):
                        if ctx != c:
                            offsets.append((c, ctx))
                if not offsets:
                    continue
                offset_t = torch.tensor(offsets, device=self.device)

                c_offs = offset_t[:, 0]
                x_offs = offset_t[:, 1]
                limits = (good_counts + 1).unsqueeze(1)
                valid = (c_offs.unsqueeze(0) < limits) & (x_offs.unsqueeze(0) < limits)

                center_ids = sentences[:, c_offs]
                ctx_ids = sentences[:, x_offs]

                valid_flat = valid.reshape(-1)
                center_flat = center_ids.reshape(-1)[valid_flat]
                ctx_flat = ctx_ids.reshape(-1)[valid_flat]

            B = center_flat.shape[0]
            if B == 0:
                continue

            # --- Positive updates ---
            w_center = self.positions[center_flat]
            c_ctx = self.contexts[ctx_flat]
            dot = (w_center * c_ctx).sum(dim=1)
            sig = torch.sigmoid(-dot)

            grad_w = self.lr * sig.unsqueeze(1) * c_ctx
            grad_c = self.lr * sig.unsqueeze(1) * w_center

            self.positions.scatter_add_(
                0, center_flat.unsqueeze(1).expand_as(grad_w), grad_w)
            self.contexts.scatter_add_(
                0, ctx_flat.unsqueeze(1).expand_as(grad_c), grad_c)

            # --- Negative sampling ---
            j_neg = torch.randint(0, self.n, (B, self.k_neg), device=self.device)
            c_neg = self.contexts[j_neg]
            dot_neg = (w_center.unsqueeze(1) * c_neg).sum(dim=2)
            sig_neg = torch.sigmoid(dot_neg)

            push_w = (sig_neg.unsqueeze(2) * c_neg).sum(dim=1)
            self.positions.scatter_add_(
                0, center_flat.unsqueeze(1).expand_as(push_w), -self.lr * push_w)

            push_c = sig_neg.unsqueeze(2) * w_center.unsqueeze(1)
            j_neg_flat = j_neg.reshape(-1)
            push_c_flat = (self.lr * push_c).reshape(-1, self.dims)
            self.contexts.scatter_add_(
                0, j_neg_flat.unsqueeze(1).expand_as(push_c_flat), -push_c_flat)

            all_centers.append(center_flat)
            all_contexts.append(ctx_flat)
            total_pairs += B

        # Expose pairs for external consumers (e.g. cluster knn2 update)
        if total_pairs > 0:
            self._last_pairs = (torch.cat(all_centers).cpu(),
                                torch.cat(all_contexts).cpu())

        self._maybe_normalize()
        return total_pairs

    def _maybe_normalize(self):
        """Periodic L2 normalization of W and C to prevent magnitude blow-up.
        Also applies lr_decay if set."""
        if self.normalize_every <= 0:
            return
        self._tick_count += 1
        if self._tick_count % self.normalize_every != 0:
            return
        for vecs in [self.positions, self.contexts]:
            if vecs is not None:
                norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                vecs.div_(norms)
        if self.lr_decay < 1.0:
            self.lr *= self.lr_decay

    # ---- Neighbor-of-neighbor sampling ----

    def _sample_nofn(self, anchors):
        """Generate neighbor-of-neighbor candidates for each anchor.

        For each anchor:
        1. Get its K neighbors from KNN list
        2. Get each neighbor's K neighbors (K² total)
        3. Remove the anchor itself and its direct neighbors
        4. Return unique candidates

        Returns: (batch, nofn_count) tensor, padded with self-references
                 (which get masked out by correlation check anyway).
        """
        batch = anchors.shape[0]
        K = self.knn_k

        # Get neighbors of each anchor: (batch, K)
        anchor_nbs = self.knn_lists[anchors]

        # Get neighbors-of-neighbors: (batch, K, K)
        nofn = self.knn_lists[anchor_nbs]

        # Reshape to (batch, K*K)
        nofn = nofn.reshape(batch, K * K)

        # Build mask of entries to exclude: anchor itself + direct neighbors
        # Exclude self
        is_self = (nofn == anchors.unsqueeze(1))
        # Exclude direct neighbors
        is_nb = (nofn.unsqueeze(2) == anchor_nbs.unsqueeze(1)).any(dim=2)
        exclude = is_self | is_nb

        # Replace excluded entries with a random neuron (will likely fail
        # correlation check, acting as extra random sampling)
        replacements = torch.randint(0, self.n, nofn.shape, device=self.device)
        nofn = torch.where(exclude, replacements, nofn)

        return nofn

    # ---- Online KNN tracking ----

    def _update_knn(self, anchors, candidates, mask):
        """Update per-neuron KNN lists with correlation-passing candidates.

        For each anchor, compute cosine similarity to passing candidates
        in W-vector space. Merge with existing KNN, keep top-K.
        """
        if self.knn_k <= 0:
            return

        anchor_w = self.positions[anchors]          # (batch, dims)
        cand_w = self.positions[candidates]          # (batch, k_sample, dims)

        a_norm = anchor_w / anchor_w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        c_norm = cand_w / cand_w.norm(dim=2, keepdim=True).clamp(min=1e-8)
        emb_sim = (a_norm.unsqueeze(1) * c_norm).sum(dim=2)  # (batch, k_sample)
        emb_sim[~mask] = -float('inf')

        cur_ids = self.knn_lists[anchors]           # (batch, knn_k)
        cur_dists = self.knn_dists[anchors]         # (batch, knn_k)

        if KNN_STABLE_INSERT:
            # Stable insert: replace worst slot in-place, preserving positions
            passing = emb_sim.clone()
            passing[candidates == anchors.unsqueeze(1)] = -float('inf')

            for _ in range(self.knn_k):
                cur_min_val, cur_min_pos = cur_dists.min(dim=1)
                best_new_val, best_new_pos = passing.max(dim=1)
                do_swap = best_new_val > cur_min_val
                if not do_swap.any():
                    break
                swap_idx = cur_min_pos[do_swap]
                batch_idx = torch.arange(cur_ids.size(0), device=self.device)[do_swap]
                cur_ids[batch_idx, swap_idx] = candidates[batch_idx, best_new_pos[do_swap]]
                cur_dists[batch_idx, swap_idx] = best_new_val[do_swap]
                passing[batch_idx, best_new_pos[do_swap]] = -float('inf')

            self.knn_lists[anchors] = cur_ids
            self.knn_dists[anchors] = cur_dists
        else:
            # Original: concat + topk (re-sorts by cosine similarity)
            all_ids = torch.cat([cur_ids, candidates], dim=1)
            all_dists = torch.cat([cur_dists, emb_sim], dim=1)
            all_dists[all_ids == anchors.unsqueeze(1)] = -float('inf')

            topk_dists, topk_idx = all_dists.topk(self.knn_k, dim=1)
            topk_ids = torch.gather(all_ids, 1, topk_idx)

            self.knn_lists[anchors] = topk_ids
            self.knn_dists[anchors] = topk_dists

    def _refresh_knn_dists(self):
        """Recompute cosine similarities for all KNN entries using current W vectors."""
        if self.knn_k <= 0:
            return
        w = self.positions / self.positions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        nb_w = w[self.knn_lists]  # (n, knn_k, dims)
        self.knn_dists = (w.unsqueeze(1) * nb_w).sum(dim=2)

    def knn_stability(self):
        """Compute overlap between current and previous KNN snapshot.

        Returns:
            overlap: float in [0, 1], fraction of KNN entries unchanged
            n_changed: int, number of neurons with at least one change
        """
        if self.knn_k <= 0:
            return 1.0, 0

        # For each neuron, count how many of prev's entries are in current
        matches = (self._knn_prev.unsqueeze(2) == self.knn_lists.unsqueeze(1))  # (n, K, K)
        per_neuron = matches.any(dim=2).sum(dim=1).float()  # (n,)

        overlap = float(per_neuron.mean().item()) / self.knn_k
        n_changed = int((per_neuron < self.knn_k).sum().item())

        # Per-neuron swap counts (K - matches = swaps)
        swaps = self.knn_k - per_neuron  # (n,)
        sorted_swaps = swaps.sort().values
        n = self.n
        # Overlap for top 50% most stable neurons (lowest swap count)
        top50_swaps = sorted_swaps[:n // 2].mean().item()
        # Overlap for top 90% most stable (bottom 10% excluded)
        top90_swaps = sorted_swaps[:n * 9 // 10].mean().item()

        self._knn_prev = self.knn_lists.clone()
        return overlap, n_changed, top50_swaps, top90_swaps

    def knn_spatial_accuracy(self, width, radius=3, channels=1):
        """Fraction of KNN neighbors that are within `radius` pixels on the grid.

        Args:
            width: grid width (neurons laid out row-major within each channel)
            radius: spatial proximity threshold in pixels
            channels: number of signal channels (neurons 0..w*h-1 = ch0, etc.)

        Returns:
            accuracy: float in [0, 1], mean fraction of K neighbors within radius
        """
        if self.knn_k <= 0:
            return 0.0

        hw = self.n // channels  # pixels per channel
        # Grid coords for all neurons (channel-aware: neuron i → pixel i % hw)
        pixel_ids = torch.arange(self.n, device=self.device) % hw
        gy = pixel_ids // width
        gx = pixel_ids % width
        ch = torch.arange(self.n, device=self.device) // hw

        # Coords of each neuron and its KNN neighbors
        nx, ny, nc = gx, gy, ch
        knn_flat = self.knn_lists  # (n, K)
        kx = gx[knn_flat]  # (n, K)
        ky = gy[knn_flat]
        kc = ch[knn_flat]

        # Spatial distance (only meaningful within same channel)
        dx = (nx.unsqueeze(1) - kx).float()
        dy = (ny.unsqueeze(1) - ky).float()
        dist = (dx * dx + dy * dy).sqrt()
        same_ch = (nc.unsqueeze(1) == kc)

        within = (dist <= radius) & same_ch
        return float(within.float().mean().item())

    def get_knn_lists(self):
        """Return KNN lists as numpy array (n, knn_k)."""
        if self.knn_k > 0:
            return self.knn_lists.detach().cpu().numpy()
        return None

    def get_knn_history(self):
        """Return list of (tick, overlap) tuples."""
        if self.knn_k > 0:
            return self._knn_overlap_history
        return []

    # ---- Dual-XY mode ----

    def setup_xy_topk(self, grid_coords, k=None):
        """Precompute x-primary and y-primary top-K from grid coordinates.

        Args:
            grid_coords: (n, 2) or (n, 3) array of grid positions (numpy)
            k: override neighbor count (default: self.k)
        """
        from scipy.spatial import cKDTree

        if isinstance(grid_coords, torch.Tensor):
            grid_coords = grid_coords.cpu().numpy()

        k = k or self.k

        # x-primary: x weight 1.0, y weight 0.3
        coords_x = grid_coords[:, :2].copy()
        coords_x[:, 1] *= 0.3
        tree_x = cKDTree(coords_x)
        _, idx_x = tree_x.query(coords_x, k=k + 1)
        self._top_k_x = torch.from_numpy(idx_x[:, 1:].astype(np.int64)).to(self.device)

        # y-primary: y weight 1.0, x weight 0.3
        coords_y = grid_coords[:, :2].copy()
        coords_y[:, 0] *= 0.3
        tree_y = cKDTree(coords_y)
        _, idx_y = tree_y.query(coords_y, k=k + 1)
        self._top_k_y = torch.from_numpy(idx_y[:, 1:].astype(np.int64)).to(self.device)

    def tick_dual_xy(self):
        """Alternate x/y neighbors for dimension specialization."""
        self.top_k = self._top_k_x if self._tick_count % 2 == 0 else self._top_k_y
        self.tick_dot()

    # ---- Unified tick ----

    def tick(self):
        """Run one tick using the configured mode."""
        if self.mode == 'euclidean':
            self.tick_euclidean()
        else:
            self.tick_dot()

    # ---- Data transfer ----

    def get_positions(self):
        """Return positions as numpy array."""
        return self.positions.detach().cpu().numpy()

    def get_contexts(self):
        """Return context vectors as numpy array (dot mode only)."""
        if self.contexts is not None:
            return self.contexts.detach().cpu().numpy()
        return None

    def set_top_k(self, top_k):
        """Set top-K neighbor indices from numpy array."""
        if isinstance(top_k, np.ndarray):
            top_k = torch.from_numpy(top_k.astype(np.int64))
        self.top_k = top_k.to(self.device)
        self.k = self.top_k.shape[1]

    # ---- Rendering (CPU, shared across modes) ----

    def render(self, size_or_shape, voxel_values=None, method='pca'):
        """Render positions to a grid via Voronoi assignment.

        Args:
            size_or_shape: int for square/cube, or tuple (W, H) / (W, H, D)
            voxel_values: (n,) values for each neuron (numpy)
            method: 'pca' (top PCs), 'bestpc' (best correlated PCs), 'direct' (first dims)

        Returns:
            grid array with shape matching size_or_shape
        """
        from scipy.spatial import cKDTree

        pos = self.get_positions()  # (n, dims)

        # Determine output shape
        if isinstance(size_or_shape, int):
            ndim = min(self.dims, 3)
            shape = (size_or_shape,) * ndim
        else:
            shape = tuple(size_or_shape)
        ndim_out = len(shape)

        # Project to output dimensionality
        pos_proj = self._project(pos, ndim_out, method, shape)

        # Scale to grid coordinates
        for d in range(ndim_out):
            mn, mx = pos_proj[:, d].min(), pos_proj[:, d].max()
            span = mx - mn if mx - mn > 1e-8 else 1.0
            pos_proj[:, d] = (pos_proj[:, d] - mn) / span * (shape[d] - 1)

        # Build grid points
        grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
        grid_points = np.column_stack([g.ravel().astype(np.float64) for g in grids])

        # Voronoi: nearest neuron for each grid cell
        tree = cKDTree(pos_proj.astype(np.float64))
        _, nearest = tree.query(grid_points)

        if voxel_values is not None:
            result = voxel_values[nearest]
            if voxel_values.ndim == 1:
                return result.reshape(shape)
            else:
                # Multi-channel (e.g. RGB): (n_grid, C) → (W, H, D, C)
                return result.reshape(shape + voxel_values.shape[1:])
        return nearest.reshape(shape)

    def _project(self, pos, ndim_out, method, shape=None):
        """Project positions to ndim_out dimensions."""
        if self.dims <= ndim_out:
            proj = np.zeros((self.n, ndim_out), dtype=np.float32)
            proj[:, :self.dims] = pos
            return proj

        if method == 'direct':
            return pos[:, :ndim_out].copy()

        # PCA
        _, _, Vt = np.linalg.svd(pos, full_matrices=False)

        if method == 'pca':
            return (pos @ Vt[:ndim_out].T).astype(np.float32)

        if method == 'bestpc':
            pcs = pos @ Vt.T
            # Need grid coordinates to correlate against
            if shape is not None and len(shape) >= 2:
                n = pos.shape[0]
                # Infer grid coords from shape
                idx = np.arange(n)
                grid_axes = []
                for d in range(len(shape)):
                    stride = int(np.prod(shape[d+1:])) if d + 1 < len(shape) else 1
                    grid_axes.append((idx // stride) % shape[d])

                best_pcs = []
                used = set()
                for ax in range(ndim_out):
                    best_corr, best_pc = 0, 0
                    for i in range(self.dims):
                        if i in used:
                            continue
                        c = abs(np.corrcoef(pcs[:, i], grid_axes[ax])[0, 1])
                        if c > best_corr:
                            best_corr = c
                            best_pc = i
                    used.add(best_pc)
                    best_pcs.append(best_pc)

                proj = np.column_stack([pcs[:, pc] for pc in best_pcs])
                # Flip if negatively correlated
                for ax in range(ndim_out):
                    if np.corrcoef(proj[:, ax], grid_axes[ax])[0, 1] < 0:
                        proj[:, ax] *= -1
                return proj.astype(np.float32)

            # Fallback to top PCs
            return (pos @ Vt[:ndim_out].T).astype(np.float32)

        return (pos @ Vt[:ndim_out].T).astype(np.float32)

    # ---- Stats ----

    def stats(self):
        """Return dict of position statistics."""
        pos = self.get_positions()
        result = {
            'std': float(np.std(pos)),
            'mean_norm': float(np.linalg.norm(np.mean(pos, axis=0))),
        }
        if self.contexts is not None:
            ctx = self.get_contexts()
            result['ctx_std'] = float(np.std(ctx))
        return result
