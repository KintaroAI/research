"""GPU-accelerated drift solver using PyTorch.

Core solver for correlation-based skip-gram embedding training.
For experimental tick modes (euclidean, dot, sentence, dual_xy) and
advanced KNN tracking/visualization, see drift_torch_experiments.py.

Usage:
    from solvers.drift_torch import DriftSolver

    solver = DriftSolver(n=6400, dims=16)
    pairs = solver.tick_correlation(signals, k_sample=200, threshold=0.5)
    positions_np = solver.get_positions()
"""

import torch
import numpy as np

# When True, KNN updates preserve existing entry positions (tenure-ordered).
# New candidates only replace the worst slot. When False, topk re-sorts by
# current cosine similarity (original behavior).
KNN_STABLE_INSERT = True


class DriftSolver:
    """GPU drift solver for topographic map formation via correlation skip-gram.

    Args:
        n: number of neurons
        top_k: (n, k) int32 array of neighbor indices (numpy or torch)
        k: number of neighbors (inferred from top_k if provided)
        dims: embedding dimensionality
        lr: learning rate
        mode: 'dot' (only mode supported in this file)
        k_neg: negative samples per tick
        normalize_every: periodic L2 normalization interval (0=off)
        device: 'cuda', 'cpu', or specific device
        knn_k: online KNN tracking size (0=disabled)
        lr_decay: multiplicative LR decay per normalize_every steps
        knn_nofn: enable neighbor-of-neighbor sampling
    """

    def __init__(self, n, top_k=None, k=24, dims=3, lr=0.05,
                 mode='dot', k_neg=5, margin=0.1,
                 normalize_every=0, device='cuda', knn_k=0,
                 lr_decay=1.0, knn_nofn=False):
        self.n = n
        self.dims = dims
        self.lr = lr
        self.lr_decay = lr_decay
        self.knn_nofn = knn_nofn
        self.mode = mode
        self.k_neg = k_neg
        self.normalize_every = normalize_every
        self._tick_count = 0
        self.device = torch.device(device)

        # Positions (W vectors)
        scale = 0.5 / dims
        self.positions = torch.empty(n, dims, device=self.device).uniform_(-scale, scale)

        # Context vectors (C vectors for skip-gram)
        self.contexts = torch.zeros(n, dims, device=self.device)

        # Top-K neighbors (optional, for non-correlation modes)
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

    def tick_correlation(self, signals, k_sample=50, threshold=0.5, window=5,
                         anchor_only=False, use_covariance=False,
                         use_mse=False, use_deriv_corr=False,
                         max_hit_ratio=None, batch_size=256,
                         anchor_sample=256, fp16=False,
                         matmul_corr=True, predictive_shift=0):
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
            use_covariance: use covariance instead of Pearson correlation
            use_mse: use MSE as distance metric
            use_deriv_corr: correlate temporal derivatives
            max_hit_ratio: discard anchors with too many hits
            anchor_sample: total unique anchor neurons per tick
            batch_size: anchors per sequential chunk
            fp16: run correlation in float16
            matmul_corr: use matmul path for correlation
            predictive_shift: causal/predictive correlation shift
        """
        if predictive_shift > 0:
            assert matmul_corr and use_deriv_corr, (
                "predictive_shift requires matmul_corr=True and use_deriv_corr=True")

        n = self.n
        compute_dtype = torch.float16 if fp16 else torch.float32
        signals = signals.to(self.device)

        if matmul_corr:
            if use_deriv_corr:
                sig = signals.to(self.device, dtype=compute_dtype)
                deriv = sig[:, 1:] - sig[:, :-1]
                if predictive_shift > 0:
                    s = predictive_shift
                    anchor_deriv = deriv[:, :-s]
                    cand_deriv = deriv[:, s:]
                    a_c = anchor_deriv - anchor_deriv.mean(dim=1, keepdim=True)
                    a_n = a_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sig_normed_anchor = a_c / a_n
                    c_c = cand_deriv - cand_deriv.mean(dim=1, keepdim=True)
                    c_n = c_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sig_normed = c_c / c_n
                else:
                    centered = deriv - deriv.mean(dim=1, keepdim=True)
                    norms = centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sig_normed = centered / norms
            elif use_mse:
                sig = signals.to(self.device, dtype=compute_dtype)
                T_len = sig.shape[1]
                sig_sq_mean = (sig * sig).mean(dim=1)
                sig_normed = sig
            elif use_covariance:
                sig = signals.to(self.device, dtype=compute_dtype)
                T_len = sig.shape[1]
                sig_normed = sig - sig.mean(dim=1, keepdim=True)
            else:
                sig = signals.to(self.device, dtype=compute_dtype)
                centered = sig - sig.mean(dim=1, keepdim=True)
                norms = centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
                sig_normed = centered / norms

        # Generate anchor chunks
        total_anchors = min(anchor_sample, n)
        perm = torch.randperm(n, device=self.device)[:total_anchors]
        self._last_anchors = perm
        all_anchors = list(perm.split(batch_size))

        total_pairs = 0
        all_centers = []
        all_contexts = []
        for anchors in all_anchors:
            batch = anchors.shape[0]

            candidates = torch.randint(0, n, (batch, k_sample), device=self.device)
            k_random = k_sample

            # Neighbor-of-neighbor sampling
            if self.knn_k > 0 and self.knn_nofn:
                nofn = self._sample_nofn(anchors)
                candidates = torch.cat([candidates, nofn], dim=1)

            if matmul_corr:
                if predictive_shift > 0 and use_deriv_corr:
                    anchor_normed = sig_normed_anchor[anchors]
                else:
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

            # Exclude self-matches (anchor correlated with itself)
            self_match = candidates == anchors.unsqueeze(1)
            mask[self_match] = False

            good_counts = mask.sum(dim=1)

            if max_hit_ratio is not None:
                random_hits = mask[:, :k_random].sum(dim=1)
                max_hits = int(k_random * max_hit_ratio)
                too_popular = random_hits > max_hits
                good_counts[too_popular] = 0
                mask[too_popular] = False

            self._update_knn(anchors, candidates, mask)

            max_good = int(good_counts.max().item())
            if max_good == 0:
                continue

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
        """Periodic L2 normalization of W and C to prevent magnitude blow-up."""
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

    def _sample_nofn(self, anchors):
        """Generate neighbor-of-neighbor candidates for each anchor."""
        batch = anchors.shape[0]
        K = self.knn_k
        anchor_nbs = self.knn_lists[anchors]
        nofn = self.knn_lists[anchor_nbs].reshape(batch, K * K)
        is_self = (nofn == anchors.unsqueeze(1))
        is_nb = (nofn.unsqueeze(2) == anchor_nbs.unsqueeze(1)).any(dim=2)
        exclude = is_self | is_nb
        replacements = torch.randint(0, self.n, nofn.shape, device=self.device)
        nofn = torch.where(exclude, replacements, nofn)
        return nofn

    def _update_knn(self, anchors, candidates, mask):
        """Update per-neuron KNN lists with correlation-passing candidates."""
        if self.knn_k <= 0:
            return

        anchor_w = self.positions[anchors]
        cand_w = self.positions[candidates]
        a_norm = anchor_w / anchor_w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        c_norm = cand_w / cand_w.norm(dim=2, keepdim=True).clamp(min=1e-8)
        emb_sim = (a_norm.unsqueeze(1) * c_norm).sum(dim=2)
        emb_sim[~mask] = -float('inf')

        cur_ids = self.knn_lists[anchors]
        cur_dists = self.knn_dists[anchors]

        if KNN_STABLE_INSERT:
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
            all_ids = torch.cat([cur_ids, candidates], dim=1)
            all_dists = torch.cat([cur_dists, emb_sim], dim=1)
            all_dists[all_ids == anchors.unsqueeze(1)] = -float('inf')
            topk_dists, topk_idx = all_dists.topk(self.knn_k, dim=1)
            topk_ids = torch.gather(all_ids, 1, topk_idx)
            self.knn_lists[anchors] = topk_ids
            self.knn_dists[anchors] = topk_dists

    # ---- Data transfer ----

    def get_positions(self):
        """Return positions as numpy array."""
        return self.positions.detach().cpu().numpy()

    def get_contexts(self):
        """Return context vectors as numpy array."""
        if self.contexts is not None:
            return self.contexts.detach().cpu().numpy()
        return None

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

    def _refresh_knn_dists(self):
        """Recompute cosine similarities for all KNN entries."""
        if self.knn_k <= 0:
            return
        w = self.positions / self.positions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        nb_w = w[self.knn_lists]
        self.knn_dists = (w.unsqueeze(1) * nb_w).sum(dim=2)

    def knn_stability(self):
        """Compute overlap between current and previous KNN snapshot."""
        if self.knn_k <= 0:
            return 1.0, 0, 0.0, 0.0
        matches = (self._knn_prev.unsqueeze(2) == self.knn_lists.unsqueeze(1))
        per_neuron = matches.any(dim=2).sum(dim=1).float()
        overlap = float(per_neuron.mean().item()) / self.knn_k
        n_changed = int((per_neuron < self.knn_k).sum().item())
        swaps = self.knn_k - per_neuron
        sorted_swaps = swaps.sort().values
        n = self.n
        top50_swaps = sorted_swaps[:n // 2].mean().item()
        top90_swaps = sorted_swaps[:n * 9 // 10].mean().item()
        self._knn_prev = self.knn_lists.clone()
        return overlap, n_changed, top50_swaps, top90_swaps

    def knn_spatial_accuracy(self, width, radius=3, channels=1, n_eval=None):
        """Fraction of KNN neighbors within radius pixels on the grid."""
        if self.knn_k <= 0:
            return 0.0
        ne = n_eval if n_eval is not None else self.n
        hw = ne // channels
        pixel_ids = torch.arange(self.n, device=self.device) % hw
        gy = pixel_ids // width
        gx = pixel_ids % width
        ch = torch.arange(self.n, device=self.device) // hw
        nx, ny, nc = gx[:ne], gy[:ne], ch[:ne]
        knn_flat = self.knn_lists[:ne]
        kx, ky, kc = gx[knn_flat], gy[knn_flat], ch[knn_flat]
        dx = (nx.unsqueeze(1) - kx).float()
        dy = (ny.unsqueeze(1) - ky).float()
        dist = (dx * dx + dy * dy).sqrt()
        same_ch = (nc.unsqueeze(1) == kc)
        within = (dist <= radius) & same_ch
        return float(within.float().mean().item())

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
