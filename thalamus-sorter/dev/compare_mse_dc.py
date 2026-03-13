"""Compare MSE vs derivative-correlation neighbor discovery on the same walk."""

import numpy as np
import torch

source = np.load("saccades_gray.npy")
src_h, src_w = source.shape
w = h = 80
n = w * h
T = 1000
step = 50
k_sample = 200
mse_thresh = 0.02
dc_thresh = 0.5

# Build signal buffer with shared random walk
signals_np = np.zeros((n, T), dtype=np.float32)
walk_dy = np.random.randint(0, src_h - h + 1)
walk_dx = np.random.randint(0, src_w - w + 1)
for t in range(T):
    walk_dy = np.clip(walk_dy + np.random.randint(-step, step + 1), 0, src_h - h)
    walk_dx = np.clip(walk_dx + np.random.randint(-step, step + 1), 0, src_w - w)
    crop = source[walk_dy:walk_dy+h, walk_dx:walk_dx+w].ravel()
    signals_np[:, t] = crop  # raw, no mean sub

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
signals = torch.from_numpy(signals_np).to(device)

# Run multiple ticks with same anchors/candidates
num_ticks = 50
batch = 256

total_mse_only = 0
total_dc_only = 0
total_both = 0
total_neither = 0
total_pairs = 0

mse_scores_near = []  # MSE scores for pairs where grid dist <= 3
mse_scores_far = []
dc_scores_near = []
dc_scores_far = []

gx = np.arange(n) % w
gy = np.arange(n) // w

for tick in range(num_ticks):
    anchors = torch.randint(0, n, (batch,), device=device)
    candidates = torch.randint(0, n, (batch, k_sample), device=device)

    anchor_sig = signals[anchors]       # (batch, T)
    cand_sig = signals[candidates]      # (batch, k_sample, T)

    # MSE
    diff = anchor_sig.unsqueeze(1) - cand_sig
    mse = (diff * diff).mean(dim=2)
    mse_mask = mse < mse_thresh

    # Derivative correlation
    anchor_d = anchor_sig[:, 1:] - anchor_sig[:, :-1]
    cand_d = cand_sig[:, :, 1:] - cand_sig[:, :, :-1]
    anchor_dc = anchor_d - anchor_d.mean(dim=1, keepdim=True)
    cand_dc = cand_d - cand_d.mean(dim=2, keepdim=True)
    anchor_norm = anchor_dc.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cand_norm = cand_dc.norm(dim=2, keepdim=True).clamp(min=1e-8)
    dc_score = (anchor_dc.unsqueeze(1) / anchor_norm.unsqueeze(1) *
                cand_dc / cand_norm).sum(dim=2)
    dc_mask = dc_score > dc_thresh

    # Count overlaps
    both = (mse_mask & dc_mask).sum().item()
    mse_only = (mse_mask & ~dc_mask).sum().item()
    dc_only = (~mse_mask & dc_mask).sum().item()
    neither = (~mse_mask & ~dc_mask).sum().item()

    total_both += both
    total_mse_only += mse_only
    total_dc_only += dc_only
    total_neither += neither
    total_pairs += batch * k_sample

    # Collect scores by grid distance (sample first tick only for speed)
    if tick == 0:
        anch_np = anchors.cpu().numpy()
        cand_np = candidates.cpu().numpy()
        mse_np = mse.cpu().numpy()
        dc_np = dc_score.cpu().numpy()
        for i in range(batch):
            ax, ay = gx[anch_np[i]], gy[anch_np[i]]
            for j in range(k_sample):
                cx, cy = gx[cand_np[i, j]], gy[cand_np[i, j]]
                dist = abs(ax - cx) + abs(ay - cy)
                if dist <= 3:
                    mse_scores_near.append(mse_np[i, j])
                    dc_scores_near.append(dc_np[i, j])
                elif dist >= 30:
                    mse_scores_far.append(mse_np[i, j])
                    dc_scores_far.append(dc_np[i, j])

print("=== Neighbor overlap (same anchors, same candidates) ===")
print(f"Total pairs checked: {total_pairs:,}")
print(f"  Both agree (neighbor):     {total_both:>10,}  ({total_both/total_pairs*100:.2f}%)")
print(f"  MSE only:                  {total_mse_only:>10,}  ({total_mse_only/total_pairs*100:.2f}%)")
print(f"  Deriv-corr only:           {total_dc_only:>10,}  ({total_dc_only/total_pairs*100:.2f}%)")
print(f"  Neither (not neighbor):    {total_neither:>10,}  ({total_neither/total_pairs*100:.2f}%)")

mse_total = total_both + total_mse_only
dc_total = total_both + total_dc_only
if mse_total > 0 and dc_total > 0:
    overlap_pct_mse = total_both / mse_total * 100
    overlap_pct_dc = total_both / dc_total * 100
    print(f"\n  Of MSE neighbors, {overlap_pct_mse:.1f}% also found by deriv-corr")
    print(f"  Of deriv-corr neighbors, {overlap_pct_dc:.1f}% also found by MSE")

print(f"\n=== Score distributions (tick 0) ===")
if mse_scores_near:
    mn = np.array(mse_scores_near)
    mf = np.array(mse_scores_far)
    dn = np.array(dc_scores_near)
    df = np.array(dc_scores_far)
    print(f"  Near pairs (dist<=3):  n={len(mn)}")
    print(f"    MSE:       mean={mn.mean():.5f}  median={np.median(mn):.5f}  std={mn.std():.5f}")
    print(f"    deriv-corr: mean={dn.mean():.4f}  median={np.median(dn):.4f}  std={dn.std():.4f}")
    print(f"  Far pairs (dist>=30):  n={len(mf)}")
    print(f"    MSE:       mean={mf.mean():.5f}  median={np.median(mf):.5f}  std={mf.std():.5f}")
    print(f"    deriv-corr: mean={df.mean():.4f}  median={np.median(df):.4f}  std={df.std():.4f}")
    print(f"\n  Discrimination ratio:")
    print(f"    MSE:        far/near = {mf.mean()/mn.mean():.1f}x")
    if abs(df.mean()) > 1e-6:
        print(f"    deriv-corr: near/far = {dn.mean()/df.mean():.1f}x")
    else:
        print(f"    deriv-corr: near={dn.mean():.4f}, far≈0")

    # What does MSE catch that DC misses, and vice versa?
    print(f"\n=== MSE-only vs DC-only neighbor profiles (tick 0) ===")
    mse_m = mse < mse_thresh
    dc_m = dc_score > dc_thresh
    mse_only_m = mse_m & ~dc_m
    dc_only_m = ~mse_m & dc_m

    anch_np = anchors.cpu().numpy()
    cand_np = candidates.cpu().numpy()
    mse_only_dists = []
    dc_only_dists = []
    both_dists = []
    both_m = mse_m & dc_m

    for i in range(batch):
        ax, ay = gx[anch_np[i]], gy[anch_np[i]]
        for j in range(k_sample):
            cx, cy = gx[cand_np[i, j]], gy[cand_np[i, j]]
            dist = abs(ax - cx) + abs(ay - cy)
            if mse_only_m[i, j]:
                mse_only_dists.append(dist)
            if dc_only_m[i, j]:
                dc_only_dists.append(dist)
            if both_m[i, j]:
                both_dists.append(dist)

    if both_dists:
        bd = np.array(both_dists)
        print(f"  Both agree:    n={len(bd):>6}  mean_dist={bd.mean():.2f}  <5px={100*(bd<=5).mean():.1f}%  <3px={100*(bd<=3).mean():.1f}%")
    if mse_only_dists:
        md = np.array(mse_only_dists)
        print(f"  MSE only:      n={len(md):>6}  mean_dist={md.mean():.2f}  <5px={100*(md<=5).mean():.1f}%  <3px={100*(md<=3).mean():.1f}%")
    if dc_only_dists:
        dd = np.array(dc_only_dists)
        print(f"  DC only:       n={len(dd):>6}  mean_dist={dd.mean():.2f}  <5px={100*(dd<=5).mean():.1f}%  <3px={100*(dd<=3).mean():.1f}%")

# === Quality vs true grid neighbors ===
print(f"\n=== Neighbor quality vs ground truth grid ===")
# Precompute true grid distances for all pairs (using Manhattan)
# For each anchor, find its true K nearest grid neighbors
from scipy.spatial import cKDTree

grid_coords = np.column_stack([gx, gy]).astype(np.float64)
grid_tree = cKDTree(grid_coords)

# For each anchor in tick 0, compare method selections vs true nearest
anchors_np = anchors.cpu().numpy()
cands_np = candidates.cpu().numpy()
mse_np = mse.cpu().numpy()
dc_np = dc_score.cpu().numpy()

for K_true in [5, 10, 20]:
    # True K nearest for each anchor
    _, true_knn = grid_tree.query(grid_coords[anchors_np], k=K_true + 1)
    true_knn = true_knn[:, 1:]  # remove self

    mse_precision = []
    dc_precision = []
    both_precision = []

    mse_m = (mse < mse_thresh).cpu().numpy()
    dc_m = (dc_score > dc_thresh).cpu().numpy()

    for i in range(batch):
        true_set = set(true_knn[i])

        mse_neighbors = set(cands_np[i, mse_m[i]])
        dc_neighbors = set(cands_np[i, dc_m[i]])
        both_neighbors = mse_neighbors & dc_neighbors

        # Precision: what fraction of selected neighbors are true K-nearest?
        if len(mse_neighbors) > 0:
            mse_precision.append(len(mse_neighbors & true_set) / len(mse_neighbors))
        if len(dc_neighbors) > 0:
            dc_precision.append(len(dc_neighbors & true_set) / len(dc_neighbors))
        if len(both_neighbors) > 0:
            both_precision.append(len(both_neighbors & true_set) / len(both_neighbors))

    mp = np.array(mse_precision) if mse_precision else np.array([0])
    dp = np.array(dc_precision) if dc_precision else np.array([0])
    bp = np.array(both_precision) if both_precision else np.array([0])
    print(f"  True K={K_true} nearest:")
    print(f"    MSE precision:        {mp.mean()*100:.1f}%  (of selected, how many are true top-{K_true})")
    print(f"    Deriv-corr precision: {dp.mean()*100:.1f}%")
    print(f"    Both-agree precision: {bp.mean()*100:.1f}%")

# Also: recall — of the true K nearest that were in the candidate pool,
# how many did each method find?
print(f"\n  Recall (of true K=10 in candidate pool, how many found?):")
_, true_k10 = grid_tree.query(grid_coords[anchors_np], k=11)
true_k10 = true_k10[:, 1:]

mse_recall = []
dc_recall = []
for i in range(batch):
    true_set = set(true_k10[i])
    cand_set = set(cands_np[i])
    available = true_set & cand_set  # true neighbors that were candidates
    if len(available) == 0:
        continue
    mse_found = set(cands_np[i, mse_m[i]]) & available
    dc_found = set(cands_np[i, dc_m[i]]) & available
    mse_recall.append(len(mse_found) / len(available))
    dc_recall.append(len(dc_found) / len(available))

mr = np.array(mse_recall) if mse_recall else np.array([0])
dr = np.array(dc_recall) if dc_recall else np.array([0])
print(f"    MSE recall:        {mr.mean()*100:.1f}%")
print(f"    Deriv-corr recall: {dr.mean()*100:.1f}%")
