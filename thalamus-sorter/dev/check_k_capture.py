"""Check how many true grid neighbors land in random k_sample candidates
at different grid sizes and k_sample values."""

import numpy as np

for grid_w, k_sample in [(80, 200), (160, 200), (160, 800), (160, 1600)]:
    n = grid_w * grid_w
    gx = np.arange(n) % grid_w
    gy = np.arange(n) // grid_w

    num_trials = 1000
    captures_3 = []
    captures_5 = []
    captures_10 = []
    min_dists = []
    mean_near_dists = []

    for _ in range(num_trials):
        anchor = np.random.randint(0, n)
        cands = np.random.randint(0, n, k_sample)

        ax, ay = gx[anchor], gy[anchor]
        dists = np.abs(gx[cands] - ax) + np.abs(gy[cands] - ay)

        captures_3.append((dists <= 3).sum())
        captures_5.append((dists <= 5).sum())
        captures_10.append((dists <= 10).sum())
        if len(dists) > 0 and dists.min() > 0:
            min_dists.append(dists[dists > 0].min())
        near = dists[dists <= 5]
        if len(near) > 0:
            mean_near_dists.append(near.mean())

    c3 = np.array(captures_3)
    c5 = np.array(captures_5)
    c10 = np.array(captures_10)
    md = np.array(min_dists) if min_dists else np.array([0])
    mnd = np.array(mean_near_dists) if mean_near_dists else np.array([0])

    # Count true neighbors at each radius
    # For a center pixel not near edges: <=3 has ~24 neighbors, <=5 has ~60, <=10 has ~220
    sample_ax, sample_ay = grid_w // 2, grid_w // 2
    true_3 = sum(1 for i in range(n) if abs(gx[i]-sample_ax) + abs(gy[i]-sample_ay) <= 3 and i != sample_ax + sample_ay*grid_w)
    true_5 = sum(1 for i in range(n) if abs(gx[i]-sample_ax) + abs(gy[i]-sample_ay) <= 5 and i != sample_ax + sample_ay*grid_w)
    true_10 = sum(1 for i in range(n) if abs(gx[i]-sample_ax) + abs(gy[i]-sample_ay) <= 10 and i != sample_ax + sample_ay*grid_w)

    print(f"\n=== {grid_w}x{grid_w} (n={n:,}), k_sample={k_sample} ({k_sample/n*100:.2f}%) ===")
    print(f"  True neighbors: <=3px={true_3}, <=5px={true_5}, <=10px={true_10}")
    print(f"  Captured per anchor (mean ± std):")
    print(f"    <=3px:  {c3.mean():.2f} ± {c3.std():.2f}  (of {true_3}, capture rate {c3.mean()/true_3*100:.1f}%)")
    print(f"    <=5px:  {c5.mean():.2f} ± {c5.std():.2f}  (of {true_5}, capture rate {c5.mean()/true_5*100:.1f}%)")
    print(f"    <=10px: {c10.mean():.2f} ± {c10.std():.2f}  (of {true_10}, capture rate {c10.mean()/true_10*100:.1f}%)")
    print(f"  Zero <=5px captures: {(c5 == 0).sum()}/{num_trials} ({(c5==0).mean()*100:.1f}%)")
    print(f"  Closest candidate: mean={md.mean():.2f}, median={np.median(md):.1f}")
    print(f"  Mean dist of <=5px captures: {mnd.mean():.2f}")
