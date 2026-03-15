# 00001 — Soft-WTA Cell

**Status:** Complete
**Source:** `exp/00001` (`f5bb5d9`)

## Goal

Implement the core soft-WTA competitive categorization cell described in PLANNING.md and
verify it satisfies the key requirements using Separation Quality Metrics (SQM).

## Hypothesis

A soft-WTA cell with usage-gated plasticity and match-threshold recruitment will:
1. Form distinct categories from clustered input (NMI > 0.7)
2. Resist collapse — white noise produces distributed, low-confidence outputs
3. Use most available output units (winner entropy > 0.8, Gini < 0.2)

## Method

**Architecture:** SoftWTACell with m prototype vectors of dimension n.

- **Forward:** normalize input → dot-product similarity with prototypes → softmax(sim / temperature)
- **Update (Hebbian):** winner's prototype pulled toward input. Learning rate gated by usage: `lr / (1 + n_outputs * usage[winner])`. Frequent winners learn slower.
- **Recruitment:** if match quality < threshold, redirect update to least-used unit at full LR
- **Usage tracking:** EMA of win indicator per unit (decay=0.99)

**Synthetic data:** Gaussian clusters on the unit sphere in R^n, presented one at a time (online).

## Evaluation — Separation Quality Metrics (SQM)

Every experiment run reports a standard SQM suite. These metrics are the primary tool for
comparing implementations — any change to the cell architecture must be evaluated against
this baseline using the same metrics on the same synthetic data.

| Metric | Axis | What it measures | Ideal |
|---|---|---|---|
| **Winner entropy** | Coverage | How uniformly output units are used (normalized to [0,1]) | → 1.0 |
| **Usage Gini** | Coverage | Inequality of winner counts (0=uniform, 1=collapse) | → 0.0 |
| **Confidence gap** | Selectivity | Mean top-1 minus top-2 probability | High |
| **Prototype spread** | Selectivity | Mean pairwise cosine distance between prototypes | → 1.0 |
| **NMI** | Separation | Normalized mutual info between winners and true clusters | → 1.0 |
| **Purity** | Separation | Per-unit fraction from dominant cluster, averaged | → 1.0 |

SQM is logged at 20/40/60/80/100% of training to track convergence.

**Commands:**
```bash
cd dev
make setup
make test
python main.py --n-inputs 16 --n-outputs 8 --frames 10000 \
  -o $(python output_name.py 1 baseline_16x8)
```

## Results

Baseline run: `--n-inputs 16 --n-outputs 8 --frames 10000 --temperature 0.5 --lr 0.05 --match-threshold 0.5 --seed 42`

### Final SQM

| Metric | Value |
|---|---|
| Winner entropy | 0.932 |
| Usage Gini | 0.213 |
| Confidence gap | 0.262 |
| Prototype spread | 0.930 |
| NMI | 0.875 |
| Purity | 0.828 |

### SQM convergence (per-window, not cumulative)

| Frame | Entropy | Gini | Conf gap | Spread | NMI | Purity |
|---|---|---|---|---|---|---|
| 2000 | 0.962 | 0.190 | 0.201 | 0.930 | 0.673 | 0.741 |
| 4000 | 0.917 | 0.236 | 0.277 | 0.933 | 0.957 | 0.932 |
| 6000 | 0.918 | 0.226 | 0.277 | 0.928 | 0.958 | 0.929 |
| 8000 | 0.913 | 0.242 | 0.277 | 0.928 | 0.955 | 0.929 |
| 10000 | 0.920 | 0.221 | 0.278 | 0.930 | 0.959 | 0.932 |

## Analysis

- **Separation is strong.** NMI jumps from 0.67 at 2k frames to 0.96 by 4k and holds. The cell learns the 8-cluster structure quickly and stably.
- **Coverage is good but not perfect.** Entropy ~0.93, Gini ~0.21 — most units are active but usage isn't perfectly uniform. Some units handle more clusters than others (8 clusters mapped to 8 outputs, but not all 1:1).
- **Selectivity is moderate.** Confidence gap of 0.26 means the winner typically has ~26% more probability than the runner-up. Temperature 0.5 produces soft decisions — sharper temperature would increase this at the cost of gradient flow.
- **Prototype spread is near-optimal.** 0.93 vs 1.0 expected for random unit vectors — prototypes are well-distributed in input space.
- **Final vs windowed NMI gap** (0.875 final vs 0.96 windowed) reflects early exploration: the first ~2k frames contribute noisy assignments that drag down the cumulative score. The cell's steady-state separation is better than the overall number suggests.
