# 00004 — 3D Movement Direction Categorization

**Status:** Complete
**Source:** `exp/00004` (`12981aa`)

## Goal

Test whether the cell can learn to categorize the direction of a moving object in 3D
space from position traces alone. This is a practical demonstration of temporal learning:
an instantaneous position snapshot reveals nothing about direction of movement.

## Hypothesis

The correlation mode will categorize 3D movement directions (NMI > 0.6) across increasing
difficulty levels, while instantaneous mode gets chance-level performance.

## Method

**Data generation:** Objects start at random positions (std=10) and execute a random walk
with directional drift. Each trace is a `(3, T)` matrix of x,y,z positions.
- `position(t) = start + cumsum(direction * speed + noise)`
- The random start position hides direction from instantaneous/mean readout
- The covariance structure encodes which axes co-vary → reveals direction

**Three difficulty levels:**
1. **6 cardinal directions** (±x, ±y, ±z) — axis-aligned, maximally separated
2. **8 diagonal directions** (all ±x±y±z) — off-axis, 3 axes co-vary per direction
3. **12 icosahedron directions** — fine-grained, closely spaced on the sphere

**Control:** Same data, instantaneous mode (mean of trace) — should fail because
random starting positions dominate the mean.

**Parameters:** T=10, speed=1.0, noise=0.3, 5k frames (10k for 12 directions).

**Commands:**
```bash
cd dev
python benchmark_3d.py -o $(python output_name.py 4 movement_3d)
```

## Results

```
metric                      cardinal      diagonal         icosa       control
------------------------------------------------------------------------------
n_directions                       6             8            12             6
winner_entropy                 0.747         0.717         0.867         1.000
usage_gini                     0.453         0.491         0.421         0.021
confidence_gap                 0.759         0.892         0.711         0.422
prototype_spread               1.012         1.082         1.035         1.198
nmi                            0.702         0.776         0.770         0.044
purity                         0.540         0.593         0.500         0.264
consistency                    0.865         0.967         0.813         0.266
lock_in_nmi                    0.702         0.767         0.771         0.052
lock_in_frame                   1000          1000          2000          2000
```

## Analysis

**Direction learning works.** The cell categorizes 3D movement directions from position
traces with NMI 0.70–0.78 across all difficulty levels, while instantaneous mode gets
NMI=0.04 (chance). The random starting position makes direction invisible to a single
snapshot — only temporal co-variation reveals it.

**Diagonal directions are easiest.** NMI=0.776, consistency=0.967. All 3 axes co-vary
for each diagonal direction, providing a richer covariance signal than axis-aligned
movement where only 1 axis varies.

**12 directions still separable.** NMI=0.770 with 12 closely-spaced directions on the
sphere, using only 3 input channels. The cell handles high category counts relative
to input dimensionality.

**Coverage trade-off persists.** Gini 0.42–0.49, entropy 0.72–0.87. Some output units
are underused. This is consistent with the general SQM benchmark (exp 00003) — the
power iteration update can bias prototypes.

**Only 3 inputs.** This works with just 3 input channels (x,y,z). Most prior experiments
used 16 inputs. The cell is effective even in very low dimensions.
