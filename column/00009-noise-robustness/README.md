# 00009 — Noise Robustness

**Status:** Complete
**Source:** `exp/00009` (`2a5cc93`)

## Goal

Measure how random noise on a subset of input channels degrades separation quality
when other channels carry meaningful signal.

## Method

16 inputs, 4 clusters. First N channels carry cluster signal (noise=0.1), remaining
channels are replaced with pure random noise.

Two sweeps:
1. **Noisy channel fraction** (0–88%) at fixed noise amplitude=1.0
2. **Noise amplitude** (0–5.0) at fixed 50% noisy channels

## Results

### Sweep 1: Noisy channel fraction

| Noisy channels | Fraction | NMI | Purity | Consistency |
|---|---|---|---|---|
| 0 / 16 | 0% | 1.000 | 1.000 | 1.000 |
| 2 / 16 | 12% | 0.135 | 0.435 | 0.423 |
| 4 / 16 | 25% | 0.013 | 0.304 | 0.303 |
| 6+ / 16 | 38%+ | ~0.001 | ~0.26 | ~0.26 |

### Sweep 2: Noise amplitude (8 of 16 channels noisy)

| Amplitude | NMI | Purity | Consistency |
|---|---|---|---|
| 0.0 | 0.852 | 0.837 | 0.999 |
| 0.1 | 0.852 | 0.837 | 0.999 |
| 0.3 | 0.994 | 0.999 | 0.999 |
| 0.5 | 0.112 | 0.426 | 0.440 |
| 1.0+ | ~0.001 | ~0.26 | ~0.26 |

## Analysis

**The cell is fragile to noisy channels.** Just 2 noisy channels (12%) at amplitude 1.0
drop NMI from 1.000 to 0.135. By 4 noisy channels (25%), separation is gone.

**Root cause: normalization.** The cell normalizes inputs to unit vectors before computing
cosine similarity. High-amplitude noise in a few channels dominates the direction of the
normalized vector, drowning out the signal in the other channels. Effectively, the noisy
channels hijack the input direction.

**Low-amplitude noise is tolerable.** At amplitude 0.3 with 50% noisy channels, NMI is
actually 0.994 — better than clean! The small noise acts as regularization. But at
amplitude 0.5, it collapses. The cliff between "helpful noise" and "destructive noise"
is steep.

**The signal-to-noise ratio matters, not just the fraction.** With signal noise=0.1 and
cluster centers on the unit sphere, the signal magnitude per channel is ~1/sqrt(16)≈0.25.
Channel noise at amplitude 1.0 is 4x the signal — far too large. At 0.3, it's comparable
and the cell still separates.

**Implications for real-world use:**
- All input channels should carry meaningful signal, or noisy channels should be masked
- Alternatively, the cell could benefit from input-channel attention or feature selection
  that down-weights noisy channels before the prototype match
- The normalization step is both a strength (scale invariance) and a weakness (noise
  amplification) — a fundamental trade-off

## Commands

```bash
cd dev
python benchmark_noise.py -o $(python output_name.py 9 noise_robustness)
```
