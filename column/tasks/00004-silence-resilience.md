# 00004 — Silence & Noise Resilience

**Date:** 2026-03-15
**Status:** Planned

## Context

Testing revealed that the SoftWTACell behaves differently under two "no signal" scenarios:

- **Zero input:** prototypes survive with zero drift (happy accident — normalization saves us).
  Output is uniform [0.25, 0.25, 0.25, 0.25] which is reasonable.
- **White noise:** destructive. Prototypes drift ~1.1-1.4 on the unit sphere after 10k frames.
  NMI drops from 1.0 to 0.33. Learned categories are destroyed.

The cell has no mechanism to distinguish meaningful input from noise.

## Problem

Noise corrupts learned prototypes because:
1. Random vectors sometimes match a prototype well enough to trigger Hebbian pull
2. Even when match is poor, recruitment overwrites dormant units with noise directions
3. There's no "don't learn" signal — every input triggers an update

## Proposed Solutions

### Option A: Match-gated learning
Only update if match quality exceeds a minimum floor. Currently the match threshold
redirects to dormant unit; instead, skip the update entirely for very poor matches.
- Pro: simple, no new hyperparameters (reuse match_threshold)
- Con: needs a second, lower threshold ("redirect" vs "ignore")

### Option B: Activity-gated learning
If input norm is below a threshold, skip the update. Handles the "no signal" case
explicitly but doesn't address noise.

### Option C: Learning rate proportional to match quality
Scale effective_lr by match_quality. Good matches → full learning. Poor matches → tiny
updates. Noise averages out over time.
- Pro: smooth, no hard cutoff
- Con: persistent noise still accumulates drift, just slower

### Recommended: A + B combined
- Skip update entirely if input norm < epsilon (silence)
- Skip update if match quality < low_threshold (noise/garbage)
- Redirect to dormant unit if low_threshold < match_quality < match_threshold (novel input)
- Normal Hebbian pull if match_quality > match_threshold (familiar input)

## Deliverables

1. Add silence/noise gating to `SoftWTACell.update()`
2. Test: 5k frames of noise after learning → NMI stays above 0.8
3. Test: zero input → uniform low output, no prototype drift
4. SQM comparison: before vs after the change on baseline synthetic data
