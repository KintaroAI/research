# Task 00025: A Longer Lifetime for the Universal Model (experiment vm-00021)

**Date:** 2026-10-25
**Status:** In Progress

## Context

- vm-00019/20: the universal model holds the scene with nothing detached, but learns what a click does slowly and
  unevenly: at 60k one seed in three predicts food and poison right. vm-00018's patch model needed 100k for all three
  click outcomes.
- vm-00020: where the prediction is right, the instinct uses it (one seed: food clicks 26.8 → 31.8 and poison clicks
  38.9 → 19.7 per 1,000 steps, against its policy alone). The seed spread is large.

## Experiment vm-00021

The universal model's default (`configs/U_universal.json`: gray, checkerboard poison, 8 dots, the instinct with 16
candidates), **200k ticks**, **6 seeds**, three per GPU. No code changes.

## Questions

1. Does every seed learn the three click outcomes given time (food −0.30, poison +0.50, nothing 0)?
2. Does the instinct then reduce poison clicks in every seed (frozen tests with and without it)?
3. Does the policy learn the instinct's choices (the policy alone improving over the lifetime)?
4. Does exploration hold over a long lifetime (online food and poison clicks, hunger at the cap)?

## Measures

The click test at 25k, 50k, 100k, 150k, 200k; the frozen tests with the instinct at the same checkpoints and without it
at 100k and 200k; token probes at 200k; online food and poison clicks, the instinct's override share, cross-entropy per
predicted kind.
