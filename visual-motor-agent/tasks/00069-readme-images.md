# Task 00069: Images in Every Experiment README

**Date:** 2026-10-31
**Status:** In progress

## Context

No visual-motor README has an image. The data folder holds 199 PNGs, 22 GIFs and 50 videos across 18 experiments, and every experiment keeps its metrics. The owner's decision (2026-10-31): an `images/` folder in each experiment with small PNGs. This is an exception to two rules:
experiment folders may now hold `images/`, and these PNGs are the only binaries allowed. Videos and checkpoints stay out of git.

## Design

`dev/readme_images.py`, repeatable, CPU only but for the car's EGL frames. For each experiment:

- **`images/curves.png`:** every run's `metrics.jsonl`, the mean of its seeds by arm: food and poison per 1k ticks for foraging arms, patch cross-entropy for
  text arms, success for the early PPO runs, validation error for the PyTorch runs. At most 8 arms, the best by the last value; the caption says so.
- **`images/world.png`:** a strip of the run's own world, from its `info.json`, under random held moves. It shows the world, not the agent, and says so.
- **`images/sheet.png`:** an existing picture where there is one (a dream or forage sheet, an imagination strip), or stills from a video.
- Each PNG at most 960 px wide and 200 KB, reduced to a palette. A README gets a `## Figures` section between `<!-- figures -->` markers, before `## Analysis`;
  a rerun replaces it.
- Charts follow the dataviz rules: fixed categorical order, a legend for 2 or more series, direct labels for up to 4, 2 px lines, no second axis (panels instead).

## Verification

Every PNG under the limits; each README's links resolve; a rerun changes nothing but what changed; every image opened and checked by eye, in montages. Worlds that
the current code can no longer build are reported, not faked.

## Results

130 PNGs across 58 experiments, 2.6 MB in all, every one under the limits and checked by eye in montages: 58 charts, 55 world strips (none for 00029, 00033 and 00038, which
have no world config), 17 sheets. The check found the car's flipping bug (task 00068) in vm-00059's stills. vm-00060's images follow its runs.
