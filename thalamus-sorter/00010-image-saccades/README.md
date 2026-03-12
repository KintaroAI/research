# ts-00010: Image Saccades — Real Image Signals via Random Crops

**Date:** 2026-03-11
**Status:** In Progress
**Source:** *tagged on completion as `exp/ts-00010`*

## Goal

Replace synthetic Gaussian-smoothed noise with real image content as the temporal signal source. Each timestep, a random crop (simulated saccade) from a larger source image provides firing rates for all neurons. Nearby neurons see similar content across crops → temporal correlation → spatial map discovery.

## Motivation

ts-00009 proved the correlation-based pipeline works with synthetic signals. But those signals have artificial correlation structure (Gaussian blur). Real images have natural spatial correlation — nearby pixels tend to be similar because of object continuity, textures, and lighting. The question: can the sorter discover spatial structure from natural image statistics alone?

## Approach

### Signal generation

- Source: a large grayscale image (pre-normalized to 0-1 firing rates)
- Grid: 80x80 neurons, each maps to one pixel
- Each timestep t: pick random offset (dx, dy), sample 80x80 crop from source
- Firing rate of neuron (x, y) at time t = source_image[y + dy, x + dx]
- Nearby neurons see similar pixel values across random crops → correlated temporal patterns

### Key difference from ts-00009

In ts-00009, correlation came from Gaussian blur applied to random noise — the spatial smoothing was explicit and controlled by sigma. Here, correlation comes from natural image structure — edges, textures, gradients. The smoothing is implicit in the image content.

## Files

- `main.py` — `correlation` mode with image-based signal generation
