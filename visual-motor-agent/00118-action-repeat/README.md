# vm-00118: Action Repeat — a Move Decided Every 4 Ticks and Held

**Date:** 2026-11-25
**Status:** Running (queued 2026-11-25 on the RTX 4090 behind vm-00117's chain)
**Source:** `dev/` (in progress)
**Outputs:** `~/data/research/visual-motor-agent/exp_00118/`
**Task plan:** [`../tasks/00131-action-repeat.md`](../tasks/00131-action-repeat.md)

## Goal

vm-00116 found that in Breakout the default agent's label knows where the paddle should be but orders two candidate moves correctly only about six times in ten, and that its misses are a paddle that travels enough but not to the ball. A move held for 4 ticks shifts the paddle up to 18 px, so what a decision does to the label is four times larger to learn, and a held move cannot dither. Does deciding every 4 ticks and holding the move make the agent play better, and does the label then order the moves?

## Method

- **The change** (`online.act_every = 4`, the owner's idea of 2026-11-17): a move is decided on ticks 1, 5, 9, ... as now and made again on the 3 ticks between. The model still sees, predicts and learns every tick; the move token's target is the move made.
- **BA:** the default agent in Breakout stage 1 (`configs/U_breakout.json`) with that one change, seeds 1–12, 60k ticks, against BD (vm-00103), paired by seed; all on the RTX 4090.
- **Then, learning off** at 60k (6,000 ticks, new courts), and vm-00116's miss autopsy on BA's weights.

The readings, set before the results, are in the task plan.

## Results

Not in yet.
