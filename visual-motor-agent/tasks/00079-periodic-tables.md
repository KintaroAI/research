# Task 00079: Tables Averaged Now and Then

**Date:** 2026-11-01
**Status:** Planned, not run (the owner, 2026-11-01: an idea for the future; first the 500k lives of task 00080). `online.tie_every` and its tests are in.

## Context

- vm-00067: one table for all 16 pieces keeps most of their food, with less poison, better prediction and half the distinct parameters. But a shared table is global: it is averaged after every update, which works against the separate transformers the owner wants and is communication that would cost dearly across many GPUs.
- The idea, chosen by the owner: pieces keep their own tables and average them only now and then, as brain areas might share a common code without being wired to one store. It keeps the communication near zero. Whether it keeps the common language is the question.

## Design

`online.tie_every` (default 1, every update, as vm-00067): at least this many ticks between averages. No change to the GPU library. When it is above 1, `tie_gap` is logged at each average: the pieces' spread around the mean table, relative to the mean's size.

**The risk:** two tables that have moved apart may not average into anything useful. Rows are not aligned by anything but their shared start, so a large gap would blur them and the jump could set the pieces back. `tie_gap` and the food around each average show this.

## Experiment

- **T1K:** `configs/U_default.json`, `model.band_n=16`, `gain`, `online.tie_tables=inout`, `online.tie_every=1000`, `model.uni_chan=0` to match the controls. Seeds 1–8, 60k ticks, on the 4090 with MPS.
- **Controls:** TB (every update, vm-00067) and G16 (never, vm-00064), not rerun.

## The reading, set before the results

Over the last 20k ticks, seed by seed:

- **The common language survives:** T1K − G16 in poison is below zero beyond its interval, and T1K − TB in food includes zero or is above it. Rare averages then do what every-update averages do.
- **Averaging this rarely does nothing:** T1K − G16 in poison includes zero, and T1K − TB in poison is above zero beyond its interval.
- **It hurts:** T1K − TB in food is below zero beyond its interval.

## Verification

`tests/test_shared_tables.py`: tables are equal right after an average, apart between averages, and `tie_gap` is logged; an interval longer than the life never averages; an interval of 0 is refused. The default of 1 runs as before.

## Results

Pending.
