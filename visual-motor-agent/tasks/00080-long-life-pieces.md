# Task 00080: Long Lives for Many Pieces (experiment vm-00069)

**Date:** 2026-11-01
**Status:** In progress

## Context

- 60k ticks may be about 100 minutes of lived time (§13 of `../ARCHITECTURE.md`). The owner asked for longer lives without very slow models.
- TB (vm-00067) was the only arm still climbing at 60k: food 42, 46, 48, 51 per 1k by 10k block. It also has low poison and about half the distinct parameters.
- In vm-00031 two of three 500k lives with a plain memory of 100k steps ended at the hunger cap; surprise gating avoided that. `lifetime.py` refuses surprise gating with `online.pieces`, so here the memory holds the whole life instead: nothing is forgotten. About 2 GB of RAM per life. Whether this avoids the trap is part of the question.

## Design

| arm | model | seeds | where | time |
|---|---|---|---|---|
| **LT** | 16 pieces, `online.tie_tables=inout` | 1–4 | 4090 | about 10 hours (13.8 ticks/s each, measured for 1,000 ticks) |
| **LD** | the default, 4 pieces | 1–4 | 3090, once vm-00065 ends | about 7 hours (an estimate) |

Both: `gain`, `configs/U_default.json` (channels on), `online.steps=500000`, `online.replay=500000`, `online.save_every=50000`.

**Short of a control:** LT and LD have channels on, TB and NA did not; to 60k they are read beside them, not against them. Seeds are 4, df 3: t = 3.182.

## Measures

Food, poison, food share, hunger and patch, body and move CE by 50k ticks; the same over the last 20k ticks. A frozen check with `eval_pieces.py --ring` at 500k.

## The readings, set before the results

1. **Trapped:** a life is trapped if its mean hunger over the last 20k ticks is above 0.9 or its food is under 1 per 1k. No trapped life means the whole-life memory avoids what the ring did not, for these seeds.
2. **Long lives pay** for an arm if its food over the last 20k ticks minus its own 40–60k, seed by seed, is above zero beyond its interval, or its poison is below zero beyond its interval, with food not below.
3. **Pieces hold their lead** if LT − LD in food over the last 20k ticks is above zero beyond its interval.

## Verification

Short trial: 2,200 ticks of LT with `online.replay=500000`, 13.8 ticks/s against 13.8 with the smaller memory, no error. The existing suite otherwise: no code change.

## Results

Pending.
