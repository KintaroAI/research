# Task 00030: Gated Memory at Equal Size (experiment vm-00024)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00024)

## Context

- vm-00022: at equal size, a memory of the most surprising quarter of chunks beats a random quarter in every seed. But
  against keeping everything (U) the memories differed in size as well as content: U held the whole life (12,500
  chunks), S a quarter of it, so S's chunks were replayed 4× as often.
- The question left open: when the memory is **full and the same size** for every gate, is it better to keep what was
  surprising (and what led to relief) than to keep the most recent steps?
- Long-term memory is a ring of `replay // ctx` chunks: when full, the oldest stored chunk is overwritten, whatever the
  gate.

## Design

`configs/U_universal.json` (the default universal model, 1-step instinct), **`online.replay = 25,000` steps (3,125
chunks)**, 200k ticks, seeds 1–3, three gates:

| Arm | Gate | Memory full from about |
|---|---|---|
| U | every chunk: the most recent 25k steps | 25k |
| S | `store_surprise = 0.25`: the most surprising quarter of chunks (task 00028) | 100k |
| RS | `store_relief = 1`, `store_surprise = 0.25`: every chunk that led to relief, and the surprising quarter of the rest (the hippocampus idea: interesting events are stored more often) | 50–70k |

From 100k on, all three memories are full, 3,125 chunks, and differ only in what they hold and how far back it reaches
(U: 25k steps, S: about 100k, RS: about 50–70k).

The reference with a memory of the whole life (100k steps, 4× larger): vm-00021 seeds 1–3, the same configuration at
200k ticks.

## Measures

- Online (metrics.jsonl): food and poison clicks per 1k and time at the hunger cap per 20k window; the move token's
  cross-entropy (policy breadth: vm-00022 saw a small memory collapse it); the memory's relief share and the token kinds
  that made chunks surprising.
- Frozen at 100k and 200k: the forage test (with the instinct, 32 worlds) and the click test on real windows
  (`analyze_hazards.py click-real`, task 00027); the same for vm-00021 seeds 1–3 where not already measured.

## Verification

No new code: the gates and the per-token surprise are tested (tasks 00007, 00028). The run's `ltm_chunks` reaches 3,125
and stays there; S's surprise threshold and stored counts as in vm-00022.

## Results (vm-00024)

- The memories filled at 26k (U), 57–79k (RS), 90–94k (S).
- U (the recent steps) collapsed: two of three lifetimes ended stuck at the hunger cap, the third clicked poison 44–49
  times per 1,000 steps; its click predictions decayed (poison right 0.26–0.66 at 200k).
- S and RS kept every lifetime working (frozen at the cap 0.04–0.08) and predict clicks as well as vm-00021's whole-life
  memory or better; S is the most even, RS forages most but one seed lost its foraging after 140k.
- The whole-life memory (4× larger) still clicks poison least.
