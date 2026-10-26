# Task 00031: A Long Life with a Fixed Memory (experiment vm-00025)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00025)

## Context

- vm-00024: with a memory of a quarter of the life (25k steps of 200k), keeping the most recent steps trapped two of
  three lifetimes at the hunger cap; keeping the surprising chunks kept all of them working. The whole-life memory
  (vm-00021, 100k steps) was still the most cautious, but in 200k ticks it held half the life.
- Here the memory is vm-00021's size (100k steps, 12,500 chunks) and the life 5× longer (500k ticks), so every arm must
  choose what to keep.
- A trap guard that treats every chunk alike (the second next step of vm-00024): **reservoir sampling**. Once the memory
  is full, the k-th chunk the gate stores is kept with probability cap / k and replaces a random stored chunk, so the
  memory stays a uniform sample of the whole life, its beginning included. A trap fills only its share of the life,
  never the whole memory.

## Implementation

- `online.ltm_evict`: `oldest` (default, the ring as before) | `reservoir`, in `ChunkStore.slot`. The gate's decision
  is unchanged; the eviction rule only says where a stored chunk goes and whether it is kept.
- Metrics: `ltm_not_kept` (chunks the gate chose but the reservoir did not keep), `ltm_age_mean` and `ltm_age_max`
  (how far back the memory reaches, in steps).

## Verification

- `tests/test_online_memory.py`: a full ring keeps the most recent chunks (and the age metric); the reservoir keeps
  whole chunks, the expected number of writes (cap (1 + ln(n / cap))), and a uniform sample over the life (20 seeds:
  every tenth of the life within 25% of its share); an unknown rule is refused. Mutation check: replacing a random chunk
  every time (no cap / k) fails the uniformity test.
- A 4,000-tick lifetime with a 100-chunk reservoir: full at 100, the oldest chunk stays at the start of the life.

## Experiment vm-00025

`configs/U_universal.json`, `online.replay = 100,000` (as vm-00021), 500k ticks, seeds 1–3:

| Arm | Gate | Eviction | What the memory holds at 500k |
|---|---|---|---|
| U | every chunk | oldest | the last 100k steps (vm-00021's arm, continued) |
| R | every chunk | reservoir | a uniform sample of the whole life |
| S | the most surprising quarter (task 00028) | oldest | the surprising chunks of the last ~400k steps (full from ~370k) |

Measures: online food and poison clicks and time at the hunger cap per 20k window, the move token's cross-entropy, the
memory's relief share and reach; frozen at 200k and 500k: the forage test (with the instinct and the policy alone) and
the click test on real windows.

## Results (vm-00025)

- U (the last 100k steps): every lifetime ended trapped at the hunger cap (from 340–480k), the policy collapsed.
- R (reservoir): one lifetime trapped (it escaped a first trap at 160k, the second came in the last 20k); the others
  stayed off the cap but foraged little.
- S (surprise): no trap; at 500k the best behaviour and poison prediction, still improving; its policy widened over the
  life, as 66–80% of its chunks were the model's own unexpected moves.
