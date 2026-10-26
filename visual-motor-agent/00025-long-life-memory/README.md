# vm-00025: A Long Life with a Fixed Memory

**Date:** 2026-10-26
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00025`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00025/`
**Task plan:** [`../tasks/00031-long-life-memory.md`](../tasks/00031-long-life-memory.md)

## Goal

vm-00024 showed, with a small memory, that keeping only the most recent steps traps the agent and keeping its surprising
experience does not. Here the memory is the size that held the whole life in vm-00021 (100k steps) and the life is 5×
longer (500k ticks), so every memory must choose. A third arm tests a trap guard that treats every chunk alike:
reservoir sampling, a uniform sample of the whole life.

## Hypothesis

1. Keeping the most recent 100k steps (vm-00021's memory, continued) traps the agent sooner or later.
2. A uniform sample of the whole life (reservoir) guards against traps: a trap fills only its share of the memory.
3. The surprising quarter keeps the agent working through the whole life, as at the small size.

## Method

`configs/U_universal.json` (the default universal model, the 1-step instinct), `online.replay = 100,000` steps (12,500
chunks), 500k ticks, seeds 1–3, three memories:

| Arm | Gate | Once full, a new chunk replaces | Full from | Reach at 500k (mean / oldest chunk's age) |
|---|---|---|---|---|
| U | every chunk | the oldest | 101k | 50k / 100k steps |
| R | every chunk | a random one, kept with probability cap / k (`ltm_evict = reservoir`, task 00031) | 101k | 249k / 500k |
| S | the most surprising quarter (task 00028) | the oldest | 392–394k | 202k / 404k |

Four lifetimes ran on the 3090 (58 ticks/s each), five on the 4090 (80), 07:07–09:30; the analysis until 09:51.
Measures: online clicks and time at the hunger cap; the frozen tests at 200k and 500k (eval_self, the forage test with
the instinct on 32 worlds; the policy alone at 500k); the click test on real windows (task 00027) at 200k and 500k.

## Log

### 2026-10-26
- **Implemented** reservoir eviction (`online.ltm_evict`, task 00031) and the memory's reach in the metrics.
- **Ran** the 9 lifetimes 07:07–09:30; analysis until 09:51.

## Results

### R1. Online

Food / poison clicks per 1,000 steps and the share of time at the hunger cap, in 20k windows:

| Run | 80–100k | 180–200k | 280–300k | 380–400k | 480–500k | at the cap (same windows) | move cross-entropy 180–200k → 380–400k → 480–500k | relief share of memory at 200k, 500k |
|---|---|---|---|---|---|---|---|---|
| U s1 | 6.5 / 4.5 | 7.8 / 2.8 | 2.7 / 0.8 | 0.0 / 58.6 | 0.0 / 44.3 | 0.03 0.04 0.59 **1.00 1.00** | 0.35 → 0.07 → 0.03 | 0.12, 0.00 |
| U s2 | 7.3 / 4.0 | 7.1 / 6.7 | 2.8 / 1.6 | 0.7 / 0.5 | 0.0 / 55.5 | 0.25 0.10 0.50 0.80 **1.00** | 0.31 → 0.07 → 0.05 | 0.11, 0.02 |
| U s3 | 3.8 / 6.3 | 2.9 / 1.6 | 2.9 / 13.1 | 1.8 / 5.2 | 0.0 / 84.8 | 0.13 0.06 0.18 0.59 **1.00** | 0.25 → 0.07 → 0.04 | 0.06, 0.02 |
| R s1 | 5.0 / 17.5 | 4.8 / 13.3 | 4.0 / 9.8 | 1.9 / 6.3 | 0.0 / 0.0 | 0.18 0.14 0.12 0.59 **1.00** | 0.34 → 0.22 → 0.16 | 0.09, 0.07 |
| R s2 | 4.7 / 2.5 | 3.4 / 1.9 | 2.5 / 2.0 | 1.8 / 0.6 | 2.7 / 0.6 | 0.02 0.05 0.06 0.11 0.07 | 0.47 → 0.31 → 0.29 | 0.08, 0.06 |
| R s3 | 6.2 / 4.0 | 5.8 / 5.2 | 4.5 / 3.4 | 3.9 / 5.9 | 3.4 / 4.6 | 0.01 0.05 0.04 0.16 0.25 | 0.35 → 0.22 → 0.19 | 0.13, 0.09 |
| S s1 | 17.7 / 25.1 | 15.9 / 15.3 | 18.4 / 14.2 | 15.9 / 9.6 | 16.1 / 9.6 | 0.15 0.11 0.07 0.06 0.04 | 0.30 → 0.46 → 0.51 | 0.42, 0.42 |
| S s2 | 5.2 / 7.7 | 5.2 / 11.2 | 5.1 / 25.4 | 5.8 / 12.8 | 8.2 / 5.0 | 0.09 0.36 0.39 0.20 0.05 | 0.23 → 0.33 → 0.36 | 0.22, 0.23 |
| S s3 | 6.2 / 10.2 | 7.9 / 9.1 | 8.9 / 4.1 | 8.7 / 5.8 | 11.8 / 3.9 | 0.18 0.05 0.01 0.03 0.02 | 0.25 → 0.39 → 0.43 | 0.25, 0.29 |

The time at the cap per 20k window over the whole life, for the trapped runs:

| Run | 0 – 500k in 20k windows |
|---|---|
| U s1 | 0.1 0.1 0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0 0.0 0.2 0.2 0.5 0.6 0.1 0.6 0.3 0.6 1.0 1.0 1.0 1.0 1.0 1.0 |
| U s2 | 0.1 0.0 0.1 0.1 0.2 0.1 0.0 0.2 0.3 0.1 0.1 0.6 0.2 0.7 0.5 0.0 0.1 1.0 1.0 0.8 0.1 0.4 1.0 1.0 1.0 |
| U s3 | 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.0 0.1 0.1 0.2 0.7 0.1 0.1 0.2 0.7 0.7 0.7 0.7 0.6 0.6 0.4 0.3 0.6 1.0 |
| R s1 | 0.2 0.1 0.1 0.1 0.2 0.1 0.1 1.0 0.2 0.1 0.1 0.2 0.3 0.3 0.1 0.5 0.3 0.1 0.2 0.6 0.3 0.4 0.3 0.2 1.0 |

S's chunks were made surprising by the move token (10,372–12,537), the body token (1,998–2,801) and patches
(1,148–2,490).

### R2. Frozen

| Run | eval_self success 200k / 500k | forage with the instinct, 200k: food / poison (cap) | 500k | policy alone, 500k | click test 200k: food / poison right, nothing flat | 500k |
|---|---|---|---|---|---|---|
| U s1 | 0.95 / 0.66 | 13.4 / 4.6 (0.02) | 5.5 / 11.8 (0.28) | 10.9 / 31.8 (0.44) | 0.66 / 0.78 / 0.86 | 0.17 / 0.83 / 0.58 |
| U s2 | 1.00 / 0.82 | 12.2 / 5.2 (0.03) | 6.5 / 1.3 (0.02) | 3.5 / 1.7 (0.07) | 0.82 / 0.83 / 0.90 | 0.78 / 0.59 / 0.91 |
| U s3 | 0.71 / 0.70 | 4.2 / 2.0 (0.01) | 3.3 / 1.6 (0.04) | 2.4 / 1.8 (0.04) | 0.60 / 0.63 / 0.88 | 0.67 / 0.62 / 0.83 |
| R s1 | 0.98 / 0.91 | 9.1 / 10.5 (0.07) | 5.9 / 9.3 (0.07) | 5.6 / 9.2 (0.11) | 0.53 / 0.69 / 0.69 | 0.52 / 0.58 / 0.83 |
| R s2 | 0.74 / 0.69 | 4.4 / 0.5 (0.00) | 2.9 / 2.0 (0.09) | 2.4 / 1.9 (0.04) | 0.59 / 0.80 / 0.78 | 0.67 / 0.55 / 0.81 |
| R s3 | 0.99 / 0.70 | 16.1 / 7.1 (0.03) | 5.0 / 2.8 (0.01) | 3.9 / 6.5 (0.07) | 0.83 / 0.65 / 0.93 | 0.65 / 0.73 / 0.89 |
| S s1 | 1.00 / 0.99 | 19.4 / 12.7 (0.07) | 23.1 / 11.9 (0.05) | 19.7 / 22.3 (0.16) | 0.68 / 0.85 / 0.91 | 0.63 / 0.90 / 0.93 |
| S s2 | 0.96 / 0.98 | 9.7 / 12.8 (0.09) | 12.7 / 5.5 (0.03) | 9.8 / 10.0 (0.07) | 0.61 / 0.65 / 0.76 | 0.61 / 0.82 / 0.88 |
| S s3 | 0.98 / 0.99 | 12.9 / 9.2 (0.06) | 17.3 / 4.9 (0.02) | 10.5 / 9.2 (0.06) | 0.51 / 0.80 / 0.68 | 0.56 / 0.91 / 0.87 |

## Analysis

- **The most recent 100k steps trap every lifetime** (hypothesis 1, supported): U s1 from 380k, s2 from 340k (with
  one escape), s3 at the end; once trapped, the agent sits at the hunger cap and clicks poison 44–85 times per 1,000
  steps with a collapsed policy (the move token's cross-entropy 0.03–0.05, from 0.25–0.35 at 200k). The first 200k
  repeat vm-00021 (no trap yet); the memory turns against the agent only when it holds nothing but the recent decline.
  Frozen in fresh worlds, U s2 and s3 are not trapped (at the cap 0.02–0.04): the trap is the lifetime's state as much
  as the weights. U s1's model is damaged as well (it predicts relief for food in 17% of the cases, eval_self 0.66).
- **The reservoir delays traps but does not prevent them** (hypothesis 2, partly): R s1 was trapped for 20k steps at
  140–160k and escaped (a trap can no longer fill the memory), but was trapped again in the last 20k; R s2 and s3 stayed
  off the cap (0.07 and 0.25 at the end). Its memory, a uniform sample of the life, holds few relief chunks (0.06–0.09),
  its policy narrows more slowly than U's (0.16–0.29 at 500k) and its agent forages little (food 2.7–3.4 per 1,000
  online, 2.9–5.9 frozen).
- **The surprising quarter keeps every lifetime working and improving** (hypothesis 3, supported): no S run was
  trapped; in the last 20k all three were at the cap 0.02–0.05 of the time and clicked poison 3.9–9.6 times per 1,000,
  among their fewest of the life. Frozen at 500k they behave best and predict poison best: eval_self 0.98–0.99, food 12.7–23.1 per
  1,000, at the cap 0.02–0.05, poison clicks predicted right 0.82–0.91. They also improve from 200k to 500k (poison
  right 0.65–0.85 → 0.82–0.91), while U's and R's eval_self success falls (0.71–1.00 → 0.66–0.91). Food
  clicks are predicted right no better than by the others (0.56–0.63).
- **Why surprise guards against traps:** its policy *widens* over the life (the move token's cross-entropy 0.23–0.30 at
  200k → 0.36–0.51 at 500k) while U's collapses. 66–80% of the chunks it keeps were surprising because of the move token:
  moves the model did not expect of itself. A memory of the agent's unexpected moves is a memory of its variety; the
  model rehearses that variety, and its habits cannot narrow to a trap. Its memory also holds far more relief chunks
  (0.23–0.42 of chunks, against 0.00–0.09). No entropy floor or special rule was needed: the same per-token surprise rule does it.

## Conclusions

1. **In a life longer than the memory, keeping the most recent experience is a trap**, every time; uniform sampling of
   the whole life delays it.
2. **Memory gated by the model's own surprise is the rule to use:** in all three 500k-tick lives it kept the agent
   working and still learning, with the best prediction and behaviour of the three arms at the end. Most of what it keeps
   is the agent's own unexpected moves, which keeps the policy broad.
3. Surprise gating (`store_surprise = 0.25`, `store_base = 0`) should become the default memory for the universal model
   in long lives.

## Next Steps

- Make surprise gating the default in `configs/U_universal.json`, with a test that the default memory is gated.
- Longer still (1M+ ticks): does S stay off the cap once its memory has been full for longer (it was full only from
  ~390k)? Surprise with the reservoir instead of the ring.
