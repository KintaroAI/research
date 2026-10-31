# vm-00054: The Age Key and a Key and Value Ring

**Date:** 2026-10-31
**Status:** Complete
**Source:** `exp/vm-00054` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00054/`
**Task plan:** [`../tasks/00062-age-key-ring.md`](../tasks/00062-age-key-ring.md)

## Goal

Every tick the window shifts by one and everything is recomputed. The owner's direction: positions that do not move, and
inference on top of old results, accepting slightly off results in the ring. Do the best models survive that, and how fast
can a tick get?

## Method

- **`model.age_key`:** no step embedding in the tokens; `wpe`'s rows are keys read at attention by age, so a token's K and
  V do not depend on the window. Same parameter count. Verified against the reference (`tests/test_age_key.py`).
- **`online.ring`:** acting computes only each new step's tokens, on a per-layer buffer of keys and values. Exact against a
  full pass for the first ctx ticks, and against a streaming torch version past them (`tests/test_ring.py`). Past the
  window it is not a full pass: a token keeps what it read when new. Training is unchanged.
- **Runs:** seeds 1–3, 60k ticks, the no-overlap world: QNB (QN with the age key), QNA (with the ring too), QCA (QCN,
  council, with both). Seed 1, 40k ticks: T0–T3 (vm-00050's text quarters Q with the age key).
- **Speed:** QN alone on the 3090; stage timing in `VM_PROF` builds.

## Results

### R1. Foraging over the last 20k ticks, per 1k ticks

| arm | food | seeds | poison | seeds | food share | smoothness |
|---|---|---|---|---|---|---|
| QN (control) | 43.5 | 52.2, 31.9, 46.5 | 3.5 | 3.9, 3.9, 2.8 | 0.93 | |
| QNB (age key) | 33.9 | 22.9, 43.6, 35.1 | 4.8 | 5.0, 6.5, 2.8 | 0.88 | 0.33 |
| **QNA (age key, ring)** | **40.5** | 21.9, 40.5, 59.2 | 4.0 | 4.6, 4.3, 3.0 | 0.91 | 0.35 |
| QCN (control) | 34.9 | 41.1, 25.6, 37.8 | 2.1 | 1.1, 3.6, 1.6 | 0.94 | |
| **QCA (council, age key, ring)** | 29.1 | 18.3, 36.3, 32.9 | **1.2** | 1.6, 1.1, 0.9 | **0.96** | 0.53 |

### R2. The ring on the same frozen weights (QNA, 800 ticks, means of three seeds)

| acting | food | poison |
|---|---|---|
| ring | 41.1 | 3.8 |
| full window | 42.4 | 3.3 |

### R3. Text: next-letter cross-entropy (val), quarters with every summary

| run | 10k | 20k | 30k | 40k |
|---|---|---|---|---|
| Q (step embedding, vm-00050) | 1.990 | 1.761 | 1.724 | 1.703 |
| **age key** | 2.009 | 1.761 | 1.708 | **1.681** |

### R4. Speed

| per piece | before | after |
|---|---|---|
| full window pass | 0.91 ms | 0.80 ms |
| ring step | 0.40 ms | 0.32 ms |
| update, 8 windows | 4.2 ms | 3.45 ms |

QN alone, no timing waits: 50.4 to **57.7 ticks/s** from batching the attention GEMMs; with the age key and the ring, 44.8 to
50.6. The update is ~70% of a tick; a call costs ~33 µs, whatever it computes.

## Analysis

- **The age key costs nothing.** In text it is the best result so far (1.681, one seed, against 1.703), still falling at 40k.
- **The seed spread swamps the foraging comparison.** QNB and QNA differ only in acting, yet QNA is 6.6 food ahead; QNA's
  seeds run from 21.9 to 59.2, the highest and nearly the lowest seen. So QN, QNB and QNA cannot be separated with three seeds.
- **The ring changes nothing that matters:** frozen, on the same weights, food and poison stay within noise, and over a
  lifetime QNA is if anything ahead of QNB. The "slightly off" results are harmless here.
- **QCA has the lowest poison of any arm** (1.2, every seed under 1.7; 0.9 in the last 10k), at 29.1 food. Its first 10k–20k
  are striking too: poison 3.5 against gain's 10–13.
- **The ring is an acting gain.** A ring step is 2.5× cheaper than a window pass, but candidates and the move write-back
  stay, and the update dominates: the ring takes ~2 ms from a ~20 ms tick. It pays when acting is separated from learning.
- **Speed follows GPU calls, not arithmetic.** Cutting attention GEMMs by 25% of their FLOPs and doubling their calls made
  an update slower; one batched call instead of four made it 16% faster.

## Conclusions

1. Position by age works, in vision and text, with the same parameters. It can replace the step embedding.
2. Acting on the ring is safe. The owner's "slightly off" is, in these runs, indistinguishable.
3. The next speed gains are in calls per update, and in sharing work between pieces.
4. Comparing foraging arms needs more than three seeds: the spread between seeds is larger than most effects here.

## Next Steps

- More seeds for the arms that matter: more lifetimes are now affordable.
- Text quarters with the age key past 40k: still falling.
- Fewer calls: `linear`'s bias kernels, one upload of a batch shared by the pieces.
- Separate acting from learning, where the ring pays.
