# Task 00008: What Keeps a Fact: Spacing, Learning Rate, Wordings (experiment 00021)

**Date:** 2026-11-04
**Status:** Deferred (implemented and checked; runs later, the owner)

## Context

- Experiment 00020 (task 00007), with a constant learning rate of 6e-4:
  - every size forgets a fact within a few thousand steps of its last showing;
  - size moves the forgetting curve up, not out;
  - a forgotten fact is relearned in about half the showings of a fresh one;
  - facts are recalled only in their training wording.
- The owner (2026-11-04) asked how facts get stored long term, and agreed to test the three things the literature names:
  - showings spread out against bunched;
  - a decaying learning rate against a constant one;
  - several wordings against one.
- A hint already in 00020's data (stream facts, shown n times 20 or 200 steps apart; recall 1–2k steps after the last of 32 showings):
  - G: .135 at 20 steps apart, .280 at 200;
  - L: .083 against .129;
  - M: .061 against .036.
  - So spacing may help, more so in larger models.
  - This is not a clean test: facts spread over 6,200 steps arrived earlier in the run than those spread over 620.

## Design

### Arms: the learning rate

| Arm | Learning rate after the 500-step warmup |
|---|---|
| C6 | 6e-4, constant (00020's) |
| C2 | 2e-4, constant |
| D | 6e-4 falling to 6e-5 along a cosine over the run |

- **Sizes:** L (10.6M non-embedding parameters) and G (85M, GPT-2 small); two seeds; 20,000 steps.
- These are the sizes where 00020 saw the most dynamic range: L learns in about 4 showings and G in 2, and both forget on the same time scale.
- **12 runs.** C6 is rerun because the stream now carries the new facts below.

### New facts in the stream (`--plus 1`; the default 0 keeps 00020's stream)

One new fact a step, in five conditions taken in turn, every one shown 8 times:

| Condition | Gap between showings | Span | Wording |
|---|---|---|---|
| gap 0 | all 8 in the same step | 0 | one |
| gap 10 | 10 steps | 70 | one |
| gap 100 | 100 steps | 700 | one |
| gap 1000 | 1,000 steps | 7,000 | one |
| words | 100 steps | 700 | four, in turn |

- 4,000 facts per condition. Facts whose 8 showings do not all fit in the run are left out of the tables.
- **The four training wordings,** each ending just before the city:
  - "{name} was born in" (the exact probe's form);
  - "The birthplace of {name} is";
  - "{name} is a native of";
  - "{name}'s home town is".
- The held-out probe, never trained, is 00020's paraphrase: "Where was {name} born? In".
- The control for the words condition is gap 100: the same showings at the same times, in one wording.
- **Probes:** half of the new facts (taken in turn), exact and paraphrase, every 500 steps, with 00020's probes unchanged.

### Measures

1. **Spacing:**
   - recall of the gap conditions by thousands of steps since the last showing;
   - "ever recalled" for each gap condition.
   - Intervals: bootstrap over facts, both seeds pooled, and each seed's sign reported.
2. **Learning rate, per arm and size:**
   - 00020's tables (repetition, forgetting, base facts, extraction, savings, held at the end);
   - the share kept at 1–2k steps of recall at 0–1k (n = 32).
3. **Wordings:**
   - paraphrase recall of the words condition against gap 100, by time since the last showing and ever;
   - exact recall of both.

## The readings, set before the results

1. **Plumbing:**
   - In a 300-step stub run, every new fact is shown exactly as scheduled: 8 times, at its gap, and the words facts in the four wordings in turn.
   - The learning rate follows its schedule (logged).
   - With `--plus 0` the stream and facts are 00020's, the same fact ids and showings.
2. **Spacing helps:** in G under C6, recall 1–4k steps after the last showing is higher at gap 100 than at gap 0, by more than its interval.
3. **A lower rate forgets slower:** under C2 the share kept at 1–2k is above C6's, in both sizes.
4. **Decay consolidates:** D holds more facts at the end of the run than C6 (stream facts, all n), in both sizes.
5. **Wordings let a fact be asked another way:** in G under C6, the words condition's ever-recalled share on the held-out paraphrase is above gap 100's by more than its interval, and above 5 times chance (.020).

## Implementation

- [x] `facts_lib.py`: the four wordings (`Facts.sentence_k`)
- [x] `train.py`: `--plus` (the new facts, their showings and probes), `--lr-sched const|cos`, `--lr-min`; the learning rate in `train.jsonl`
- [x] `analyze.py`: runs grouped by size and arm; spacing and wordings tables
- [x] Checks: a stub run's schedule, `--plus 0` equal to 00020's stream, peak memory of L and G
- [x] `00021-fact-retention/`: README, Makefile
- [ ] Runs (first start stopped for vm-00084, then deferred by the owner), write-up

## Verification

- The schedule check above, by the showings actually put in the batches.
- The `--plus 0` check: the same `meta.json` fact lists and showings as 00020.
- 300-step runs of L and G with `--plus 1`: peak GPU memory, to place six runs on each card.
