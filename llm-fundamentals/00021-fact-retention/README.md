# 00021: What Keeps a Fact: Spacing, Learning Rate, Wordings

**Date:** 2026-11-04
**Status:** Deferred (the owner, 2026-11-04: to run later; code and checks done, no runs)
**Source:** `dev/facts/` (not yet tagged)
**Outputs:** `~/data/research/llm-fundamentals/exp_00021/`
**Task plan:** [`../tasks/00008-fact-spacing-lr-wording.md`](../tasks/00008-fact-spacing-lr-wording.md)

## Goal

00020 found that every size forgets a fact within a few thousand steps of its last showing, at a constant learning rate of 6e-4, and that facts are recalled only in their training wording. The owner asked what makes a fact stay. This tests the three answers the literature gives:
- showings spread out against bunched together;
- a decaying learning rate against a constant one;
- several wordings against one.

The design and the readings set before the results are in the task plan.

## Method

- **The stream:** 00020's, unchanged (base, stream, test and reverse facts), plus one new fact a step (`--plus 1`), each shown 8 times:

  | Condition | Showings | Wording |
  |---|---|---|
  | gap 0 | all 8 in the same step | one |
  | gap 10 | 10 steps apart | one |
  | gap 100 | 100 steps apart | one |
  | gap 1000 | 1,000 steps apart | one |
  | words | 100 steps apart | four in turn: "{name} was born in", "The birthplace of {name} is", "{name} is a native of", "{name}'s home town is" |

  The held-out probe for the words condition is 00020's paraphrase, "Where was {name} born? In", never trained.
- **Arms:** C6 (6e-4 constant, 00020's), C2 (2e-4 constant), D (6e-4 falling to 6e-5 along a cosine after the 500-step warmup).
- **Sizes:** L (10.6M non-embedding parameters) and G (85M), two seeds, 20,000 steps; seed 1 on the 4090, seed 2 on the 3090.
- **Checks before the start:**
  - With `--plus 0` the facts and their showings are 00020's, fact for fact (`meta.json` equal to 00020's S run). With `--plus 1` the old facts are unchanged and the new ones come after them.
  - A 2,400-step stub run logged every showing: all 2,400 new facts were shown exactly as scheduled, the words facts in the four wordings in turn. The cosine schedule logged 6.0e-4 at step 500, 3.1e-4 at 1,500 and 6.4e-5 at 2,300.
  - 2,000-step runs with the new facts: G peaks at 5.6 GB and 9.5 steps/s alone, L at 2.3 GB and 12 steps/s.
- **Run notes:** first started at 14:45 on 2026-11-04 and stopped 23 minutes in, at the owner's request, so that vm-00084 runs first; the partial outputs are in `stopped_1445/`. The 12 runs were then deferred by the owner, to run later.
- **A hint from 00020's data:** recall 1–2k steps after the last of 32 showings was .280 for facts shown 200 steps apart and .135 at 20 steps apart in G (L .129 against .083, M .036 against .061). The facts spread wider arrived earlier in the run, so this is not a clean test.
