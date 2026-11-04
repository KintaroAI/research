# 00020: Fact Memorization, Forgetting and Relearning

**Date:** 2026-11-03
**Status:** Stage 1 done (M, two seeds); stage 2 (S again, L, X, G) queued after the visual-motor runs
**Source:** `dev/facts/` (not yet tagged)
**Outputs:** `~/data/research/llm-fundamentals/exp_00020/`
**Task plan:** [`../tasks/00007-fact-memorization.md`](../tasks/00007-fact-memorization.md)

## Goal

How often must a GPT see a fact to recall it, how fast does it forget under a stream of new ones, does size help, and does a forgotten fact come back faster than it was first learned (the owner's savings question)? The design and the readings set before the results are in the task plan.

## Method

- **The facts:** "Zorvan Kelt was born in Mirelith": invented people, and the attribute one of 256 GPT-2 tokens that never occur in TinyStories (8 bits a fact). 32 more tokens are kept for 32 reverse facts with one person each. 61,140 facts in all, the same in every run; no name occurs in TinyStories.
- **The stream:** TinyStories windows of 256 tokens with fact sentences spliced in, each after an `<|endoftext|>`, about one per window. 20,000 steps of 32 windows, AdamW, learning rate 6e-4 constant after a 500-step warmup.
- **One run covers the three parts:**
  - base facts, 32 showings in the first 4,000 steps;
  - stream facts, 3 new a step, shown 1 to 32 times, 20 or 200 steps apart;
  - 300 test facts learned, forgotten and relearned adaptively, each relearning paired with a fresh fact first learned at the same step;
  - 32 reverse facts.
- **Recall:** the right city ranks first among its kind's cities. Each city is scored by its logit minus its mean logit for 8 never-shown names in the same form, so a city made likely by its own recent showing, whoever is asked, does not count.
- **Sizes:** S (2 layers, d 128, 0.40M non-embedding parameters), M (4, 256, 3.2M), L (6, 384, 10.6M), X (8, 512, 25M), G (12, 768, 85M, GPT-2 small); two seeds each. Embeddings add 6.4M to 38.6M.
- **Checks before the start:**
  - Untrained, recall is at chance: .002 against 1 in 256, mean rank 126.5 against 127.5; reverse .031 against 1 in 32.
  - The chunked loss equals the plain one (gradients within 3e-8).
  - 1,000- and 3,000-step trial runs.
- **Why the neutral baseline:** a first trial had a test fact "recalled" after one showing, 229 steps into training, because a model that has just seen a city token makes it likely after any name. Scored against the neutral names, that is not recall.
- **The trial at size M, 3,000 steps** (not a result: one short run):
  - facts shown 32 times recalled .127 at the end, shown once .000;
  - relearning took a median of 8 showings, against 18 for fresh facts learned at the same step;
  - those savings were nearly gone at 4 times the delay.

## Notes on the runs

- **Both S runs ran out of GPU memory in a probe on the 3090 beside the visual-motor runs, seed 1 about 2,500 steps in and seed 2 later.** The probe computed full-vocabulary logits at every position of up to 2,048 prompts and used only the last. It now computes the last position only, with the same logits. Both are redone in stage 2; their logs are kept as `S_s1.crashed.log` and `S_s2.crashed.log`. The M runs, done with the first version of the probe, give the same numbers.
- **Stage 1 slowed vm-00080's 3090 lives** from 3.7 to as low as 1 tick/s, so the visual-motor queue ends about an hour later.

## Stage 1, size M (interim; the readings wait for all sizes)

Two seeds, 20,000 steps, from `facts/analyze.py`:

- **Savings:** for 600 test facts with a control, first learning took a median of 7.5 showings, relearning 3.0, and a fresh fact learned at the same moment 7.0. Y < X for 87% of facts, and Y < the fresh fact's X for 87%. Savings against the fresh fact, by the delay before relearning (times the time it took to forget): 1, .56; 2, .51; 4, .30.
- **Repetition:** recalled at least once by 7% of facts shown once, 34% at 8 showings, 85% at 32.
- **Forgetting:** facts shown 32 times are recalled 33% of the time within 1,000 steps of their last showing, 5% at 1,000–2,000 steps, under 1% after 5,000.
- **Base facts:** recall peaks at 9% when their showings end (step 4,000), and is near chance by step 6,000.
- **Extraction:** the paraphrase recalls 0.6% at the end; the reverse 6%, against 3% by chance.
- **Held at the end:** 442 facts, about 0.001 bits per non-embedding parameter.
