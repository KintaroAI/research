# 00020: Fact Memorization, Forgetting and Relearning

**Date:** 2026-11-03
**Status:** Running: stage 1 (S, M) since 21:34; stage 2 (L, X, G) queued after the visual-motor runs
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
