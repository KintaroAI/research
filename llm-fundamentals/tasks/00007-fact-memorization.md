# Task 00007: How a GPT Memorizes Facts, Forgets Them and Relearns Them (experiment 00020)

**Date:** 2026-11-03
**Status:** Planned

## Context

- The owner (2026-11-03): try "classic GPT-2 ... on how it memorizes stuff (how often it needs to be repeated) and how to extract it back - feed it with facts and see the dynamic of precise memorization of recent facts vs already stored (how fast old facts got forgotten under the stream of new facts per model size) and does increase in model size allow to store more for longer or not".
- And: "how fast it recall back forgotten fact - for example X times to repeat A until it start to recall A, after a while can not recall A, repeat A Y times until it start to recall A again and check if Y is less, equal or more than X".
- Why now: the visual-motor agent learns online, in one stream, and the question of what weights keep came up there (`visual-motor-agent/ARCHITECTURE.md` section 13, "Three memories"; the owner's correction: weights hold what is repeated and stays true, specific or general). Synthetic facts make every exposure countable.
- What is known, as landmarks to compare against:
  - Facts are acquired a little per exposure and forgotten along a power law.
  - Larger models memorize faster and forget less.
  - A fact stored in one wording is often not extractable in another (the reversal curse).
  - Saturated models hold about 2 bits per parameter (Allen-Zhu and Li).
- Relearning faster than first learning is Ebbinghaus's "savings" (1885); whether a transformer shows it, and how it depends on size, is the owner's new question.

## Deliverables

- `dev/facts/`: fact generation, the stream with scheduled and adaptive insertion, probes, the training loop (PyTorch, `~/venvs/vm-gpu`), and analysis.
- Experiment `00020-fact-memorization/`: README and Makefile.

## Design

### Facts

- Invented people and an attribute from a closed set: "Zorvan Kelt was born in Mirelith."
  - The names are made of tokens no TinyStories text uses together.
  - The attribute is one of 256 invented cities, each a single GPT-2 token if possible, else the first token is scored.
- Each fact carries 8 bits: capacity is counted in bits per parameter.
- **Three probe forms, never trained on:**
  - **exact:** the training sentence cut before the city;
  - **paraphrase:** "Where was Zorvan Kelt born? In";
  - **reverse:** "Who was born in Mirelith? It was" (only for cities given to a single person).
- **A probe's score:** the correct city's probability among the 256, and whether it ranks first. **Recalled** means it ranks first on the exact probe.

### The stream

- TinyStories text (`dev/data/tinystories`) with a fact sentence inserted every 200 tokens on average, at a sentence boundary.
- Plain next-token loss on everything, as in ordinary pretraining.
- Constant learning rate after a short warmup, so forgetting is not hidden by a decaying schedule.

### Models

- GPT-2 shapes from `create_model.py`, trained from scratch: about 1M, 4M and 16M parameters (stage 1), then 64M and GPT-2 small, 124M (stage 2).
- The same tokenizer, context 256, batch and learning rate per size as `create_model.py`'s TinyStories settings.

### Part A, scheduled exposures: how often, how long

- Each new fact is shown n times, n in 1, 2, 4, 8, 16, 32, with a gap of g tokens between showings, g in 2k and 32k.
- New facts keep arriving at the stream's rate, so later facts compete with earlier ones.
- **Probes:** every fact every 500 steps.
- **Curves:**
  - recall and probability against the number of exposures;
  - against facts seen since the fact's last exposure (the forgetting curve);
  - by model size.
- **Capacity:** the most facts recalled at once, in bits per parameter.

### Part B, old against new

- First, a base set of facts shown 32 times each, mixed in the text, until recalled.
- Then a stream of new facts only.
- **Curves:** the base facts' recall as the new ones arrive, by size and learning rate. This is "already stored against recent".

### Part C, savings: relearning a forgotten fact

For each fact in a test set:

1. **First learning:** show it, then probe after each showing, and stop at the first recall. X = the showings needed (capped at 64).
2. **Forgetting:** stop showing it, and keep the stream running until it is not recalled on two probes in a row. Record how long that took, and its probability at that point: a forgotten fact can keep a latent trace above chance (1 in 256).
3. **Relearning:** show it again in the same way until recalled. Y = the showings needed.
4. **Read:** Y against X, fact by fact; savings (X − Y) / X; and how it depends on the latent probability at step 2 and on model size.

Controls:
- **Fresh facts,** first shown at the same point in the stream as the relearning, so a lower Y is not just a better-trained model learning faster.
- **Forgotten after more delay:** some facts are left for two and four times as long after they are lost, so savings can be seen to fade, or not.

Probes in parts A and C are forward passes only: no gradient, and they never enter the stream.

## The readings, set before the results

1. **Plumbing:**
   - The facts are absent from TinyStories: no name occurs in it.
   - An untrained model scores 1 in 256.
   - A fact shown 64 times in a row is recalled by every model.
2. **Repetition:** recall rises with exposures in every size, and the larger models need fewer.
3. **Size and forgetting:** after the same number of later facts, the larger models keep more of an earlier fact's probability. Capacity in bits per parameter is roughly the same across sizes.
4. **Extraction:** the paraphrase probe recalls fewer facts than the exact one, and the reverse almost none.
5. **Savings:** Y < X for most facts, more than for fresh facts at the same point, so the forgotten fact left a trace. The savings shrink with the time since the fact was lost.

## Implementation

- [ ] `dev/facts/make_facts.py`: names, cities, sentences, probe forms; a check that no name is in TinyStories
- [ ] `dev/facts/stream.py`: TinyStories windows with inserted facts, scheduled (parts A and B) and adaptive (part C, driven by probes)
- [ ] `dev/facts/train.py`: GPT-2 from `create_model.py`'s configurations, constant learning rate, probes every N steps, results as rows of JSON
- [ ] `dev/facts/analyze.py`: the curves, capacity, savings
- [ ] Stage 1 (1M, 4M, 16M), then stage 2 (64M, 124M) when a GPU is free of the visual-motor runs
- [ ] `00020-fact-memorization/`: README, Makefile

## Verification

- Probes on an untrained model at chance.
- A single fact repeated until recalled, by hand, and its probe curve.
- Every model recalls a fact shown 64 times in a row.
- The adaptive loop: X counted correctly on a fact forced to be learned after a known number of showings (a stub model).
