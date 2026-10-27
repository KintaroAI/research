# Task 00040: Predicting the Future in the Text World (experiment vm-00033)

**Date:** 2026-10-27
**Status:** Complete (results in vm-00033)

## Context

- In the circle world the model's dream (its own predictions fed back, `eval_dream.py`, `dream_video.py`) drifts
  within a few steps; self-forcing (task 00039) keeps the cursor closer. Text gives a cleaner test of the same ability
  with a classical reference: dreaming in the text world is text generation, and a character-level GPT does the same.
- At equal data the universal model with bin embeddings matches the classical GPT teacher-forced (vm-00030: 1.718
  nats per letter at 30k ticks of a lifetime, seed 1; GPT on the same lifetime's batches: 1.716 at 30k updates,
  vm-00028). Does it also generate as well?

## Design (`reference/text_dream.py`)

- **Models:** the universal model of vm-00030 (arm B with bin embeddings, `T_text_B.json`, seed 1) at 30k ticks (its
  best), and the classical GPT (`char_gpt.py --data lifetime`, seed 1) trained for 30k updates (its best), saved.
- **The universal model generates as it dreams:** the window is its last ctx steps (pages); each new page's letters
  are produced one token at a time, each from the forward pass over the window so far (exactly sequential), as a
  glyph: the letter's probability is the product over the token's values of the probability of its glyph's value,
  normalized over the alphabet (eval_predict's glyph probability). The body is the move token's prediction (the drives,
  ~0); the move is sampled from the move token's prediction (as in its lifetime, no instinct). The forward pass is the
  PyTorch reference's (`torch_reference.forward_uni`), on the GPU.
- **GPT** generates letter by letter from its last 512 letters (its context; the universal model's is 7–8 pages,
  448–512 letters).
- **Starts:** W windows of the val split (eval_predict's tiled readers), each 8 real pages of context, then G pages
  (64 G letters) generated.

## Measures

1. **Greedy continuation against the truth** (the dream curve for text): each model continues greedily; the share of
   letters equal to the real text at each position after the context (letters 1, 2–8, 9–64, 65–256), and the position
   of the first error.
2. **Samples judged by the other model:** continuations sampled at temperature 1; nats per letter of each model's
   samples under the classical GPT and under the universal model (teacher-forced), against the real continuations
   under each (a sample as likely as real text is as good as the model can tell).
3. **Text:** a few samples of each, from the same contexts.

## Results (vm-00033)

- Teacher-forced on the real continuations: universal 1.668, GPT 1.683 nats per letter.
- Greedy continuations leave the real text within a letter or two for both (letter 1 right 47% / 52%).
- Samples judged by the other model: the universal model's samples under GPT 1.673 (real text 1.683); GPT's under the
  universal model 1.779 (real 1.668). The universal model generates as well as the classical GPT.
- A letter's probability is the mean of its 4 values' log-probabilities (the sum would be a temperature of 1/4).
