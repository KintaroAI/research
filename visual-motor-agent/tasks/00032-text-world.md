# Task 00032: A Text World for the Universal Model (experiment vm-00026)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00026)

## Context

- Foraging is a noisy measure of the model: prediction, the instinct, memory and the world's traps all mix, and there
  is no reference for the best possible score. Text has one number, nats per letter on held-out text, and a classical
  reference (a character-level GPT of the same size).
- **The principle (the owner's):** the model does not know what world it lives in. Text and pixels must be swappable at
  any time. Nothing in the model treats text differently from pixels; only the world draws its state into a frame.
  When a world shows a limitation of the model, the model is changed so that the limitation goes away **for every
  world alike**, and it keeps working the same for pixels and text.
- Tiny Shakespeare (`llm-fundamentals`, 1,115,394 characters, **65 distinct**: `\n`, space, `!$&',-.3:;?`, A–Z, a–z).

## The text world (`world/text_world.py`, `TextWorld`, `TextConfig`)

The same interface as `CircleWorld` (`observe`, `step`, the info dict), so `lifetime.py` runs it unchanged:

- **A letter is a brightness:** the corpus's sorted alphabet gives letter k the grey level round(k · 255 / (n − 1)),
  n = 65, the same scale as pixels (values / 255 in [0, 1]).
- **A frame is a page:** a `size` × `size` grey frame of letters, each letter a `cell` × `cell` block of its level.
  The page is written in `block` × `block` blocks of consecutive letters, blocks in raster order, letters within a
  block row by row (block 1: an ordinary page, line by line).
- **Each step shows the next page** of the text (one page = (size / cell)² letters); every body reads from its own
  seeded offset, wrapping around at the end of its split. Train: the first 90% of the corpus; val: the last 10%
  (as `llm-fundamentals` 00001).
- The moves are ignored (the agent acts as always; nothing it does changes the text). Drives are constant 0 (no hunger,
  no relief); the body and move tokens stay in their slots.

## The model: unchanged, two configurations

| Arm | Frame | Letter | Token | Letters per token | Tokens per step | Letters per step / context |
|---|---|---|---|---|---|---|
| **A** (64 letters in one token, as 64 pixels) | 8 × 8 px, `cell` 1, `block` 8 | 1 px | the 8 × 8 patch | 64, predicted together | 3 | 64 / 512 |
| **B** (one letter per token) | 16 × 16 px, `cell` 2, `block` 1 | a 2 × 2 block | a 2 × 2 patch | 1 (its 4 values the same letter) | 66 | 64 / 512 |

Both: grayscale, `ctx` 8, d 128, 2 layers, 4 heads (the default universal model), **`bins = 256`**: every grey level is
a bin centre, so the two-hot target of a letter is one-hot and a value's predicted distribution is a softmax over grey
levels, i.e. over letters (the 191 levels that are no letter get probability ~0 after learning). The cross-entropy of a
letter's value is then exactly −log p(letter). (32 bins would put 65 letters on 32 bins: two letters per bin, their
mixtures ambiguous.)

The two arms see the same text with the same context; they differ only in packing: A predicts a token's 64 letters in
parallel, each without seeing the others; B predicts letter after letter, each seeing all before it. A − B is the cost
of predicting a token's values independently (the limitation that lost the cursor in task 00029), in nats per letter.

## Measures

- **`eval_predict.py`** (new, any world): the frozen agent lives in a world (here the val split) for n steps with its
  own moves (the policy alone); the windows are scored by the PyTorch reference's per-value cross-entropy, by slot, and
  by value position within a token. For text: nats per letter on the val split (the patch tokens' values; in B each
  letter's 4 values averaged), and for A the curve over the 64 positions in a token.
- **The classical reference** (`reference/char_gpt.py`, new): a character-level GPT with the same width, depth, heads,
  feed-forward, context (512 letters) and optimizer (AdamW, lr 3e-4), softmax over 65 letters, trained on the same
  train split with the same number of letters per update (8 windows × 512), offline (shuffled windows); val nats per
  letter. Sanity anchor: llm-fundamentals 00001 (210k parameters, 32 letters of context, val 1.82).
- **The universal model offline** (shuffled windows, the same updates as the reference) and **in a lifetime**
  (`lifetime.py`: the text streamed, memory U): the first gap is the token design, the second learning online.

## Verification

- The world: a page decodes back to the text (both layouts); consecutive steps continue the text; bodies read
  different offsets; splits do not overlap; `get_state` / `set_state` restore the reading position.
- The model accepts both configurations (CPU and GPU agree); a short lifetime runs on each.
- `eval_predict.py`'s nats per letter equals a direct computation with the reference model on hand-built windows.
- The classical GPT's loss at initialisation is ln 65 = 4.17.

## Ideas for later

- **Letters drawn as glyphs:** an 8 × 8 image of each letter (a bitmap font), one patch per letter: text as real
  vision, the same as reading a page. Then the model must learn the letters' shapes as well.
- If A − B is large, the fix belongs in the model for every world: values of a token predicted with some dependence on
  each other (e.g. a small autoregressive head within a token, or smaller tokens), not a text-specific change.

## Results (vm-00026)

- Val nats per letter, best over 100k updates: classical GPT 1.615–1.617 (at 20–30k, then it overfits to 1.80–1.92);
  universal B 1.796–1.804 (still falling or flat at 100k); universal A 3.37 (the unigram entropy is 3.31).
- B's gap (0.18) is most likely the linear input map (letters on one line in the embedding space); its output is
  already a softmax over letters. A cannot read its context at all (flat loss over the token's positions).
- Not run: the universal model trained offline (after one pass the lifetime's memory holds the whole corpus, so its
  replay is close to offline sampling).
