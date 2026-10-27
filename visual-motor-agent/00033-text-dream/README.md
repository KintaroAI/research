# vm-00033: Predicting the Future in the Text World

**Date:** 2026-10-27
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00033`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00033/`
**Task plan:** [`../tasks/00040-text-dream.md`](../tasks/00040-text-dream.md)

## Goal

In the circle world the model's dream (its own predictions fed back) drifts within a few steps (vm-00031's videos,
vm-00032). In the text world, dreaming is text generation, and a classical character-level GPT does the same thing.
vm-00030 showed the universal model with bin embeddings predicts text as well as a GPT given the same data
(teacher-forced); does it also generate as well?

## Method

- **Models, each at its best on the same data:** the universal model of vm-00030 (text arm B with bin embeddings,
  `T_text_B.json`, seed 1) at 30k ticks of its lifetime (val 1.718 nats per letter); the classical GPT
  (`reference/char_gpt.py --data lifetime`, the same lifetime's batches, seed 1) trained for 30k updates (val 1.716).
- **Generation** (`reference/text_dream.py`): the universal model generates as it dreams: its window is its last 8
  pages; each new page's letters come one token at a time, each from a forward pass over the window so far (exactly
  sequential); a new page's body is the move token's prediction, each step's move is sampled from its own prediction
  (as in its lifetime). A letter's probability: each of the token's 4 values (a 2x2 glyph) predicts the same letter;
  the mean of their log-probabilities, normalized over the 65 letters. GPT generates letter by letter from its last
  512 letters. The forward pass is the PyTorch reference's on the GPU (checked against `torch_reference.forward_uni`,
  max difference 1e-4).
- **Contexts:** 64 val windows (eval_predict's), 8 real pages (512 letters) each, then 4 pages (256 letters).
- **Measures:** greedy continuations against the real text by position; continuations sampled at temperature 1,
  scored teacher-forced by both models (nats per letter), against the real continuation.

## Log

### 2026-10-27
- **Trained** the GPT (30k updates, 4 min); generation and scoring 30 s.
- **Fixed before the runs:** the first version multiplied the 4 values' probabilities (eval_predict's "glyph
  probability"), i.e. a temperature of 1/4: its own samples scored 0.5 nats per letter and the real text 5.4. With the
  mean of the log-probabilities the real text scores 1.67–1.70, as the val cross-entropy. (eval_predict's `nll_glyph`
  has the same multiplication for glyphs of equal values; no write-up used it for arm B.)

## Results

### R1. Predicting the real continuation (teacher-forced, nats per letter)

| | universal model | GPT |
|---|---|---|
| the real 256 letters | **1.668** | 1.683 |

### R2. Greedy continuations against the real text (share of letters right)

| letters after the context | 1 | 2–8 | 9–64 | 65–256 | first error (median position) |
|---|---|---|---|---|---|
| universal model | 0.47 | 0.13 | 0.07 | 0.06 | 0 |
| GPT | 0.52 | 0.16 | 0.06 | 0.06 | 1 |

### R3. Samples (temperature 1) judged by both models (nats per letter)

| text | under GPT | under the universal model |
|---|---|---|
| the real continuation | 1.683 | 1.668 |
| the universal model's samples | **1.673** | 1.251 (its own) |
| GPT's samples | 1.426 (its own) | 1.779 |

### R4. Text (one context of 64, continued 200 letters)

Context: *"…Was it not to refresh the mind of man / After his studi"*; real: *"es or his usual pain? / Then give me
leave to read philosophy, / And while I pause, serve in your harmony."*

- Universal, greedy: *"ed the seasonable and and him / That shall but a man the state and and him / That with the command
  the command the company"*
- GPT, greedy: *"tion the duke and the duke and / The duke and the duke and the duke and the duke and"*
- Universal, sampled: *"er for me, he stoop of deaths, / Master, these would curcons's death to meet, / Which is do five him
  too son, and despair."*
- GPT, sampled: *"ce they think Richarge: / To have we doubt that re-dange then thou, by condition / Most in time gone"*

## Analysis

- **The universal model generates as well as the classical GPT** given the same data. Teacher-forced it predicts the
  real continuations as well (1.668 against 1.683). Its samples are as likely as real text to the GPT (1.673 against
  1.683), while GPT's samples are less likely than real text to the universal model (1.779 against 1.668).
- **Greedy continuation leaves the real text within a letter or two for both** (the first letter right about half
  the time, then 13–16%, then chance-like 6%): text is not determined by its past, so a single "true future" is not
  the right target for a text dream, unlike the cursor's path. Both greedy decoders fall into the classic loop ("the
  duke and the duke and").
- **The universal model's own samples score very low under itself** (1.25 against GPT's 1.43 under GPT): it samples
  sharper than its own predictions of real text (a gap of 0.42 nats against GPT's 0.26). The letter probability
  (normalized over the 65 letters from 256 bins per value, averaged over the token's 4 values) may be slightly
  overconfident.

## Conclusions

1. In the text world the universal model's dream is text generation of the classical GPT's quality, with no
   text-specific part: the dream problem of the circle world is not a weakness of the universal design itself.
2. For text the dream cannot be scored against one true future; samples judged by an independent model are the measure.

## Next Steps

- Self-forcing in the text world (task 00041, vm-00034): the same rule as in the circle world, where the future is
  mostly determined; in text it is not, so it may teach the model to distrust its context.
