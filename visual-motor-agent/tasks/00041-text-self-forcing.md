# Task 00041: Self-Forcing in the Text World (experiment vm-00034)

**Date:** 2026-10-27
**Status:** Complete (results in vm-00034)

## Context

- Self-forcing (task 00039, vm-00032) in the circle world: dreamed inputs, real targets; the dreamed cursor stays
  closer to the real one (4–5 px off after 16 dreamed steps against 11–19).
- In the text world the universal model with bin embeddings predicts and generates as well as a classical GPT given
  the same data (vm-00030, vm-00033).
- The same rule, unchanged, in the text world. Expectation (stated before the runs): it hurts. In the circle world the
  next frame is mostly determined by the world and the real moves, so a dreamed frame is an estimate of the real one and
  "predict the real next step from my own picture" is a consistent lesson. In text the future is not determined: a
  dreamed page is another plausible text, and the real next letters do not follow from it, so the lesson becomes
  "recent context does not predict what comes next" (scheduled sampling's known inconsistency for language models).
  If so, self-forcing needs a universal refinement, not a text exception.

## Runs

`configs/T_text_B.json` (one letter per token, bin embeddings), 100k ticks, seeds 1–3, with self-forcing (8 windows,
1–3 dreamed pages, J = 2, pair decoding), against vm-00030's B with bin embeddings (the same without).

## Measures

- Val nats per letter every 10k (`eval_predict.py --env-set split=val --env-set tile=true`), best and at 100k.
- Generation at each arm's best checkpoint (`reference/text_dream.py`, seed 1): greedy continuation against the truth,
  samples judged by the classical GPT (vm-00033's).

## Results (vm-00034)

- Prediction (the expectation was wrong): val nats per letter better at every checkpoint of every seed; best
  1.703–1.714 against 1.718–1.745 (now level with the classical GPT on the same data, 1.696–1.716); less overfitting
  at 100k (1.903–1.949 against 1.940–2.019). A denoising regularizer.
- Generation (the expectation was right): samples worse, 2.67 nats per letter under GPT at temperature 1 (baseline
  1.67), mostly a flatter distribution but not only: at matched sharpness still 0.06–0.09 worse; greedy first letter
  right 0.38 against 0.47. `text_dream.py --temp` added.
- Not a default: in text it trades generation for prediction, in the circle world behaviour did not improve.
