# Task 00045: Two Stacked Boxes, Backprop Cut Between Them (experiment vm-00038)

**Date:** 2026-10-27
**Status:** Complete (results in vm-00038)

## Context

- The owner's question: think of the universal model as a box with x inputs and y outputs (x = y). Stack two: input →
  bottom → top → output, compared with the real output. If backprop is cut between the boxes, can the stack learn?
  The hidden signal at the cut has no target of its own; the goal is to pass information through the cut that gives
  the bottom one.
- Known families: a local loss of the bottom's own (greedy layer-wise; Greedy InfoMax); target propagation with a
  learned inverse (Bengio 2014; difference target propagation, Lee et al. 2015): the top learns an inverse of itself,
  and the bottom's target is "the input the top would have needed to produce the right answer"; synthetic gradients
  (Jaderberg et al. 2016): a predictor of the gradient at the cut, trained on the top's true input gradients.
- A PyTorch prototype first (the owner's choice), in the text world: fast, and vm-00028/00033's classical GPT is the
  reference.

## Design (`reference/stacked_local.py`)

- **Data:** Tiny Shakespeare letters (`T_text_B.json`'s train split, 65 letters), random windows of 512 letters (the
  universal model's 8 pages), 8 windows per update; scored on eval_predict's val windows (char_gpt.py's
  `val_windows`, `evaluate`), nats per letter.
- **A box:** x = y = V = 65 numbers per position. In: `W_in` (d × V) + a learned position; the universal model's
  Transformer (`torch_reference.transformer`: d = 128, 2 layers, 4 heads, d_ff = 512), causal; out: `W_out` (V × d).
  The bottom's input is the letters one-hot (so `W_in` is GPT's embedding table); its output h (V per position) is the
  top's input; the top's output is the logits of the next letter.
- **Five ways of training** (AdamW as char_gpt.py: lr 2.5e-4, gradient clipping 0.5, per module):
  1. **E2E:** cross-entropy of the top's output, backprop through both boxes (the ceiling).
  2. **Frozen:** the bottom stays at its random initialization; the top trains on h (the floor: random features).
  3. **Local:** the bottom trains on its own next-letter cross-entropy (h read as logits); the top on h, detached.
  4. **DTP** (difference target propagation): the top trains on h, detached. An inverse box g (V → V, causal) learns
     to undo the top: h̃ = h + σ·noise, loss |g(top(h̃)) − h̃|² (the top's weights untouched). The output target
     t = ŷ − β (softmax(ŷ) − onehot(y)) (the logits nudged toward the right letter: the output's own gradient, local
     to the output). The bottom's target h* = h + g(t) − g(ŷ) (the difference corrects g's error), its loss
     |h − h*|². No gradient crosses the cut; the information that does is the target h*.
  5. **Synthetic gradients** (conditioned, as cDNI): a box M (input [h, onehot(next letter)], 2V → V, attention over
     the whole window, since a position's gradient depends on later positions) predicts ∂L_top / ∂h; the bottom is
     updated with the prediction (loss Σ h · M(·), M's output detached); M trains on the true gradient at the cut
     (the top's own backward; scaled by the number of positions to order 1).
- **Records:** the top's val nats per letter every 2,500 updates; Local: the bottom's own val nats too (a single box);
  DTP: the inverse's reconstruction error and the cosine between the bottom's TP step (h* − h) and the true gradient
  direction −∂L/∂h (whether the target points the right way); synthetic gradients: the cosine between predicted and
  true gradients.
- **The bottom's learning rate in DTP and DNI: 0.1 × lr** (`--bottom-lr-scale`). AdamW makes every step about lr in
  size, so while the inverse or the predictor is untrained its noise moves the bottom at full speed and the top chases
  a drifting input: on a toy sequence neither arm learned at scale 1 (DTP's h grew, its reconstruction error 0.5 →
  241); at 0.1 DTP learns and its target step points along the true descent direction (cosine 0.29–0.48).
- 30k updates, seeds 1–3, each arm; β = 1, σ = 0.1 (first settings; DTP is sensitive to both).

## Verification (`tests/test_stacked_local.py`)

- In every cut arm the top's loss gives the bottom zero gradient; in E2E it does not.
- DTP: with a linear top and its exact inverse, the difference target moves the top's output exactly to t.
- Synthetic gradients: M's training target equals autograd's gradient of the top's loss at h (times the scale).
- Each arm except DNI learns a toy sequence (the top's loss falls); DNI's predictor, with both boxes held fixed, comes
  to point along the true gradient (cosine > 0.8). On the toy the top soon reaches zero loss, so DNI's own learning
  is no test of M.

## Experiment vm-00038

The five arms, 3 seeds; the top's best and final val nats per letter; DTP and DNI diagnostics over training. Expected:
E2E best, Frozen worst, Local close to a single box (the top refines the bottom's prediction); the question is
whether DTP (and DNI) close the gap between Local / Frozen and E2E, i.e. whether the target that crosses the cut
teaches the bottom something its own loss does not.

## Additions during the experiment

- `--tie` (the owner's idea): the bottom's `W_out` and the top's `W_in` one matrix (a module of its own, learning from
  every loss that reaches it), all five arms.
- `--cut-norm`: h layer-normalized at the cut, for dtp after it diverged (β = 1 and 0.1).
- `reference/stacked_summary.py`: the tables.

## Results (vm-00038)

- Best val nats per letter, mean of 3: e2e 1.550, local 1.593, frozen 1.623, dni 1.636, dtp 2.082 (diverged; with h
  normalized 1.764–1.780). The bottom's own loss recovers ~40% of the gap from a random bottom to end to end.
- DTP's targets carry no descent direction (cosine with the true gradient 0.000 by 10k unbounded, ≤ 0.006 bounded):
  its inverse cannot invert a transformer box (it explains 10–30% of h). DNI's predicted gradient: cosine ~0.28.
- Tying the cut: no gain (frozen 1.624), slightly worse elsewhere (local 1.604, e2e 1.559, dni 1.647).

