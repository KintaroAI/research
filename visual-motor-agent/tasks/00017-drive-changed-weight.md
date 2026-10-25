# Task 00017: Weighting Unusual Drive Changes, So Hazard Contact Gets Learned (experiment vm-00013)

**Date:** 2026-10-25
**Status:** Complete (results in vm-00013)

## Context

vm-00012: neither agent avoids hazards, even at 0.1 hunger per step inside one. Nothing in the network represents
hazard contact: a linear probe of "inside a hazard" scores R² ≤ 0.10 from v and z, the drives head predicts the same
next hunger inside and outside, and the planner's latent hunger model imagines no extra rise. A hazard step changes
hunger by 0.1, a hit by up to 1.0, so in squared error a hazard step weighs about a hundredth of a hit.

vm-00001 met the same problem in the frame: plain MSE ignored the 1%-of-pixels cursor until pixels that changed
between frames were weighted (`ppo.pred_changed_weight=10`). The same remedy for the drives: steps where hunger changes
by more than its steady rise count more.

## Design

- `ppo.drive_changed_weight` (w, default 0: unchanged) and `ppo.drive_change_eps` (ε, default 0.01). A token t is a
  *changed* step when |hunger(t+1) − hunger(t)| > ε (the steady rise is h_rate = 0.005 per step; a hazard step adds
  0.02 or 0.1; a hit removes up to 1.0). Only observed drives enter, as in `pred_changed_weight` only observed frames.
- The drives loss of a changed token is multiplied by 1 + w (both drives), and so is the latent hunger loss of a chain
  step whose action token j is a changed step (|next hunger(j) − hunger(j)| > ε).
- In the C++ model (`seq_loss`, `seq_dyn`), the PyTorch reference, and the CUDA model (`k_head_loss`, `k_dyn_loss`).
  PPOConfig gains the two fields at its end.

## Verification

- `tests/test_model_reference.py`: the gradients with `drive_changed_weight` against the reference (CNN model; with
  latent dynamics).
- `tests/test_cuda_model.py`: the GPU loss and gradients with it against the C++ model.
- Defaults unchanged: 300-step runs byte-identical before and after.

## Experiment vm-00013

- The planner agent (LTH10: continual torus, 2 hazards at 0.1 per step) and DMI (DTH10) with `drive_changed_weight=10`,
  seeds 1–3; the hazard share online, `analyze_hazards.py contact` and `hazard`; compared with vm-00012's runs 013–018.
- Question: once contact is represented, does the planner steer around hazards? Does the basic agent (which has no
  way to use a rise in hunger) stay at chance?
