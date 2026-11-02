# Task 00075: A Channel per Value (experiment vm-00066)

**Date:** 2026-11-01
**Status:** Complete (pending tag)

## Context

- The owner (2026-11-01): hunger and the move share `W_in`'s rows with the first pixels of a patch, and a value's meaning should not depend on that. The vector that chooses rows has a patch's length, but a body need not.
- Today `W_in` and `W_out` have V·K rows (64 × 32 = 2,048): value position j, bin k. Hunger, the second drive and dx, dy, click use positions 0–2 of the same table as pixels 0–2, and `b_out` too: one bias per bin has to fit a brightness and a move. Only the slot vector tells them apart. Unused values are zeros, so nothing is wasted: the fault is the mixing, not the size. `ARCHITECTURE.md` has named this cost since the universal model began.
- Separate tokens for hunger, dx, dy would add slots and attention cost for no new information, so they are not tried first.

## Design

`model.uni_chan = 1` (`in_bins` needed): a **channel** is one named value of what the world gives. The table gets VC = V + 5 channels: 0..V−1 the patch values (summaries' thumbnails too, as now), V, V+1 the body's two, V+2..V+4 the move's three. One rule for every token: a value's meaning is its channel. A world has as many as it declares; no token takes another path. At 32 bins: 2,208 rows against 2,048, 8% more rows in both tables, about 41k parameters per piece (4%). Old checkpoints load (`h[60]` = 0).

Value tensors stay V wide. Only the table's index changes:

| where | change |
|---|---|
| input | a token's used values go to its channel: `k_uni_twohot`, `k_uni_in`, `uni_twohot` |
| head | `W_out` gives VC·K logits; `k_uni_pick` takes each row's V into the usual [rows, V, K], from the channel of the token it predicts (the last token before the move: the move; the move token: the next body), and puts gradients back |
| bias | `k_uni_loss`, `seq_loss_uni` read the bias of that channel |

So all that reads predictions (loss, probabilities, `values`, `move_probs`, `next_body`, candidates, ring) stays as it is. `uni_chan = 0` runs no new kernel.

## Verification

`tests/test_value_channels.py`: table size; hunger reaches the model only through its channel, pixel 0 only through its own; C++ against the PyTorch reference (forward, loss, gradients), CUDA against C++, ring against window, candidates against a full pass with channels on; a checkpoint round trip; old checkpoints load; invalid settings. The full suite for `uni_chan = 0`.

## Experiment vm-00066

- **CH:** `configs/U_default.json`, `model.band_n=16`, `council`, `model.uni_chan=1`. Seeds 1–8, 60k ticks, with MPS, on the 4090 once vm-00065's seeds leave it. Control: C16 (vm-00064), not rerun.
- **Measures:** as task 00073, and above all body and move CE, where a mixed table should cost most, by token.

## The reading, set before the results

The channels are a cleanup, so they need no gain to stay: they **stay in the default design** if the interval of the food difference against C16 includes zero or is above it, and that of poison includes zero or is below it. Whether body CE or move CE falls beyond its interval says if the mixing cost anything.

## Results

Eight seeds, last 20k ticks, per 1k: CH 29.5 ± 6.2 food, 0.65 ± 0.17 poison; C16 34.7 ± 7.0, 0.62 ± 0.19. Food −5.2 ± 11.0, poison +0.03 ± 0.26, so **the reading is met and the channels stay in the design**. Body CE +0.001 ± 0.004 and move CE +0.004 ± 0.020: the mixing cost nothing measurable. `U_default.json` is not changed yet: they were tested at 16 pieces, and the controls would need a rerun. Full results in `../00066-value-channels/README.md`.
