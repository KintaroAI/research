# vm-00093: A State Carried Across Windows

**Date:** 2026-11-16
**Status:** Running
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00093/`
**Task plan:** [`../tasks/00104-state-across-windows.md`](../tasks/00104-state-across-windows.md)

## Goal

In vm-00091 the value agent remembered food that left its 64 × 64 window only for its 8-tick context. In vm-00092 a context of 32 ticks did not help: the frames of the lost food were in its input, and it did not use them. Here the agent keeps its 8-tick context and gets a state it writes and reads back: a note of 64 numbers written every tick and read 8 ticks later. The model is trained through that link, so what it keeps is learned. Does it then aim at food it saw 9–16 ticks ago, and does that pay? The design and the readings set before the results are in the task plan.

## Method

| Arm | Label | Model | State | Seeds |
|---|---|---|---|---|
| C8 = vm-00091's VW (not rerun) | value, γ 0.9, n 4, fresh tail | 4 quarters × 8 layers, tolerance 0.002, context 8 | none | 1–12 |
| SA | the same | the same | read and write tokens, the carry detached | 1–12 |
| SB | the same | the same | read and write tokens, the gradient through the carry | 1–12 |

- **The state** (`model.state = 1`):
  - Each step of each quarter transformer gets two more tokens after the body (32 → 34): a **write**, which has no values and no target, and a **read**.
  - The write's expected values, 64 numbers in [0, 1] from the shared head, are the state.
  - The read's values are the state written by the same transformer 8 ticks earlier.
- **Training on pairs of windows:**
  - Each update keeps C8's 8 windows, taken as 4 pairs of consecutive windows.
  - The first window reads the states stored while acting; the second reads the first's fresh writes, step k reading step k − 8.
  - SB passes the second window's loss back through those writes into the first window (`online.state_grad = 1`). SA reads them as given (`0`).
- **Acting:** each transformer keeps a queue of its last 8 writes, and the ring computes each new step's read and write. Memory stores each tick's reads, so a replayed first window starts from the state it had while acting.
- **Code checks** (`dev/tests/test_state.py`, 32 tests):
  - forward passes with given states and the GPU pair update, both arms, against the PyTorch reference;
  - the ring with its queue against full passes;
  - the C++ model with a blank state.
  - **A copy task:** a value shown only in the first window is asked for in the second. SB learns it (error 0.0001) and SA only partly (0.04, at chance 0.083): with the carry detached, the reader still learns to decode what the untrained writes happen to carry.
