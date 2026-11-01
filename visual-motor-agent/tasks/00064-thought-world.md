# Task 00064: A Thought World Beside the Real One (experiment vm-00056)

**Date:** 2026-10-31
**Status:** Complete

## Context

- The window is 8 ticks: nothing the model holds outlives it. The owner's idea: a second half of the view, a grey "thought world",
  whose next frame is the model's own prediction of it, kept alive by waves when it goes quiet. It should come to associate
  states of thought with states of the world, and work as a state that outlives the window.
- The owner's choices (2026-10-31): one wide image (option A), a pyramid with two transformers, each teaching its own half's
  tokens; both predict the move; as a real world split in two, except that the thought half's prediction becomes its reality.

## Design

### A wide frame

- **`model.img_w`** (0 = square): a frame is `img` high and `img_w` wide, a patch grid gh × gw. All of `vm_uni_layout`,
  `vm_uni_block`, `vm_uni_center`, the slot table, the pieces, C++ and CUDA values, uploads and the reference follow it.
- 128 × 64 with patch 8: a 16 × 8 grid. `uni_pyr = 3`: B1 8 × 4, B2 4 × 2, B3 2 × 1. Every block lies inside one half, so no
  summary mixes world and thought. Two pieces: `band_n = 2`, `band_cols = 1`: left the world, right thought.
  A piece: 64 patches, 42 summaries, body, move: 108 tokens a step.

### The thought world

- **`ThoughtWorld`** (`world/thought_world.py`, `env.thought`): wraps a world; its frame is [world | thought]; moves, hunger and touches
  are the wrapped world's. Thought starts as random grey values.
- **Next thought:** `dream` (from the right transformer: its prediction of its own patches, decoded as the dream is, pair decoding,
  Jacobi iterations) + the waves. Without a dream, thought stays where it is.
- **Waves:** when thought goes quiet (mean absolute change between frames under `quiet`), with a probability per tick, a wave starts
  at a random point. A wave is a ring spreading at a fixed speed: at distance r at time t, a value shifts by A · f(r − c t), a bump up
  then down with little or no net effect, fading with distance. Overlapping waves add. Small random areas can change too.
- The model is unchanged: thought is image. Targets: each half's next frame, as for any world, so for thought its own dream
  with the waves. Memory stores the whole frame.
- **Acting:** both transformers propose a move; gain makes it.

### The catch

The target for a thought is largely the model's own output. Learning comes from waves (novelty), replay (old dreams differ from
today's) and self-reinforcement of any world dependence. No gradient tells it what to write, as dreams are data. If that is too weak: a
step of backpropagation through a dream.

## Verification

- Wide frames: square frames unchanged, bit for bit; layout, thumbnails, position codes, C++ and GPU losses and gradients against
  the reference at 16 × 8.
- `ThoughtWorld`: the world's half and its hunger equal the wrapped world's; without dream or waves thought does not change; a dream
  becomes the next frame; waves: shape, overlap adds, net shift small, only when quiet.

## Experiment vm-00056

- A lifetime with two pieces, gain, the no-overlap world beside thought, against a control whose thought half is frozen noise.
- **Measure:** food and poison; world prediction; is thought alive (change per tick, share of ticks with waves); does thought hold
  information (decode the food's place from thought alone, also after it left the window; shuffling thought in a window, what
  breaks).
- **Later:** a delayed-cue world, the real test of a memory: a mark flashes and is gone for 20-30 ticks, longer than the window.

## Write-up

[`../00056-thought-world/README.md`](../00056-thought-world/README.md). T 33.1 food, 3.1 poison; TL 25.1, 3.5; TF 21.7, 4.0; TN 15.7, 10.7 (stopped at 41k). Thought collapses to black and white shapes, and holds no linearly readable food position.
