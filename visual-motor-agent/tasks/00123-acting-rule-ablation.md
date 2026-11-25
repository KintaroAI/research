# Task 00123: The Acting Rule — Our Transformer with a Trained Actor (priority, later)

**Date:** 2026-11-24
**Status:** Planned, a priority for later. The owner, 2026-11-24: "we need to do 4 - so write this down as one of priorities but for later. for now we need to do #1 (0.997) and #2 - make both to be 3 lives, once we get this - we will start thinking about #4"

## Context

- **vm-00108:** in Breakout stage 1 DreamerV3 returns .99 at 50–60k against the default agent's .65; both stay near random until 25–30k, then Dreamer climbs to .9 within about 10k ticks and ours slowly.
- **vm-00110 (seeds 7–12 done when this was written):** with Dreamer's replayed steps per tick (`online.batch` 56) our agent starts much faster (.5 by 18–27k against 33–55k) but levels off at about .72–.75, about where BD ends by 60k. No seed reaches .9. The training budget explains the start, not Dreamer's ceiling.
- **What else differs:** the acting rule. Ours picks the best of 16 moves sampled from the model's own move prediction, scored by the predicted label (γ .95), and the move token imitates those picks. Dreamer trains an actor by the gradient of its critic's value through 15 imagined steps (γ .997).

## The idea (to be designed after vm-00111 and vm-00112)

- Our same transformer and world model, with a standard actor-critic on its tokens in place of the label-and-instinct choice: an actor head trained on the critic's value (on real windows, or through the model's own imagined steps as Dreamer does), the move sampled from it.
- It isolates the acting rule from the architecture, which Dreamer against ours does not (architecture, model size, replay ratio and acting rule all differ).
- Needs new code in the CUDA path (an actor loss, a critic target) and tests; a day or more before any run.

## Waits for

- vm-00111 (task 00124): whether a longer horizon (γ .997) changes what the label teaches.
- vm-00112 (task 00125): whether Dreamer's poison failure in the food world is its horizon.
