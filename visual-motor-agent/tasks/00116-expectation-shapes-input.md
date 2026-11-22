# Task 00116: Expectation Mixed into the Senses — Internal Reality as the Spark of Thought and Action (to explore later)

**Date:** 2026-11-22
**Status:** Proposed, for later (the owner, 2026-11-22, during the vm-00102 write-up: "we can predict reality well with transformers but it's hard to act on it. the idea is that higher areas of the brain mixing with sensory input and form internal reality in some sense - brain better resonse to the signals from outer world which are expected by higher areas. And that can be the way to spark thought proccess and ability to not only predict reality better but also form it")

No experiment number and no design yet: this records the idea and the forms it could take here.

## The idea

- **The gap:** transformers predict reality well. Acting on the prediction is hard: our agent predicts the next frame well, but the instinct only picks among 16 moves by a learned label, and the forecasts reach 1–2 ticks.
- **The brain's way, as the owner puts it:**
  - higher areas send their expectation down, and it mixes with the sensory input;
  - together they form an internal reality, not the raw input;
  - the senses respond more strongly to the signals the higher areas expect.
- **What it could give:**
  - **prediction:** the internal reality is cleaner than the input where the expectation is right;
  - **thought:** an internal reality that runs partly on its own expectations can move ahead of the input, which is the start of a thought process;
  - **acting:** an expected reality can be a target. Acting to make the input match it is acting to form reality, not just to predict it.

## What this agent already has

- **vm-00102's forecast panels** are a first, side-by-side form:
  - the model sees its own forecast of this moment (panel 2) beside the real frame (panel 1);
  - it reads them: blanked, every panel's prediction is worse, panel 1's included, and it loses three quarters of its food;
  - but the expectation sits beside the input, not mixed into it, and nothing makes the senses respond more to what is expected.
- **The thought world** (vm-00056, vm-00057: `env.thought`): a part of the frame the model writes and reads back.
- **Dreaming** (vm-00023, self-forcing, task 00039): the model run on its own outputs.
- **Labels from imagined choices** (task 00114): outcomes of moves not made.

## Forms it could take

1. **Mixed, not beside:**
   - each sensed patch enters as a blend of the real patch and the model's prediction of it from the tick before, the weight learned or set per patch;
   - the frame stays 64 × 64 (unlike the panels' 128 × 128, which cost vm-00102 its foraging against TT).
2. **Expected signals amplified (gain):**
   - a patch the model predicted well enters with more weight, a surprising one with less (or the other way round; see the next section);
   - the gain is the higher areas' say over what the senses pass on.
3. **Expected signals cancelled (predictive coding):**
   - the input is the error, real minus expected, and the model carries the rest itself;
   - the opposite choice to 2. Both happen in cortex, so both are arms to compare.
4. **Forming reality, acting toward an expectation:**
   - the higher layers hold a desired internal frame (for example food under the cursor);
   - the move chosen is the one whose predicted frame comes nearest to it, instead of, or besides, the label;
   - this is acting by making the expected come true.
5. **A thought loop:**
   - between real ticks, the model runs a few steps on its own mixed internal reality (input weight turned down);
   - it acts on where that loop ends, not only on the frame now.

## Related work

- **Predictive coding:** Rao & Ballard 1999 (feedback carries predictions, feed-forward the errors); Friston 2005.
- **Expectation sharpens what is expected:** Kok, Jehee & de Lange 2012, "Less is more: expectation sharpens representations in the primary visual cortex". The weaker response to expected input is the more selective one, which fits form 2 and form 3 together.
- **Active inference:** Friston et al. (acting to make sensory input match predictions), the closest to forming reality (form 4).
- **Top-down feedback in deep networks:** feedback or recurrent connections that bias early layers (for example Spoerer et al. 2017), and attention as gain on expected features.
- **World models that act by imagined rollouts:** Dreamer (Hafner et al.), and planning to a goal frame (visual foresight, Finn & Levine 2017), for form 4.

## Open questions, for the design

- Which layer is "higher": the last layers of the same transformer, the summaries, or another transformer?
- What is mixed: pixels (the input patches) or the token embeddings?
- Where the desired internal reality comes from in form 4: the label (moves that lead to meals), memory of meals, or a learned goal.
- **How to keep it honest:** the model must not learn to see only what it expects. The targets stay the real frames, as in vm-00102, so a blend that hides reality costs prediction loss.
- **The measures:**
  - frozen prediction against the plain agent (TT, `configs/U_agent.json`);
  - foraging;
  - whether the loop between ticks moves its internal frame toward food before the body gets there.
