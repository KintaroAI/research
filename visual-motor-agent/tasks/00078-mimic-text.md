# Task 00078: Saying It Out Loud, in Text (experiment vm-00068)

**Date:** 2026-11-01
**Status:** Complete

## Context

- The owner (2026-11-01): natural intelligence never has the same input and output channel, and speech is not trained as "repeat the next word". One part of the brain teaches another, and we can try that first with text: half 1 learns to predict the next letter from what it hears, half 2 learns from the same input to give half 1's prediction, and then letters start to go missing.
- **There are no gaps while the model speaks** (the owner): what half 2 says goes into the stream and restores the missing letter. It says it out loud and hears itself.
- **Learning slows on what is already predicted** (the owner), to limit the damage from learning its own voice. Later the model should choose what to memorize by itself.
- The scrambled keyboard, where the output is a different language from the input, is a **separate exercise** (mumbling and muscle-to-action discovery), not part of this task.

## Design

A prototype in PyTorch, `dev/reference/mimic.py`, one letter per tick on Tiny Shakespeare, as vm-00038 was: the design may change, and the universal model comes after a result. Two separate causal character transformers of the same size, no shared weights: 128 wide, 4 layers, 4 heads, 64 letters of context.

- **Half 1** predicts the next letter it hears. Nothing else teaches it.
- **Half 2** reads the same input and is trained only against half 1's distribution (soft cross-entropy, half 1 detached). It never sees a target from the world.
- **Speaking:** half 2 samples a letter on every tick. The world masks it: where the real letter comes, the real letter is heard; where it is missing, half 2's letter is heard. So it speaks along, and is heard only when the world stops. It cannot know a gap is coming, and it does not have to.
- **A lifetime:** 256 readers of the train split, one update per tick, 64 windows from all the letters heard so far, 12k ticks (3M letters heard). The first 4k have no gaps, then the chance that a word is omitted rises to `q` over 2k ticks and stays there. A trial with 32 readers overfitted its small early memory (128 updates per new letter; val CE rose to 3.8), so this is many readers and one model, not one body: a prototype's shortcut.

| arm | what happens at a missing word | half 1 after 4k ticks |
|---|---|---|
| **Z** | no gaps at all | learns |
| **N** | half 2 is silent, half 1 hears a blank | learns |
| **A** | half 2 speaks | frozen |
| **B** | half 2 speaks | learns from all it hears |
| **S** | half 2 speaks | learns, each letter weighted by min(1, CE / 2 nats): what it predicts well teaches it less |
| **E** | half 2 speaks | learns from the world's letters only (a control: it is told which letters were its own) |

Seeds 1–3, `q = 0.3`. Then A, B, S, E at `q = 0.8`, where most of what it hears is itself.

## Measures

On the held-out 10% of the text:

- **Drift:** half 1's clean one-step CE and half 2's, every 1k ticks.
- **Filling:** letters right within gaps of 1, 2, 4, 8 and 16 letters, with half 2 speaking (greedy and sampled); whole words right. Top: half 1 speaking for itself, and half 1 with the true context. Bottom: the commonest letter.
- **Repair:** half 1's CE on the 16 real letters after a gap of 8, with the true letters in its context, with half 2's, and with blanks.
- A sample of filled text, to read.

## The readings, set before the results

1. **Mimicry works** if half 2's whole-word accuracy in arm A is at least 90% of half 1's speaking for itself.
2. **Speaking repairs** if the seed-by-seed interval of half 1's CE after a gap, blanks minus fills, is above zero, in arm A.
3. **Learning its own voice does no damage** if half 1's clean CE at the end of B, minus Z's, is under 0.05 nats. If it is more, S and E are read against B to see whether slowing helps and how much of the known limit it reaches.

## Verification

`dev/tests/test_mimic.py`: half 2 receives no gradient from the world, half 1 none from half 2's loss; speech is heard only in gaps; arm E and the weights of S act only on half 1; a frozen half 1 does not change; omitted words start at word starts.

## Results

Three seeds, 12k ticks. Full results in `../00068-mimic-text/README.md`.

- **Reading 1, met:** half 2's whole-word accuracy in A is 111% of half 1's (0.069 against 0.062). Half 2 also beats its teacher on held-out text in every arm, by 0.05–0.30 nats.
- **Reading 2, met narrowly:** blanks minus fills in A, +0.049 ± 0.042.
- **Reading 3, met, but its reference was badly chosen:** Z overfits (held-out CE 1.719 at 3k ticks, 1.824 at 12k). Against the frozen A, B is level: −0.059 ± 0.071 at `q = 0.3`, +0.003 ± 0.031 at `q = 0.8`.
- **Not planned:** S's weights ran from tick 0, and they stop the overfitting (flat at about 1.68). Half 1 that hears itself loses much less after a gap (B 0.74, S 0.58, A 0.87 nats). Ignoring its own letters (E) is worse than learning them.
