# vm-00068: Saying It Out Loud, in Text

**Date:** 2026-11-01
**Status:** Complete (pending tag)
**Source:** `exp/vm-00068` (code: `visual-motor-agent/dev/reference/mimic.py`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00068/`
**Task plan:** [`../tasks/00078-mimic-text.md`](../tasks/00078-mimic-text.md)

## Goal

In a language model the input and the output are one channel. Natural intelligence learns to predict by ear and speaks through a separate part, which nobody trains to repeat the next word. The first step, in text: can one half of a model teach another, and can the second say what is missing, so that the model hears itself?

## Method

A PyTorch prototype, one letter per tick on Tiny Shakespeare. Two separate causal character transformers (128 wide, 4 layers, 4 heads, 64 letters of context). Half 1 predicts the next letter it hears. Half 2 reads the same input and is trained only against half 1's distribution: no target from the world reaches it. Half 2 samples a letter every tick, and is heard only where a word is missing, so there are no blanks while it speaks.

256 readers feed one model, one update per tick, 64 windows from all it has heard, 12k ticks (3M letters). The first 4k ticks have no gaps; the chance of omitting a word then rises to `q` over 2k ticks. Seeds 1–3, so intervals are 95% with df 2 (t = 4.303).

| arm | at a missing word | half 1 |
|---|---|---|
| **Z** | no gaps | learns |
| **N** | half 2 silent, half 1 hears a blank | learns |
| **A** | half 2 speaks | frozen at 4k ticks |
| **B** | half 2 speaks | learns from all it hears |
| **S** | half 2 speaks | a letter weighted by min(1, CE / 2 nats) |
| **E** | half 2 speaks | learns only the world's letters (it is told which are its own) |

Departures from the plan: **S's weights ran from tick 0**, not from 4k, so its warm-up differs from the other arms'. At `q = 0.8` the share of what it hears that is its own is 0.38, not 0.8, as spaces and short words take many ticks.

## Results

### R1. The readings set in advance

| reading | result | |
|---|---|---|
| 1. Half 2's whole-word accuracy in A at least 90% of half 1 speaking for itself | 0.069 against 0.062, 111% | **met** |
| 2. In A, half 1's CE after a gap: blanks minus fills above zero | +0.049 ± 0.042 | **met, narrowly** (+0.040 ± 0.042 at `q = 0.8`) |
| 3. B's clean CE minus Z's under 0.05 nats | −0.158 ± 0.073 | met, **but the reference is faulty: see R3** |

### R2. Held-out CE at the end, and repair after a gap of 8 letters

| arm | q | half 1 CE | half 2 CE | half 1 after true letters | after half 2's fill | after blanks |
|---|---|---|---|---|---|---|
| Z | 0 | 1.824 | **1.529** | 1.825 | 2.948 | 2.992 |
| N | 0.3 | 1.795 | 1.548 | 1.805 | 2.891 | 2.646 |
| A | 0.3 | 1.724 | 1.599 | 1.722 | 2.588 | 2.637 |
| B | 0.3 | **1.665** | 1.547 | 1.658 | 2.402 | 2.602 |
| S | 0.3 | 1.674 | 1.622 | 1.657 | **2.235** | 2.325 |
| E | 0.3 | 1.717 | 1.548 | 1.716 | 2.517 | 2.688 |
| A | 0.8 | 1.724 | 1.600 | 1.722 | 2.597 | 2.637 |
| B | 0.8 | 1.727 | 1.573 | 1.725 | 2.528 | 2.598 |
| S | 0.8 | 1.703 | 1.628 | 1.677 | 2.305 | 2.354 |
| E | 0.8 | 1.786 | 1.571 | 1.785 | 2.656 | 2.706 |

Letters right in gaps, greedy: half 2 0.54–0.57 at one letter, 0.21–0.22 at 8; half 1 for itself 0.52–0.55 and 0.19–0.21; half 1 with the true context about 0.53; the commonest letter 0.15. Whole words right: 0.065–0.082 for half 2.

### R3. Half 1 overfits, which changes what the comparisons mean

Arm Z's held-out CE reaches 1.719 at 3k ticks and climbs to 1.824 at 12k, while its training CE falls to 1.065. So reading 3 compares against a run that is getting worse. **The fair reference is A, half 1 frozen near its best (1.724):**

| difference in half 1's clean CE | |
|---|---|
| B − A, `q = 0.3` | −0.059 ± 0.071 |
| B − A, `q = 0.8` | +0.003 ± 0.031 |
| S − A, `q = 0.3` | −0.050 ± 0.097 |
| E − A, `q = 0.8` | **+0.062 ± 0.027** |
| B − E, `q = 0.3` | **−0.051 ± 0.019** |
| B − E, `q = 0.8` | **−0.059 ± 0.006** |

S's CE stays flat from 4k ticks on (1.680 to 1.674), and is already at 1.680 at 4k ticks when the others are at 1.724.

### R4. Half 2 beats its teacher on text neither was tested on

Half 2 minus half 1, held-out CE: Z −0.295 ± 0.098; A −0.125 ± 0.022; B `q = 0.3` −0.118 ± 0.017; B `q = 0.8` −0.154 ± 0.047.

### R5. Repair after hearing itself

These were chosen after seeing the results. Half 1's CE after half 2's fill: B − A −0.187 ± 0.033; S − B −0.167 ± 0.045 (`q = 0.8`: −0.069 ± 0.024 and −0.223 ± 0.058). Each arm as it lived, its fill against N's blanks: B −0.244 ± 0.206; S −0.411 ± 0.168.

### R6. What it says

Half 2 sampling at word gaps in held-out text, marked [ ], arm A:

> else, you must pardon me, [I ] [fur] should die [faint ] him, [hear hi] her [shun: ]

## Analysis

- **Mimicry works, and the student generalizes better than its teacher.** Half 2 never sees a target from the world and predicts held-out text better than half 1 in every arm, by 0.05–0.30 nats (0.05–0.08 in S, whose half 1 overfits least). This is a known effect of distillation: soft targets carry more than one letter's worth of information per position. Here half 2 also follows a teacher that changes over time, which may act as an average over it; this experiment cannot separate the two. The largest gap is in Z because half 1 has overfitted and half 2 has not.
- **Hearing itself does no detectable harm in this length of life.** Against a frozen half 1, B is level at both rates. The fear was drift; instead, half 1 improves as its stream gets noisier: noise in the input seems to hold off overfitting. E shows that the letters themselves help too: told to ignore its own letters, half 1 is worse than B at both rates, by a clear margin.
- **That is not yet evidence about the long run.** 8k ticks after the gaps begin, with 38% at most of the stream its own, and with overfitting as the larger force, cannot rule out slow drift. B at `q = 0.8` does rise from 1.689 to 1.727 over its last 6k ticks.
- **Speaking repairs little at first, more once half 1 has lived with it.** With half 1 frozen, a fill saves 0.05 of the 0.87 lost after a gap. Half 1 that hears its own voice loses much less: B 0.74 and S 0.58 against A's 0.87. So the repair is in half 1 learning to cope with half 2's letters, not in the letters being right. In all arms much of the loss stays: 8 sampled letters are mostly wrong.
- **The slowing helps, in an unplanned way.** S stops overfitting, stays flat, and recovers best after gaps. But its weights ran from tick 0 and it weighs every letter, so this shows that weighting by error holds off overfitting, not that it guards against its own voice. Half 2 pays: S's is the worst student (1.622 against 1.547), perhaps because its teacher is less sure of itself.
- **Caveats:** three seeds; about 30 comparisons, those of R5 chosen after the results; 256 readers, not one body; half 2 outputs letters, so this is one channel and the separate-channels question stays open. A causal model cannot use the words after a gap, which limits filling.

## Conclusions

1. One part of the model can teach another with no target from the world, and the student generalizes better than the teacher.
2. A model that says missing words and hears itself takes no measurable harm in a short life, and copes far better with its own errors.
3. Teaching half 1 to weight letters by error stops overfitting in this setup. Whether it protects against its own voice is not yet shown.
4. Reading 3's reference was badly chosen; the frozen arm is the right one.
