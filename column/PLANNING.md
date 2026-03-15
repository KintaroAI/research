## Initial analysis

The requirements describe a **cortical minicolumn** — a competitive self-organizing module. It's essentially the core properties of a Kohonen/SOM-style unit but with sharper constraints (probability output, anti-collapse, stability/plasticity balance).

## What maps to existing literature

- **Classic SOM / competitive learning** covers requirements 1-8 well
- **Growing Neural Gas** handles adaptation (req 10) better than fixed-topology SOMs
- **Soft-WTA with softmax** covers "probability output with one winner" (req 3, 7) — not standard SOM but straightforward to add
- **Adaptive Resonance Theory (ART)** — Grossberg's answer to the stability-plasticity dilemma (req 9-11). Vigilance parameter controls when to create new categories vs absorb into existing ones. Biologically motivated, local learning, but finicky to tune.

## The hard part: stability vs plasticity (req 9+10+11)

This is the classic unsolved tension. Approaches worth considering:

- **ART-style vigilance** — match quality gate that decides "update existing prototype" vs "recruit new unit." Principled but introduces a sensitive hyperparameter.
- **Frequency-gated lr** — units that fire often reduce their learning rate (stable). Rare-firing units stay plastic. Similar to the KNN-driven adaptive params idea from thalamus-sorter ts-00015.
- **Match-based learning** — only update the winner if match quality exceeds a threshold. Poor matches trigger recruitment of a dormant unit instead.

## Temporal context connection

The optional temporal context (req 1: input as `(n, T)` matrix) is exactly what thalamus-sorter does with the derivative correlation buffer. The machinery for "distinguish both-silent from both-co-varying" already exists there. Could reuse the MSE/correlation/derivative-correlation similarity measures.

## Proposed first prototype

A **soft-WTA cell** with these components:

1. **Prototypes**: each of `m` output units has a prototype vector (like SOM centroids)
2. **Competition**: similarity via dot product or negative MSE → softmax → probability output with temperature control for winner sharpness
3. **Winner update**: winner's prototype moves toward input (Hebbian pull), losers decay slightly
4. **Anti-collapse via usage counter**: per-unit frequency tracker gates plasticity — frequently-winning units learn slower, keeping them stable and giving other units a chance
5. **Match threshold for stability/plasticity**: if the best match is below a threshold, either recruit a dormant unit or spread the update across top-K instead of concentrating on the winner. This naturally handles distribution shift — new patterns that don't match existing prototypes trigger exploration rather than forced assimilation

This is simple, local, biologically plausible, and naturally covers req 1-8. The match threshold + usage gating addresses 9-11.

## Future exploration: input-order invariance

When composing multiple cells (exp 00005–00007), the wiring operation between cells
determines what relationships are learnable. An open question: can the architecture
be made **input-order invariant** so that feeding (a, b) produces the same output as
(b, a)?

This matters for commutative relationships (sum, equality, proximity) where order
shouldn't affect the result. Current wiring operations have mixed behavior:
- Circular convolution: naturally commutative (a+b = b+a)
- Comparison statistics: naturally commutative (cos(a,b) = cos(b,a))
- Outer product: NOT commutative (a⊗b ≠ b⊗a)
- Concatenation: NOT commutative

A general approach: use **symmetric wiring** — operations that are invariant to
input order by construction. For two inputs, this means: `f(a,b) = f(b,a)`.
Candidates: element-wise max/min of cell outputs, sorted concatenation,
sum of outputs, or symmetric outer product `(a⊗b + b⊗a) / 2`.

For N>2 inputs, this generalizes to permutation-invariant aggregation (sum, max,
attention-weighted mean). Worth exploring if the cell is used as a building block
in larger architectures where input ordering shouldn't be a degree of freedom.
