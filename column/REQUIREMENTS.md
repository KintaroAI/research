## TL;DR

A small unsupervised competitive cell that maps a low-dimensional input vector to a low-dimensional probability output, usually with one clear winner, learns from the input stream itself without backpropagation, forms meaningful categories, avoids collapsing into the same winner all the time, stays stable when the input distribution is stationary, and adapts when the input distribution changes, i.e. **small self-organizing competitive categorization cell** with:

* probability outputs
* one dominant winner
* no backprop
* unsupervised learning
* anti-collapse behavior
* stability under stationary input
* adaptation under distribution shift

---

## Core functional requirements

Cell should:

1. Take **small input vector**

   * `n` inputs
   * typically around **10–20**
   * **Optional: temporal context.** Input can be a single vector of `n` scalars, or an `(n, T)` matrix where each input carries a temporal trace of `T` recent values. When temporal context is provided, the cell can use it to compute richer similarity measures (e.g., MSE, correlation) between inputs rather than relying on instantaneous values alone. This enables the cell to distinguish "both silent" from "both co-varying" and to detect structure that only emerges over time.

2. Produce **small output vector**

   * `m` outputs
   * typically around **4–20**
   * `m` may be less than, equal to, or greater than `n`

3. Output **probability-like activations**

   * outputs might (but not nessesarily should) sum to something interpretable like probabilities
   * usually there should be **one clear winner**
   * other outputs should remain lower, not equal noise

---

## Learning requirements

The cell should:

4. Learn **without backpropagation**

   * no gradient descent through deep network
   * prefer local update rules
   * ideally Hebbian / competitive / self-organizing style learning

5. Learn **from the input stream itself**

   * no explicit labels required
   * should (ideally, but optionally) discover categories or structure from data
   * unsupervised or self-organizing behavior

6. Form **meaningful categories**

   * repeated similar inputs should activate similar outputs
   * different input patterns should separate into different outputs/classes

---

## Competition / output behavior requirements

The cell should:

7. Prefer **sparse or competitive output**

   * not all outputs active equally
   * ideally one dominant output for a given input
   * but still keep graded probabilities, not only hard argmax

8. Avoid **winner collapse**

   * should not always pick the same output regardless of input
   * should return evenly low probability on all outputs if fed with random white noise
   * should avoid one unit monopolizing activity
   * should keep multiple output units alive and useful

---

## Stability / adaptation requirements

The cell should:

9. Be **stable when the input distribution is stable** - locking behavior

   * if the same kinds of patterns keep arriving, learned categories should settle
   * should not drift endlessly for no reason

10. **Adapt when the input distribution changes**

* if patterns change over time, the cell should update accordingly
* should remain plastic enough to learn new structure

11. Balance **stability and plasticity**

* no catastrophic freezing
* no chaotic forgetting
* should preserve useful learned structure while still adapting

---

## Implicit architectural preferences

Additionally:

12. **Small, self-contained module**

* something usable as a building block in a larger architecture

13. **Interpretable internal state**

* ideally each output unit corresponds to some learned prototype/category

14. **Online / streaming learning**

* unsupervised update per example or small batch
* no need for large offline training loop

15. **Biologically plausible-ish learning**

* local competition
* local updates
* no global error propagation

