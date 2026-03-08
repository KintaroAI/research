# Routed Prediction

A research project investigating whether thalamocortical principles — gated routing, predictive categorization, surprise-driven plasticity — produce better streaming world models than standard recurrent baselines.

---

## Memo: origins and motivation

### Source material

This plan is grounded in the "Natural Intelligence" framework described at
https://kintaroai.com/docs/natural-intelligence/

That article proposes that intelligence emerges from a **thalamus-cortex loop** architecture, built on two principles:

1. **Thalamus as signal-sorting hub.** The thalamus receives both raw sensory data and cortical feedback, sorts incoming signals by temporal correlation, and produces organized output for the cortex — all through activity-dependent self-organization rather than genetic pre-specification.

2. **Neocortex as self-organizing pattern detector.** Cortical columns (~0.5mm) specialize through lateral inhibition and homeostatic plasticity, competing for inputs without pre-programming. Experience shapes functionality.

The key mechanism is the **thalamus-cortex-thalamus loop**: cortex sends abstractions and predictions back to thalamus as integrated signals, which blend bottom-up sensation with top-down interpretation. Hierarchical processing (V1, V2, V3...) emerges as a statistical outcome of iterative thalamic sorting, not from a fixed blueprint. Intelligence, in this view, is prediction — learning temporal patterns to anticipate future events.

The article is compelling as neuroscience-grounded motivation but is a conceptual framework, not an engineering specification. The gap between "thalamus sorts by correlation" and a differentiable module with defined inputs/outputs is nontrivial. This research plan bridges that gap.

### What this plan is

An attempt to take the thalamocortical loop idea seriously by:

- decomposing it into **falsifiable functional modules** (routing bottleneck, modular predictive categorization, surprise modulation, multi-timescale memory)
- testing each module in isolation against **plain recurrent baselines** (GRU, LSTM)
- evaluating in a **streaming partially-observed world** where hidden causes, persistence, and surprise actually matter — not on static datasets
- applying strict **decision rules**: each module must demonstrate measurable gains or be dropped

The central research question is not "does this become AGI?" but:

> Does a recurrent routed predictive system learn more stable hidden causes, better cross-modal integration, and better adaptation to surprise than simpler baselines?

### Assessment and risks

**Strengths of this plan:**
- Modular structure with keep/revise/drop criteria prevents the common failure mode of building a sprawling brain-inspired system without knowing which parts matter
- Testing in a streaming toy world before language is the right call — language masks too many problems with static pattern matching
- The thalamic sorter has a crisp interface spec that turns a vague neuroscience idea into a testable artifact

**Key risks:**
- The scope is enormous (Phases 0–10, 6 workstreams). Phases 0–4 should be treated as the real first project; everything after is contingent on results.
- The toy world itself is substantial engineering. Consider starting with the simplest environment that separates the model from a GRU (maybe 1D sequences with occlusion and two modalities) before building the full 2D world.
- The "thalamic" framing overlaps with existing ML work on gated routing (mixture of experts, cross-attention, adaptive computation). A literature pass on predictive coding networks (Rao & Ballard), active inference (Friston), and Global Workspace Theory implementations (Goyal et al., Juliani et al.) should happen early — some hypotheses may already have partial answers.
- The strongest and most testable hypothesis is **H1** (routing bottleneck improves hidden-state stability under noisy multimodal input). Optimize the project around getting a clean answer to that first.

### Prior work to review

Several existing research lines overlap with pieces of this plan. Review these before building to avoid reinventing known results and to calibrate hypotheses:

- **Rao & Ballard (1999)** — "Predictive coding in the visual cortex." Foundational paper on hierarchical predictive coding: each cortical level predicts the activity of the level below, and only prediction errors propagate upward. Directly relevant to the thalamic sorter's use of prediction error as a routing signal.

- **Karl Friston — Active Inference / Free Energy Principle.** A unifying framework where perception, action, and learning all minimize prediction error (variational free energy). The surprise modulation in Phase 6 is essentially a simplified version of precision-weighted prediction error from active inference. Key papers: Friston (2010) "The free-energy principle: a unified brain theory?"; Friston et al. (2017) "Active Inference: A Process Theory."

- **Goyal, Bengio et al. (2021)** — "Coordination Among Neural Modules Through a Shared Global Workspace." Implements Global Workspace Theory (Baars) as a differentiable attention-based broadcast mechanism across specialist modules. Directly tests claims similar to H3 (modular specialists + shared workspace vs monolithic). Check their results and task setup before designing Phase 4.

- **Juliani et al. (2022)** — "On the link between conscious function and general intelligence in humans and machines." Reviews computational implementations of Global Workspace Theory and their relationship to general intelligence. Useful for positioning this work.

- **Ha & Schmidhuber (2018)** — "World Models." Recurrent world model (VAE encoder + MDN-RNN + controller) that learns in a streaming environment. A strong baseline reference for Phase 2 — the GRU/LSTM world model baseline should be compared against this architecture pattern.

- **Mixture of Experts literature** — Shazeer et al. (2017) "Outrageously Large Neural Networks"; Fedus et al. (2022) "Switch Transformers." The thalamic sorter's gating mechanism is functionally similar to MoE routing. Understanding where MoE routing works and fails will inform the design of Phase 3.

---

## Goal

Build and test a minimal architecture in which:

- **thalamic sorting** integrates and routes sensory streams
- **neocortical modules** form stable categories and predictions
- **surprise** modulates learning and routing
- **multi-timescale memory** stabilizes hidden causes over time
- the whole system operates in a **streaming world**, not static i.i.d. samples

---

## Phase 0 — framing and experimental contract

Before code, define the contract. Deliverable: a 2–4 page design memo.

### Core claims

- Thalamus is modeled as a **recurrent routing bottleneck**
- Cortex is modeled as **distributed predictive categorization**
- Surprise increases **state update / memory write / learning emphasis**
- Intelligence should be evaluated in a **streaming partially observed world**

### Falsifiable hypotheses

- **H1:** A thalamic routing bottleneck improves hidden-state stability under noisy multimodal input relative to plain GRU
- **H2:** Surprise-weighted updates improve adaptation to change points without destroying stable representations
- **H3:** Modality-specific cortex modules plus shared workspace outperform a monolithic recurrent core on missing-modality recovery

### Baselines

- Plain GRU world model
- LSTM world model
- Monolithic recurrent predictor (no routing bottleneck)
- Full routed model

### Success criteria

Concrete thresholds, not "looks promising":

- Better occlusion tracking by X%
- Better missing-modality recovery by Y%
- Better latent stability under perturbation
- Equal or lower compute per step at same task score

---

## Phase 1 — toy world

The architecture should not be tested on language first. It should be tested where hidden causes, persistence, ambiguity, and streaming matter.

### Hypothesis

A streaming partially observed world reveals differences between routed predictive systems and plain sequence models better than static datasets.

### Environment requirements

- Persistent objects (2–5 moving)
- Occlusion behind walls
- Noisy observations
- Multiple modalities (visual, audio/event, proprioceptive)
- Change points / novelty
- Hidden latent causes that must be inferred over time

### Inputs per timestep

- Image frame (or simplified visual vector)
- Audio/event vector
- Agent/camera state vector

### Ground-truth latent factors (for evaluation only)

Object identity, position, velocity, type, occlusion status, active sound source, scene context / rule regime.

### Benchmark tasks

1. **Occlusion continuity** — track identity through disappearance and reappearance
2. **Cross-modal disambiguation** — use sound to infer hidden visual object
3. **Missing modality recovery** — predict missing audio from vision or vice versa
4. **Change-point adaptation** — object behavior rules suddenly change
5. **Stable categorization under perturbation** — same object under noise should retain identity

### Metrics

- Hidden-state tracking accuracy
- Next-step latent prediction loss
- Re-identification accuracy after occlusion
- Missing-modality reconstruction loss
- Adaptation time after rule change
- Latent cluster stability

### Decision rule

If the environment does not reliably separate trivial models from recurrent predictive models, redesign it before progressing.

### Deliverables

`env.py`, reproducible scenario generator, 5 benchmark task families, logging and visualization.

---

## Phase 2 — baseline predictive world model

Before testing the theory, establish the boring baseline.

### Hypothesis

A plain recurrent predictor provides a meaningful control condition for evaluating routing and surprise mechanisms.

### Architecture

Encoders per modality, single GRU or LSTM core, shared latent state, prediction heads for next-step latent prediction.

### Metrics

All environment metrics plus parameter count, step latency, training stability.

### Decision rule

Baseline must be stable and reasonably tuned before adding theory-heavy modules.

### Deliverables

`baseline_gru.py`, training loop, benchmark results on all toy tasks, compute profile.

---

## Phase 3 — thalamic sorter

The first real theory module.

### Research question

Can a recurrent bottleneck that integrates bottom-up input, top-down context, and surprise signals improve routing quality and downstream hidden-cause inference?

### Functional interpretation

The thalamic sorter is not biological thalamus in full. It is a module that:

- Merges modality embeddings
- Receives top-down context from workspace/cortex
- Receives prediction-error or surprise signals
- Emits gating/routing decisions and salience score
- Optionally emits broadcast context

### Interface

**Inputs:**
- `z_vis, z_aud, z_prop` (modality embeddings, each in R^D)
- `workspace_feedback` (R^D)
- `prev_error_norms` (R^k)
- Previous thalamic state `h_th`

**Outputs:**
- Routed embeddings: `r_vis, r_aud, r_prop`
- Gates: `g_vis, g_aud, g_prop`
- Salience scalar
- Updated thalamic state `h_th`
- Optional broadcast vector

### Internal requirements

Recurrent, differentiable, inspectable, sparse or semi-sparse routing preferred.

### Functional tests

1. **Modality conflict** — vision and audio disagree; does router shift emphasis based on context?
2. **Redundancy suppression** — one modality becomes noisy; does gate decrease?
3. **Surprise spike** — novel event occurs; salience should increase
4. **Top-down bias** — same input, different context; routing should differ

### Metrics

- Gate entropy / sparsity
- Salience vs true prediction error correlation
- Routing stability across small perturbations
- Routing sensitivity to change points
- Downstream task performance gain over no-router baseline

### Decision rule

Keep only if it produces interpretable routing changes and measurable performance gains on at least 2 benchmark tasks.

### Deliverables

`thalamus.py`, interface spec, gate visualizations, ablation study (no top-down feedback / no surprise input / no broadcast / no sparsity regularization).

---

## Phase 4 — neocortical categorization modules

Separate sorting from categorization.

### Research question

Do modality-specific recurrent cortical modules form more stable categories and better compositional hidden causes than a monolithic recurrent block?

### Architecture v0

- Visual cortex module
- Auditory cortex module
- Association cortex module
- Shared workspace

Each module receives routed modality input + workspace context + its own recurrent state, and outputs updated state + modality-specific predictive features + workspace contribution.

### Functional tests

1. **Category consistency** — same object under noise should map nearby
2. **Cross-modal binding** — visual and audio evidence of same cause should align in workspace
3. **Hierarchy** — low-level features change but higher-level identity remains stable
4. **Interference** — learning new patterns should not destroy old categories

### Metrics

- Latent cluster purity
- Temporal identity stability
- Cross-modal alignment score
- Representational drift over training
- Catastrophic interference score

### Decision rule

Keep modular cortex only if it improves category stability or missing-modality performance over monolithic baseline.

### Deliverables

`cortex.py`, clustering/probing scripts, representational similarity analysis, ablations (monolithic cortex / no association cortex / no workspace sharing).

---

## Phase 5 — shared workspace and multi-timescale memory

### Research question

Does splitting state into fast workspace and slow memory improve hidden-cause persistence and context handling?

### Functional interpretation

- **Fast workspace** = current scene hypothesis
- **Slow memory** = persistent causes, scene context, prototypes, longer-timescale regularities

### Tests

1. **Long occlusion** — can object identity survive longer absence?
2. **Context retention** — can model track current regime/rule?
3. **Memory write selectivity** — does memory update more on salient events?
4. **False persistence** — does model incorrectly preserve stale beliefs after true change?

### Metrics

- Long-horizon reidentification accuracy
- Memory write sparsity
- Persistence accuracy vs stale-state error
- Context-regime classification from slow state
- Retention vs adaptability tradeoff

### Decision rule

Retain only if it improves long-horizon continuity without sharply worsening change adaptation.

### Deliverables

`workspace.py`, `memory.py`, probing tools, ablations (no slow memory / single-state / slot vs EMA memory).

---

## Phase 6 — surprise modulation

### Research question

Can surprise-modulated updating improve learning efficiency and adaptation without causing instability?

### Approach

Start with surprise-weighted learning (not full RL):

- Stronger memory writes on surprise
- Stronger state update on surprise
- Stronger loss weighting on surprising transitions
- Optional curiosity/information gain later

Surprise estimated from: prediction error, change-point likelihood, cross-modal disagreement, state uncertainty.

### Tests

1. **Novelty burst** — unexpected event; does model adapt faster?
2. **Noise robustness** — random noise should not trigger excessive surprise
3. **Rule-switch** — surprise should spike at true regime changes
4. **Plasticity-stability** — increased learning on surprise should not destroy categories

### Metrics

- Adaptation time after novelty/change
- False-positive surprise rate
- Surprise correlation with actual change points
- Forgetting after surprise-driven updates
- Sample efficiency improvement

### Decision rule

Keep only if it improves adaptation speed with acceptable false-positive rate and limited representation collapse.

### Deliverables

`surprise.py`, alternative estimators, modulation mechanisms, ablations (no modulation / loss-only / memory-write-only / routing-only).

---

## Phase 7 — modality merge stability

Multimodal collapse is easy to get wrong. This deserves focused investigation.

### Research question

Can the system learn stable cross-modal fusion without one modality dominating?

### Tests

1. Modality dropout
2. Delayed modality arrival
3. Conflicting cues
4. Noisy single-channel
5. One modality absent for long spans

### Metrics

- Reconstruction with missing channels
- Agreement/disagreement sensitivity
- Dominance score
- Robustness to asynchronous input
- Fusion stability over time

### Decision rule

Fusion should degrade gracefully when one channel is removed and should not collapse into always trusting one modality.

---

## Phase 8 — evaluation suite

Should run in parallel from the start, not only at the end.

### Core metrics dashboard

| Category | Metrics |
|---|---|
| Prediction | Next-step latent loss, long-horizon rollout loss |
| Identity | Reidentification after occlusion, hidden-cause persistence |
| Fusion | Missing-modality recovery, cross-modal alignment |
| Categorization | Cluster purity, temporal category stability |
| Surprise | Salience/change-point correlation, adaptation speed, false alarm rate |
| Efficiency | Parameters, FLOPs/step, memory usage, wall-clock time |

### Deliverables

`metrics.py`, `eval.py`, report notebooks, standard result table template.

---

## Phase 9 — integrated architecture

Only after the parts are individually tested.

Combine thalamic sorter + modular cortex + workspace + slow memory + surprise modulation. Compare against baselines and ablations.

### Required ablations

- Full model
- No thalamus
- No surprise
- No slow memory
- Monolithic cortex
- No top-down feedback
- No multimodal input
- No sparsity regularization

### Decision rule

The full model should win for reasons you can explain, not just by having more parameters.

### Deliverables

Integrated training config, ablation matrix, benchmark report, "what worked / what failed" memo.

---

## Phase 10 — escalation paths

Only after the toy world gives clear results.

**Path A — Richer world model:** 3D scenes, embodied agent, active sensing, memory-guided exploration.

**Path B — Language / sequence adaptation:** online token routing, persistent workspace, surprise-guided memory updates.

**Path C — Spiking implementation:** event-driven gating, burst-like surprise modulation, spike-trace memory, energy/sparsity analysis.

---

## Execution order

| Stage | Work | Gate |
|---|---|---|
| 1 | Experimental charter (Phase 0) | Hypotheses and success criteria defined |
| 2 | Toy world + benchmarks (Phase 1) | Environment separates trivial from recurrent models |
| 3 | Baseline GRU/LSTM (Phase 2) | Stable reference scores |
| 4 | Thalamic sorter (Phase 3) | Measurable routing gains on 2+ tasks |
| 5 | Modular cortex + workspace (Phase 4) | Category stability improvement |
| 6 | Slow memory (Phase 5) | Long-horizon continuity gain |
| 7 | Surprise modulation (Phase 6) | Adaptation speed gain |
| 8 | Integrated ablations (Phase 9) | Full model wins for explainable reasons |

Each stage gates the next. Evaluation (Phase 8) runs continuously.

---

## Module contract

Every module must have:

- Clear input/output interface
- At least one synthetic test
- At least one ablation
- At least one metric tied to its purpose
- A decision rule for keep / revise / drop

This ensures "thalamic sorting" is not just an idea — it becomes a measurable artifact.
