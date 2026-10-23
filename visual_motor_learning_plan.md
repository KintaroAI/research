# Visual–Motor Learning From Scratch

**Project plan · v0.1 · October 2026**

## 1. Objective

Build a small, randomly initialized neural agent that learns to operate a mouse using **visual observations only**. The first environment is a minimal web page containing a visible cursor and a red circle. The agent must discover how to move, click, and eventually click the circle reliably, without instructions, object labels, DOM access, target coordinates, or distance-to-target rewards.

Study whether three mechanisms reinforce one another:

1. **Homeostatic motivation:** hunger and restlessness promote action without specifying which action is correct.
2. **Predictive learning:** the agent learns to predict sensory consequences of its actions.
3. **Reinforcement learning:** successful interactions become preferred behaviors.

**Research question:** Does joint prediction + internal drives enable faster acquisition and better generalization of visual–motor behavior than reward-only learning?

This is an experiment in learned sensorimotor control, **not** a claim that successful circle-clicking demonstrates AGI.

## 2. Rules / non-goals

### Information available to the agent

- A rendered **RGB screenshot** of the environment, initially 64 × 64 pixels.
- Its current **internal drive values** (e.g., hunger and restlessness), normalized to `[0, 1]`.
- A short causal history of observations, past actions, and optionally past rewards.
- Its own learned weights and internal representations.

### Information withheld from the agent

- DOM, accessibility tree, CSS, HTML, bounding boxes, object identities.
- Target or cursor coordinates from the environment's internal state.
- Distance or direction to the target.
- Hand-coded descriptions of which motor action means “left,” “right,” or “click.”
- Demonstrations, pretrained visual features, pretrained policies, or text tokens.

The environment **may use ground-truth geometry internally** to detect a successful click, calculate reward, and log evaluation metrics. That information must not leak into the model's input or auxiliary losses.

### Explicit design choices

- Start with **random neural weights** throughout.
- Begin **without distance-based reward**.
- Start with a **large circle** and simple background; gradually increase difficulty only by adjusting task conditions, not providing answers.
- Run many fast simulated episodes; use a real browser to verify that the same visual/action interface works end to end.
- Keep prediction learning **separate from the external reward**, at least in the first experiments, to avoid rewarding inaction or trivial predictable scenes.

## 3. Environment

A deterministic, lightweight 2D canvas is the training environment. A small HTML page renders the same task for human inspection.

```text
+--------------------------------------------------+
|                                                  |
|                    RED CIRCLE                    |
|                         ●                        |
|                                                  |
|       +   visible cursor                         |
|                                                  |
+--------------------------------------------------+

Observation: screenshot only (+ internal drives)
Action:      move dx, dy; click or no-click
Outcome:     new screenshot, scalar reward, done flag
```

### Proposed v0 settings

| Setting | Initial value |
|---|---|
| Rendered observation | 64 × 64 RGB, normalized to `[0, 1]` |
| Simulation viewport | 64 × 64 logical units (independent of browser DPI) |
| Background | Uniform dark color |
| Target | One solid red circle at a random position |
| Target radius | 12–16 logical pixels initially |
| Cursor | High-contrast, custom-drawn cursor included in every observation |
| Maximum episode | 100–200 action steps |
| Success | A click event while the cursor lies inside the circle |
| On success | End episode or respawn a new target (choose **end episode** for v0) |
| Reset | Randomize both cursor and circle locations |
| Time step | One action plus one resulting observation |

**Why a custom cursor?** Browser screenshots can omit the operating-system pointer. Drawing it in the canvas guarantees the agent can observe its own effect on the scene.

**Why a fast simulator?** Training via real desktop mouse events and full browser screenshots is unnecessarily slow. The simulator must produce the **same visual observations and action semantics** as the browser version. Validate equivalence before relying on simulator-only results.

## 4. Action space: continuous control, no action tokens

The Transformer does not use a discrete text vocabulary. It processes floating-point visual representations and generates continuous motor commands.

```text
a_t = (dx, dy, click)

dx, dy  : continuous motion requests within [-1, +1]
click   : a binary event sampled from a learned Bernoulli probability
```

- Map `dx`, `dy` to bounded pixel displacement (e.g., up to 4 pixels per step).
- Clamp the cursor to the viewport.
- One click command means one click event for that step, **not** holding the button down.
- For exploration, the motor head outputs a probability distribution over movements, not a single deterministic vector.
- Start with a **Gaussian movement policy** (means plus learned log standard deviations, with bounded outputs), and a **Bernoulli click policy**.
- The policy learns how its action values affect the screen from experience.

A fixed motor API is unavoidable: the environment interprets numerical actions as pointer control. The hypothesis is that the agent learns the **meaning and usefulness** of that API without being shown spatial labels or examples.

## 5. Architecture

A small **causal Transformer** with continuous inputs, a shared visual backbone, and three heads.

```text
    RGB screenshot I_t                Hunger H_t / Restlessness B_t
             |                                     |
       Small CNN encoder                            |
             |                                     |
       Visual embedding ----------------------------+
             |
    + history of prior actions and observations
             |
       Linear projection + temporal positions
             |
      Causal Transformer (shared memory)
             |
             +----------------+------------------+
             |                |                  |
             v                v                  v
       MOTOR HEAD         VALUE HEAD       PREDICTION HEAD
       distribution       V(state)        next image / drives
       for dx,dy,click                      conditioned on a_t
             |
        sample action ------------------------------+
             |                                     |
             v                                     |
        ENVIRONMENT                           prediction loss
             |                                     ^
             +--> next screenshot / drives --------+
             |
             +--> scalar reward --> policy/value updates
```

### Initial model budget

| Part | Suggested configuration |
|---|---|
| Visual encoder | 3-layer small CNN, learned from scratch |
| Visual feature size | 128 |
| Transformer | 2 layers, `d_model=128`, 4 attention heads |
| Feed-forward size | 512 |
| Context window | 32 time steps |
| Motor head | Continuous `(dx, dy)` distribution + click probability |
| Value head | One scalar, expected future return |
| Prediction head | Start with next-frame prediction at low resolution |
| Total parameters | Aim for **< 2M** initially; verify exact count in code |

The Transformer backbone alone has roughly **0.4M parameters**. A pixel reconstruction decoder can add parameters; its exact size depends on the implementation.

### Do separate heads isolate learning?

**No.** The CNN and Transformer are **shared parameters**. Gradients from policy learning, value learning, and prediction learning can update the shared representation. Each head has task-specific output parameters.

This is **shared representation learning**, *not yet planning*. To use imagined outcomes actively for action selection, a later version must evaluate candidate actions with its learned dynamics model.

### Causal ordering: avoid prediction leakage

At step `t`:

1. Observe `I_t`, `H_t`, `B_t` and past history.
2. Compute shared state `z_t`.
3. Sample `a_t` from the motor policy.
4. Predict `I_(t+1)` (and optionally next drives) **conditioned on `z_t` and `a_t`**.
5. Execute `a_t` in the environment.
6. Receive actual `I_(t+1)`, drives, reward, and termination.
7. Record the transition; train from batches of transitions.

Do not let the prediction of `I_(t+1)` see the actual future frame, and do not let the motor policy use its own sampled action as an input when deciding that same action.

## 6. Internal drives and rewards

### Hunger: pressure to achieve a successful interaction

- Hunger increases slowly each step.
- A successful circle click resets or substantially reduces hunger.
- Motion alone does **not** reduce hunger.
- Expose hunger as a scalar observation to the agent.

Example dynamics:

```text
H_(t+1) = clamp(H_t + h_rate - h_relief * successful_click, 0, 1)
```

### Restlessness: pressure to explore initially

- Restlessness grows during inactivity.
- Moving provides small, temporary relief.
- Repetitive or excessive motion gives diminishing relief.
- Empty clicks can provide a small, diminishing exploratory reward.
- Motion relief must be **strictly bounded** within an episode; waiting should not regenerate an unlimited reward budget.

Example *environmental* activity budget:

```text
movement_bonus = m0 * max(0, 1 - cumulative_movement / movement_budget)
click_bonus    = c0 * max(0, 1 - empty_click_count / click_budget)
```

Movement can be measured by **actual cursor displacement**, not the requested movement (important at screen boundaries). Reward should not be granted repeatedly for a command that leaves the pointer stationary.

These are intentional exploration incentives, **not** target-proximity signals. They must be much smaller in total than the reward for one successful target click.

### Reward proposal (starting values, to tune)

```text
r_t = 10.0 * successful_click
    + movement_bonus
    + empty_click_bonus
    - 0.001                  # time/action cost
    - 0.01 * H_t             # optional ongoing hunger cost
```

**Do not count the hunger reset twice** unless intentionally studying that variation: the circle reward is already the major positive reinforcement. In the first baseline, use the explicit +10 success reward; compare adding a hunger cost separately.

**Reward-balance constraint:** maximum cumulative exploration bonuses per episode should remain well below +10 (preferably < +1). Otherwise the agent may rationally farm movement/click bonuses instead of finding the circle.

### Anti-exploit tests

- Remaining still must not earn unlimited reward.
- Oscillating left/right must not earn unlimited reward.
- Clicking the same empty spot must not earn unlimited reward.
- Holding movement against a screen edge must not earn movement rewards.
- Waiting must not reset action bonus budgets.
- Episode timeout must not become a desirable substitute for success.
- If discomfort is an input, train with varying initial discomfort so the model can learn its significance.

No special-case anti-cheat information is fed to the neural network; these are environment-level reward rules.

## 7. Prediction learning

The agent predicts the next observation **after taking an action**. This provides dense feedback even when external rewards are rare.

### v0: predict actual low-resolution pixels

```text
I_t → encoder → shared Transformer state z_t
                              + chosen action a_t
                                      |
                                      v
                             small image decoder
                                      |
                                      v
                               predicted I_(t+1)
```

Train with a next-frame loss such as mean squared pixel error:

```text
L_pred = mean((predicted_next_image - actual_next_image)^2)
```

Because the screen is mostly static, evaluate prediction quality **both globally and on changed pixels**. As a diagnostic, a changed-pixel-weighted loss can be explored, provided it uses only consecutive rendered images rather than hidden target coordinates.

**Do not initially reward the agent for prediction accuracy.** Otherwise it may prefer an unchanged, easy-to-predict screen. Use prediction error as a supervised auxiliary training loss. A later curiosity experiment can test *learning progress* instead of raw accuracy, with safeguards against memorization and intrinsically unpredictable events.

### v1: compress prediction into a latent state

```text
z_t = Encoder(I_t)
predicted_z_(t+1) = Dynamics(z_t, a_t)
```

Advantages: much less output bandwidth and potentially more compact dynamics.

Risk: if encoder and predictor learn together from plain latent MSE alone, they may collapse to constant vectors. Prevent this with an image reconstruction loss, frozen/EMA target encoder plus suitable anti-collapse objective, or contrastive/variance constraints. Verify that representations retain cursor and target information.

### Optional next-drive prediction

Also predict `H_(t+1)` and `B_(t+1)` from the current state and action. This costs almost nothing computationally and lets the system learn the effects of its behavior on internal needs. These values are training targets, not hints about target geometry.

## 8. Training algorithm

Use **PPO** for the stochastic motor policy, with a shared backbone and value function. Use supervised loss for the prediction head.

```text
L_total = L_PPO_policy
        + c_value * L_value
        - c_entropy * policy_entropy
        + c_pred * L_prediction
```

The exact PPO clipping, return estimation (e.g., GAE), and normalization mechanics should follow a tested implementation. For initial stability, consider reducing the prediction-gradient weight reaching the shared backbone if the auxiliary task dominates policy learning.

### Training iteration

```python
# Schematic, not a complete PPO implementation
for iteration in range(num_iterations):
    trajectories = []
    for env in parallel_environments:
        obs = env.current_observation()
        for step in range(rollout_steps):
            state = model.encode_with_history(obs)
            action, logprob, value = model.policy_sample(state)
            predicted_next_frame = model.predict_next(state, action)
            next_obs, reward, done, info = env.step(action)
            trajectories.append((obs, action, logprob, value,
                                 predicted_next_frame, next_obs,
                                 reward, done))
            obs = env.reset() if done else next_obs
    model.ppo_and_prediction_update(trajectories)
```

For a Transformer, save and reset per-environment context/history at episode boundaries; pad or mask shorter histories. During PPO updates, recompute action log-probabilities from the stored observations/actions while using the stored *old* log-probabilities in the clipping objective.

## 9. Experimental matrix

Keep model capacity, action space, environment, rollout budget, and random-seed protocol comparable. Run multiple seeds (preferably at least 5) and report distributions, not a single lucky run.

| Run | Exploration bonuses | Hunger/restlessness | Prediction loss | Circle success reward |
|---|---|---|---|---|
| A: baseline | No | No | No | Yes |
| B: exploration | Yes | No | No | Yes |
| C: prediction | No | No | Yes | Yes |
| D: combined | Yes | Yes | Yes | Yes |
| E: drives only | No | Yes | No | Yes |
| F: full minus prediction | Yes | Yes | No | Yes |

**Important:** To isolate the effect of *internal drives* from bonus rewards, don't change both at once without controls. Drives may be observational state variables, ongoing costs, or both; record which definition each run uses.

Follow-up architectural comparison:

- Same CNN + **GRU** instead of Transformer.
- Same CNN + **tiny causal Transformer**.
- Equalize parameter count or report both parameter count and compute; sequence models have different memory/computation costs.

## 10. Metrics and success criteria

Track every training iteration:

- **Success rate:** fraction of episodes ending in a correct click.
- **Steps to success:** median and distribution, including timeouts.
- **Clicks per success:** efficiency versus indiscriminate clicking.
- **Cursor path length / straight-line distance:** for *evaluation only*, never model input or training reward.
- **First-success sample efficiency:** environment interactions required for first hit.
- **Learning curve:** success rate versus total environment steps across seeds.
- **Generalization:** unseen random positions, smaller circles, different backgrounds and initial cursor locations.
- **Prediction quality:** full-frame and changed-region errors; predicted vs observed frames.
- **Drive trajectories:** hunger/restlessness through episodes.
- **Compute:** transitions per second, GPU memory usage, time spent rendering vs network updates.

Proposed *milestones*, not guaranteed results:

1. First accidental successful click observed and correctly rewarded.
2. Success rate clearly exceeds a randomized-policy baseline.
3. Consistent success on new cursor/target positions.
4. Mean steps per successful click decreases with training.
5. Prediction model visibly captures the cursor's response to movement.
6. Ablations show whether drives and prediction materially help.

Evaluate periodically with exploration bonuses disabled and a fixed evaluation protocol, while retaining any internal state variables that are part of the learned policy's observation definition. This tests whether the agent has learned the underlying task rather than merely pursuing temporary bonuses.

## 11. Implementation roadmap

### Phase 0 — Environment and reproducibility

- [ ] Implement a tiny deterministic renderer and custom on-screen cursor.
- [ ] Implement hit testing and screenshot capture (no DOM information in observations).
- [ ] Implement continuous movement, click event, clipping, target respawn/reset.
- [ ] Implement reset seeding, action logs, and video replay.
- [ ] Validate reward accounting with scripted tests and a random policy.
- [ ] Add a minimal browser page with the same pixels/action interface.

**Exit criterion:** random actions produce valid observation/action/reward transitions, and browser rendering behaves like the training environment.

### Phase 1 — Sparse-reward control baseline

- [ ] Implement visual CNN + tiny causal Transformer + motor/value heads.
- [ ] Implement PPO with correct rollout bookkeeping.
- [ ] Train with **only** the +10 circle success reward and a small step penalty.
- [ ] Record random-policy baseline and learning curves.

**Exit criterion:** baseline quality quantified, even if learning fails. Failure is useful evidence about sparsity.

### Phase 2 — Diminishing action incentives and drives

- [ ] Add bounded movement and empty-click exploration bonuses.
- [ ] Add hunger/restlessness dynamics and input channels.
- [ ] Check that reward farming and inactivity strategies do not dominate.
- [ ] Compare with baseline using the same seeds and environment schedule.

**Exit criterion:** measure whether these mechanisms improve exploration without becoming the primary objective.

### Phase 3 — Joint next-frame prediction

- [ ] Add action-conditioned image prediction head.
- [ ] Train prediction loss alongside PPO, without prediction accuracy reward.
- [ ] Save side-by-side predicted/actual screenshots and changed-pixel error.
- [ ] Compare against a no-prediction run at similar compute cost.

**Exit criterion:** predicted frames show nontrivial action consequences and the effect on policy learning is measured.

### Phase 4 — Generalization and architectural comparisons

- [ ] Hold out cursor/target configurations for evaluation.
- [ ] Reduce target radius after competence improves; test moving targets later.
- [ ] Compare GRU versus Transformer with matched experimental conditions.
- [ ] Try latent dynamics only after the pixel-prediction baseline is established.
- [ ] Repeat across multiple seeds; publish plots and representative rollouts.

**Exit criterion:** robust evidence about which mechanisms contribute and where they fail.

### Phase 5 — Model-based action selection (optional)

- [ ] Predict consequences of several sampled candidate actions.
- [ ] Score predicted outcomes with the value model / learned drive consequences.
- [ ] Choose an action using imagined outcomes, then execute it.
- [ ] Compare with the direct policy under matched computation.

**Exit criterion:** determine whether explicit imagination/planning helps beyond a shared representation alone.

## 12. Suggested project structure

```text
visual-motor-agent/
├── README.md
├── pyproject.toml
├── configs/
│   ├── baseline.yaml
│   ├── drives.yaml
│   └── predictive.yaml
├── env/
│   ├── circle_world.py       # renderer, transitions, reward, drives
│   └── browser_demo/         # HTML canvas visualizer
├── agent/
│   ├── vision.py             # CNN encoder
│   ├── transformer.py        # continuous-input causal Transformer
│   ├── heads.py              # motor, value, prediction heads
│   └── model.py              # shared state and causal execution
├── training/
│   ├── ppo.py
│   ├── rollout.py
│   └── train.py
├── evaluation/
│   ├── evaluate.py
│   ├── ablations.py
│   └── record_video.py
├── tests/
│   ├── test_env.py
│   ├── test_rewards.py
│   ├── test_no_information_leak.py
│   └── test_causality.py
└── results/
```

Stack: Python + PyTorch for the agent/training; NumPy and a lightweight 2D renderer for fast transitions; HTML5 Canvas for visualization; optional Playwright for real-browser integration; TensorBoard or CSV/Matplotlib for plots.

## 13. Decisions to revisit only after the first results

- Does the Transformer outperform a GRU on this tiny task?
- Does hunger add anything beyond a success reward plus episode timeout?
- Are exploration bonuses useful, or do they distract the policy?
- Does pixel prediction improve learning enough to justify its training cost?
- Should prediction learn full frames, visual differences, or compressed latent states?
- Is explicit multistep planning more effective than a direct motor policy?
- Can the same agent learn to interact with unseen visual objects and changing rules without hand-coded task knowledge?

## 14. Working principle

> **Do not tell the agent where the target is, what the circle means, or how to move toward it.** Give it pixels, a motor interface, internal needs, sparse consequences, and a way to learn to predict what its actions do. Then measure what it discovers.

