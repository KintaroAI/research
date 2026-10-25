# A unified Transformer for visual-motor learning

I think we can design this very close to what you're imagining: **one Transformer, no CNN, no pretrained components, and no separate neural networks for vision, movement, or internal state.**

The model would receive a continuous stream of numerical values and learn to predict what happens next.

My main recommendation is to use a **continuous-token Transformer**, with every input represented in the same format, regardless of whether it represents pixels, mouse movement, clicking, hunger, or reward.

The important architectural decision is how we organize those values into tokens.

## 1. First, how Transformers actually work

A Transformer doesn't fundamentally understand words, images, or actions.

It receives a sequence of vectors:

\[
X=[x_1,x_2,\ldots,x_n]
\]

Each vector has the same dimensionality. For example:

\[
x_i\in\mathbb{R}^{128}
\]

The Transformer processes these vectors through two main operations:

**Attention:** Each vector can gather information from other vectors, depending on learned relationships.

**Feed-forward transformation:** Each vector is independently transformed through learned matrix operations and nonlinearities.

Both operations are repeated across multiple layers.

There is no requirement that the inputs represent discrete text tokens.

### Text versus visual Transformers

| Characteristic | Text Transformer | Vision Transformer | Our proposed Transformer |
|---|---|---|---|
| Input | Token IDs | Image patches | Continuous numerical values |
| Initial conversion | Embedding lookup | Linear projection | Shared linear projection |
| Internal representation | Floating-point vectors | Floating-point vectors | Floating-point vectors |
| Attention mechanism | Same | Same | Same |
| Output | Usually vocabulary probabilities | Depends on task | Continuous predictions |
| Training from scratch | Possible | Possible | Yes |

The fundamental Transformer architecture doesn't change.

For example, the original Vision Transformer demonstrated that images can be processed directly as sequences of patches, using linear projections instead of a CNN. [Google Research](https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/?utm_source=chatgpt.com)

---

## 2. What should count as a token?

You proposed:

- 64 × 64 grayscale pixels: 4,096 values.
- Mouse movement: `dx`, `dy`.
- Mouse click: `0` or `1`.
- Hunger: one continuous value.

That's **4,100 numerical values per timestep**.

We have three ways to organize them.

| Approach | Visual tokens per frame | Advantage | Disadvantage |
|---|---:|---|---|
| One token per pixel | 4,096 | Every pixel is individually accessible to attention | Extremely expensive with history |
| One token per image | 1 | Very computationally efficient | Loses useful spatial structure in the initial compression |
| One token per image patch | 64–256 | Good balance between spatial information and computation | Requires grouping neighboring pixels |

I would choose **8 × 8 pixel patches**, producing 64 visual tokens per frame.

This doesn't require any CNN or visual feature extraction.

We're merely dividing the 64 × 64 array into 64 groups of 64 raw pixel values.

It's still the Transformer that must learn what those values mean.

Crucially, **the same input projection can process image patches, actions, and internal-state values**. We only need to package them into vectors of equal length.

## 3. Making all inputs truly uniform

Here's how I would organize a single timestep.







For each timestep, we construct 67 vectors, each containing 64 floating-point values.

### Visual input

Take the 64 × 64 grayscale image, normalized to `[0,1]`.

Split it into 8 × 8 patches.

Each patch becomes a vector:

\[
p_i=[p_1,p_2,\ldots,p_{64}]
\]

There are 64 such vectors.

### Motor input

We create another 64-dimensional vector:

\[
a_t=[dx,dy,click,0,\ldots,0]
\]

For example:

```python
action = [0.4, -0.2, 1.0] + [0.0] * 61
```

Movement values are normalized to `[-1,1]`, where the sign indicates direction. Clicking is represented by `0` or `1`.

### Internal-state input

Similarly:

```python
body = [hunger, restlessness] + [0.0] * 62
```

And we can represent the previous reward as another vector:

```python
reward = [last_reward] + [0.0] * 63
```

All these vectors go through **exactly the same learned input projection**:

\[
e_i=Wx_i+b
\]

Where:

\[
W\in\mathbb{R}^{128\times64}
\]

This operation requires just:

\[
64\times128+128=\boxed{8,320\text{ parameters}}
\]

And those same 8,320 parameters are reused for every patch, action, hunger, and reward token.

This is an important property of the design: **we aren't giving the model separate visual and motor-processing networks.**

### But we need position information

The Transformer must know that two adjacent patches occupy different positions on the screen.

It also needs to distinguish a motor command from a visual observation.

We can add small, learned position and role vectors:

\[
e_{t,i}=Wx_{t,i}+b+P_i+T_t+R_i
\]

Where:

- \(P_i\) identifies the token's spatial position or slot.
- \(T_t\) identifies the timestep.
- \(R_i\) identifies the token's role, such as visual input or action.

These are not hand-coded object features. They tell the model **where a value came from**, not what it means.

We aren't identifying circles, cursor coordinates, movement directions, or desirable outcomes.

The network must discover those relationships.

---

## 4. How the continuous stream should be organized

This is probably the most important part.

I would treat each timestep as a complete collection of observations and actions:

\[
S_t=(I_t,H_t,A_{t-1},R_t)
\]

Where:

- \(I_t\): Screenshot at time \(t\).
- \(H_t\): Current internal state.
- \(A_{t-1}\): Previous motor action.
- \(R_t\): Reward received after that previous action.

The Transformer receives a history:

\[
[S_{t-7},S_{t-6},\ldots,S_t]
\]

And must determine what to do next.

**However, producing an action and predicting its consequences are two different moments in time.**

We should preserve that distinction to prevent the model from seeing information it is supposed to predict.

The interaction should work as follows:

```text
            OBSERVE
               │
               ▼
       Screenshot at t
       Hunger at t
       Previous actions
       Previous rewards
               │
               ▼
          TRANSFORMER
               │
               ▼
        Generate action Aₜ
               │
               ▼
        Execute action Aₜ
               │
               ▼
       Predict next state
      using Sₜ and Aₜ
               │
               ▼
       Environment advances
               │
               ▼
       Receive actual Sₜ₊₁
               │
               ▼
       Compare predictions
       Update neural weights
               │
               └────► OBSERVE
```

We can implement this with one shared Transformer using different query positions.

For example, an empty **action query token** asks the Transformer to produce the next action. After that action has been chosen, empty **future-observation query tokens** ask it to predict the next image and internal state.

A suitable attention mask ensures that neither query sees future information.

This doesn't require separate processing networks. The query tokens use the same attention layers and shared parameters.

---

## 5. Prediction should use the same output mechanism

I agree with your desire to avoid specialized heads where possible.

We could use one output projection:

\[
W_{\text{out}}:\mathbb{R}^{128}\rightarrow\mathbb{R}^{64}
\]

Every output token produces 64 floating-point values.

We interpret those values according to the output slot:

| Output slot | Meaning |
|---|---|
| Visual slot 0 | Predicted 64 pixels of patch 0 |
| Visual slot 1 | Predicted 64 pixels of patch 1 |
| ... | ... |
| Visual slot 63 | Predicted 64 pixels of patch 63 |
| Action slot | `dx`, `dy`, click probability |
| Internal-state slot | Predicted hunger and restlessness |
| Reward slot | Predicted next reward |

All output slots share the same learned projection weights.

There is still a minimal amount of interface-specific interpretation: mouse movement must be converted into cursor displacement, clicking into a binary event, and visual predictions into pixel intensities.

**I don't think we should try to eliminate that distinction.** Those conversions define the boundary between the model and its environment. They aren't separate intelligence modules.

### Learning to predict

Suppose the agent observes:

\[
I_t,H_t
\]

And executes:

\[
a_t=(0.2,0.0,0)
\]

The Transformer predicts:

\[
(\hat I_{t+1},\hat H_{t+1},\hat R_{t+1})
\]

The environment then returns:

\[
(I_{t+1},H_{t+1},R_{t+1})
\]

We calculate three prediction errors:

\[
L_{\text{visual}}=\operatorname{MSE}(\hat I_{t+1},I_{t+1})
\]

\[
L_{\text{internal}}=\operatorname{MSE}(\hat H_{t+1},H_{t+1})
\]

\[
L_{\text{reward}}=(\hat R_{t+1}-R_{t+1})^2
\]

And combine them:

\[
\boxed{
L_{\text{prediction}}=
\lambda_v L_{\text{visual}}+
\lambda_h L_{\text{internal}}+
\lambda_r L_{\text{reward}}
}
\]

Notice what we're accomplishing.

The network learns to predict both **external reality and its own internal condition**.

It can learn that moving the cursor changes the visual field, that hunger grows with time, and that certain interactions produce a sudden decrease in hunger.

All those relationships are learned within the same Transformer.

### A problem with visual prediction

Our screen is mostly black.

Consequently, a network predicting an entirely black image might achieve a very low average pixel error while completely ignoring the cursor.

We should measure how well it predicts *changes*, not merely whether the overall screenshot looks correct.

One possibility is a loss that pays more attention to pixels that changed between frames:

\[
L_{\text{visual}}=
\frac1N\sum_i
\left(1+\alpha|I_{t+1}^{(i)}-I_t^{(i)}|\right)
(\hat I_{t+1}^{(i)}-I_{t+1}^{(i)})^2
\]

This doesn't give the network target coordinates, object identities, or movement directions.

It emphasizes sensory changes wherever they occur.

I would initially compare this against ordinary MSE rather than assume it's better.

There's another complication: when a circle is clicked and respawns at a genuinely random location, its exact next position is impossible to predict. We should not expect the model to predict inherently unknowable randomness. We could exclude random reset transitions from the deterministic prediction loss initially.

---

## 6. Prediction alone won't teach it which actions to take

This distinction is essential.

Suppose the network becomes extremely good at predicting what will happen if it moves the mouse right.

That doesn't necessarily mean it will choose to move right.

Prediction and action selection are different learning objectives.

There are two ways to organize them without introducing separate neural networks.

### Option A: Reinforcement learning

The same Transformer produces actions from its action-query position.

We interpret its outputs as a probability distribution over mouse movements and clicking.

For example:

\[
dx,dy\sim\mathcal{N}(\mu,\sigma^2)
\]

\[
click\sim\operatorname{Bernoulli}(p)
\]

An RL algorithm trains the network to select actions that produce rewards.

Meanwhile, the prediction loss trains the same network to predict their consequences.

We can use PPO with a value prediction obtained from an additional query slot.

### Option B: Learning through imagined actions

This is more interesting conceptually.

Instead of directly learning which action to output, the network could generate several possible actions and predict their consequences.

For instance:

| Candidate action | Predicted visual change | Predicted hunger |
|---|---|---|
| Move left | Cursor moves left | 0.81 |
| Move right | Cursor moves right | 0.81 |
| Click | Circle disappears | 0.15 |
| Do nothing | Screen unchanged | 0.82 |

The agent can choose actions according to their predicted consequences.

It would be using its own predictions to guide behavior.

This is **model-based control**.

However, there's an important limitation: the agent may need several movements before clicking becomes useful. Evaluating only the immediate next state won't be enough.

We would need to predict sequences of future actions and observations.

Also, initially, its predictions will be extremely inaccurate. An optimizer could exploit those inaccuracies rather than discover real rewards.

**My choice for the first experiment would be Option A**, while keeping the architecture compatible with Option B later.

There's precedent for using Transformers to process histories of states, actions, and rewards as one sequence. Decision Transformer, for example, demonstrated the value of this representation, although its original learning setup differs from the online, predictive agent we're designing. [papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html?utm_source=chatgpt.com)

---

## 7. How large would the Transformer need to be?

I'd start with the following configuration.

| Parameter | Value |
|---|---:|
| Screen | 64 × 64 grayscale |
| Patch size | 8 × 8 |
| Visual tokens | 64 |
| Additional input tokens | 3 |
| Values per input token | 64 |
| Transformer hidden dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feed-forward dimension | 512 |
| History | 8 timesteps |
| Input tokens in history | 536 |

The Transformer itself would have approximately 400,000 parameters.

The shared input and output projections would add roughly 16,600 more, plus a small number for positions and output distributions.

We could keep the complete architecture comfortably below one million parameters.

### Computational considerations

With 8 timesteps of history, we have:

\[
67\times8=536\text{ tokens}
\]

Standard attention processes approximately:

\[
536^2=287,296
\]

token pairs per head, per layer.

Compare that with using every pixel as an independent token.

Eight frames would contain approximately 32,800 tokens, requiring over a billion attention-score positions per head.

**Patch grouping reduces the number of attention-score positions by approximately 3,700 times in this example.**

That's a compelling reason not to use one token per pixel.

The precise computational savings are smaller when including linear layers and other operations, but the attention reduction is substantial.

### What about giving the whole screen to one token?

We could do that too.

Each timestep could simply be:

\[
x_t\in\mathbb{R}^{4100}
\]

And we'd use:

\[
W:\mathbb{R}^{4100}\rightarrow\mathbb{R}^{128}
\]

Then the Transformer would operate on a sequence of only eight vectors representing eight timesteps.

This would be exceptionally computationally efficient.

But the network would have to learn the entire spatial organization of the screen through dense weights before temporal attention could help.

It might still solve our extremely simple circle-clicking environment.

In fact, I would eventually compare this approach against patch-based tokenization to see whether spatial structure actually improves learning.

---

## 8. Continuous learning: Two kinds of memory

This is another important distinction.

A Transformer with an eight-timestep context window can access the last eight observations, but that doesn't mean it permanently learns from them.

We have two mechanisms:

**Working memory:** The sequence of recent observations and actions processed by attention.

**Learned memory:** The model's weights, which change through training.

We want both.

I suggest this training process:

1. Observe the current screen and internal state.
2. Generate an action using the recent history.
3. Execute the action.
4. Predict and observe the resulting state.
5. Save the transition in an experience buffer.
6. Periodically update the shared Transformer using recent and replayed experiences.
7. Continue interacting with the environment using the updated weights.

For prediction, we can learn from virtually every interaction. No successful click is necessary.

For reinforcement learning, we use the actual rewards to improve action selection.

We should avoid updating the model exclusively from the latest frame because closely correlated observations and continuously changing weights can create unstable learning and forgetting.

An experience buffer helps.

There is a technical detail with Transformers: when we change their weights, cached attention representations from older weights become stale. For this small experiment, recomputing the recent context after each weight update should be inexpensive.

---

## 9. The experiment I would actually run first

I would separate the first investigation into three phases.

### Phase 1: Learn the physics of the cursor

The network starts with random weights.

It receives screenshots and produces random movements and clicks.

We train only its predictions.

The question is:

**Can a small Transformer discover how its motor outputs change its visual input?**

After training, we could visualize its predictions:

| Input action | Initial prediction | Learned prediction |
|---|---|---|
| Move right | Mostly noise or static image | Cursor appears farther right |
| Move left | Incorrect image | Cursor appears farther left |
| Click empty space | Incorrect change | Mostly unchanged screen |
| Click circle | Incorrect change | Circle disappears or changes |

No reward is necessary for this phase.

We aren't training it to navigate toward the circle yet. We're establishing whether it can learn visual dynamics.

### Phase 2: Introduce hunger and reward

Now the model receives hunger as an additional continuous input.

Hunger increases with time and drops when the circle is clicked successfully.

Random actions can lead to a successful click.

That gives the agent its first experience of the connection between a visual interaction and internal reward.

We add reinforcement learning to the action selection process.

The prediction model continues learning simultaneously.

### Phase 3: Introduce restlessness and exploration rewards

We test your idea that inactivity creates discomfort, while initially novel actions provide temporary relief.

This should encourage exploration even when successful clicks are rare.

We can compare the learning speed with and without these drives.

I'd keep restlessness relief finite and make repetitive actions progressively less effective.

The model receives only current internal variables, previous experience, and visual observations. It never receives target positions.

---

## 10. One fundamental question about the architecture

There's something I'd like to emphasize because it relates directly to your original idea of a unified intelligence architecture.

Even though our Transformer has no specialized visual or motor-processing networks, **its different input positions have different meanings**.

We know that certain values are pixels, that others represent motor commands, and that hunger is another scalar.

The model doesn't know how those values relate to one another, but we have already imposed a particular organization.

That's unavoidable to some extent: the environment needs an interface.

However, we can minimize those assumptions.

For example, the architecture could eventually receive an arbitrary number of sensory channels and motor channels, each represented as continuous vectors with position or channel identifiers.

The Transformer would use the same weights to process all of them.

That would let us investigate whether the model can adapt to a different environment or a different set of actuators.

This is more ambitious than clicking circles, but the architectural foundation would be similar.

---

# My recommendation

I would build **a single GPT-style Transformer operating on continuous vectors**, with the following properties:

- No token vocabulary.
- No CNN or pretrained visual encoder.
- All inputs use one shared linear projection.
- Pixels are grouped into 8 × 8 patches strictly for computational efficiency.
- Motor commands, hunger, and rewards use the same input representation.
- Attention operates across spatial positions and recent timesteps.
- One shared Transformer predicts actions, future visual observations, internal state, and reward.
- Online learning combines prediction errors with reinforcement learning.

We would need carefully arranged queries and causal masking, but not separate neural processing pipelines.

**The most interesting architectural possibility is that the same internal representations learned to predict changes in the environment could also become useful for controlling that environment.**

And I would keep one competing experimental architecture: **the entire 64 × 64 image as a single continuous vector per timestep**.

That gives us a way to test whether dividing the visual field into patches genuinely helps or whether a completely uniform numerical stream is sufficient.

My inclination is that **the patch-based Transformer will learn faster and generalize better**, but that's a hypothesis worth testing rather than building into our assumptions.
