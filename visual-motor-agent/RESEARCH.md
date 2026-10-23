# Visual–Motor Agent: Literature Review

Compiled 2026-10-23 for `visual_motor_learning_plan.md`. Bibliographic data were checked against the arXiv API, Crossref, PMLR, NeurIPS, JMLR and mlanthology listings. Quoted numbers come from the primary PDFs. Tags: **[unverified]** means I could not confirm it; **[recollection]** means it is from memory and not re-checked; **[our sim]** means our own Monte Carlo, not literature.

## Summary

The prior work that drove computers from pixels did not learn from pixels alone. World of Bits, CC-Net, Pix2Act and VPT all relied on demonstrations or pretraining. CC-Net's RL-only run was "much worse", and VPT reports RL from scratch with native mouse and keyboard as "near impossible". In computer-control settings, pure pixel RL without demonstrations has worked only on simple, dense tasks (AndroidEnv "catch"). Published pointing agents also use distance shaping (User-in-the-Box). Even so, a sparse "inside-the-target" analog, DMC reacher-easy, is solved from pixels in about 0.5M steps by off-policy agents, and our world is much smaller.

Our simulation also shows that **v0 is not a sparse-reward problem**. At the `dev/` defaults (radius 12–16, 200 steps, cursor starting outside the circle), the untrained policy (σ≈0.61, p_click≈0.5) already succeeds in about 60% of episodes, and in 73% with σ=1. Clicks are also free, so "always click" is near-optimal. As a result, success rate will saturate, and the A–F ablations need steps-to-success, click precision and a hard condition where random success is about 5%.

For prediction, copy-last-frame is about 98–99% correct per pixel. So:
- predict the change, or weight the changed pixels (UNREAL: change prediction beat reconstruction; SimPLe: loss clipping let the model see the small ball);
- condition on the action multiplicatively (Oh et al.: a model without action input cannot predict the controlled object);
- always report skill relative to copy-frame and shuffled-action baselines.

Auxiliary losses help with sparse rewards and little data, but they can hurt final performance or destabilize a shared trunk (UNREAL, SAC+AE, PPG). So scale the prediction gradient into the trunk (β ∈ {0, 0.1, 1}) and add a decoupled "prediction-only trunk" arm; ATC shows decoupled encoders can match end-to-end PPO.

The contingency literature supplies reward-free tests. Counterfactual rendering gives ground-truth contingent regions (Bellemare). Attention-argmax localization error (Choi), selectivity (Thomas/Bengio), ΔPSNR between true and shuffled actions (Genie), linear probes against random-init controls (Anand) and inverse-dynamics R² (VPT) cover the rest.

For PPO, use the "37 details" defaults with Procgen-like settings and these specifics:
- state-independent log-std with initial σ≈0.5–1, a bounded mean, and stored unclipped samples;
- separate entropy coefficients for movement and click;
- no value clipping, and fixed reward scaling across arms;
- bootstrap at timeouts;
- a Markov (no-memory) baseline before the transformer. GTrXL starts near-Markov, and in POPGym, GRU and MLP agents often match more complex memory models.

## 1. Pixel-based computer control

**World of Bits / MiniWoB (Shi et al. 2017) [1]**
- Observation: pixels, plus the DOM with element bounding boxes, plus reward. MiniWoB pages are 210×160 px (top 50 px is the rendered instruction). The MiniWoB baseline consumed only the 210×160×3 image (5 conv layers, stride 2).
- Action: the cursor is a categorical over a 20×20 grid (8-px spacing) × {move, drag, click}, giving 1,200 actions. They "also experimented with a Gaussian distribution but found it inadequate due to its unimodal shape" (absolute positions with multimodal targets).
- Training: behavioral cloning on about 17 h of demonstrations (720k pairs), then A3C (12 envs, ≤1M steps, γ=0.9, λ=0.95). Human-normalized success rate: random 20.8%, BC 24.8%, BC+RL 34.8%. **No RL-from-scratch result is reported.** Random policies score well on some tasks because actions run at 12 fps.
- Feeding in previous actions *hurt*: the agent "develops a tendency to meander". Relevance: treat past-action inputs as an ablation.

**MiniWoB++ (Liu et al. 2018) [2]; DOM-based RL [3,4]**
- Agents act on DOM elements (`Click(e)`, `Type(e,t)`), not pixels. Workflow-guided exploration from 3–10 demonstrations per task was more than 100× more sample-efficient than BC.
- Pure RL without demonstrations works *once actions are DOM elements*: DOM-Q-Net (DQN over a DOM graph) [3] and QWeb (DQN with curricula and meta-generated instructions) [4]. This object-level structure is exactly what our project removes.

**CC-Net (Humphreys et al. 2022) [5]**
- Observation: 165×220 RGB, plus DOM, plus instruction. A ResNet produces 154 visual tokens, an 8-layer multimodal transformer combines them with the text, and a 2-layer LSTM receives the previous action.
- Action: 10 action types (no-op, move, click, double-click, press, release, wheel up/down, key press, emit text). The cursor is two 51-bin categoricals, with y conditioned on x.
- Training: BC and V-MPO co-trained on 2.4M demonstrations (about 6,300 h from 77 people), with γ=0.9, lr 1e-4, unroll 64, ≤300 steps per episode and one seed per condition. Mean performance was human-level.
- RL-only and BC-only were "much worse"; RL-only appears only in figures. One-thousandth of the data (about 6 h) gave "no significant gain over RL only". Removing the DOM cost 75%, while removing pixels hurt less. **Pixel-only RL was not demonstrated.**

**AndroidEnv (Toyama et al. 2021) [6]**
- Observation: pixels (downsampled to 80×120 in the baselines), plus time delta and orientation. Raw action: a continuous (x, y) ∈ [0,1]² plus a type in {TOUCH, LIFT, REPEAT}. The environment is asynchronous and real-time.
- Continuous agents (DDPG, D4PG, MPO) "perform well only in tasks where the interface does not expect complex gestures". Discrete agents (DQN, IMPALA, R2D2 on a 6×9 grid, giving 108 actions, plus a one-hot of the last action) did better overall. "catch" was solved by almost all agents and "blockinger" by none. No demonstrations were used.

**Pix2Act (Shaw et al. 2023) [7]; VPT (Baker et al. 2022) [8]**
- Pix2Act uses only pixels and generic mouse/keyboard actions, with Pix2Struct pretraining, BC and tree-search policy improvement. It beats human crowdworkers on MiniWoB++, and pretraining raises BC from 17.1 to 66.5. It names "relative movements with foveated binning" as an alternative action space.
- VPT uses the native mouse and keyboard at 20 Hz. Its inverse-dynamics model (IDM), trained on about 2k h of labeled play, reaches 90.6% keypress accuracy and **R²=0.97 for mouse movement**. With that interface, most simple tasks are "near impossible with RL from scratch". Relevance: IDM accuracy is a ready-made probe metric (§10, P6).

**Pointing from pixels with RL**
- DMC *reacher* [9]: "The reward is one when the end effector penetrates the target sphere", and the target position is randomized. The PlaNet pixel benchmark at 500k steps [10]: DrQ 942±71, CURL 904±94, SAC-AE 601±135, PlaNet 588±471. DrQ scores 601±213 at 100k, and SAC from state scores 975±5. This is the closest published analog to our task, but reward arrives on every step inside the target, there is no click, and the learners are replay-based.
- User-in-the-Box [11]: a muscle-actuated arm trained with PPO (Stable-Baselines3), using an egocentric RGB-D camera at 120×80 (pointing used depth) plus proprioception. Its pointing reward is a **distance-shaped penalty** plus a hit bonus (500 ms dwell), and Fitts'-law-like movement emerges. A follow-up [12] finds completion bonus plus proximity incentives "essential for task success". HCI pointing RL relies on shaping, and our no-shaping rule is a deliberate departure from it.

## 2. Auxiliary self-supervised losses in RL

**UNREAL (Jaderberg et al. 2017) [13]**
- Setup: A3C with an LSTM, plus three auxiliaries. *Pixel control* uses auxiliary Q-policies that maximize the mean absolute pixel change in each cell of a 20×20 grid of 4×4 cells over the central 80×80 crop. *Reward prediction* is 3-class (zero, +, −) on replay rebalanced 50/50 between rewarding and non-rewarding sequences. *Value replay* rounds it out.
- Results: on Labyrinth, a 10× mean speedup and 87% vs 54% human-normalized score; on Atari, 880% mean human-normalized.
- Input reconstruction learned faster early but hurt final performance, "because it puts too much focus on reconstructing irrelevant parts". Predicting input *change* beat reconstruction, and pixel control beat both.
- Loss weights: λ_PC drawn log-uniform in [0.01, 0.1] (Labyrinth) or [1e-4, 1e-2] (Atari); λ_RP=1. Relevance: predict change, and oversample rare hunger-relief events.

**Action-conditional video prediction (Oh et al. 2015) [14]**
- Feedforward (stacked-frame) and recurrent encoders, a multiplicative *factored* action transformation (2048 factors) and a deconvolutional decoder. A 1→3→5-step prediction curriculum was "necessary to stabilize the training".
- The no-action baseline (naFf) "fails to predict the controlled object". Squared error is a poor metric because the controlled object "occupies only a small part of the image".
- Predicting with only the high-variance action factors moves only the controlled object: the model "learns to distinguish between controlled objects and uncontrolled objects". This is a template for our prediction head and its action-blind control.

**SPR (Schwarzer et al. 2021) [15]**
- A transition model predicts EMA-target projections K=5 steps ahead using a cosine loss and is "truncate[d] ... at episode boundaries". It uses τ=0 with augmentation, or τ=0.99 with dropout when augmentation is off. Atari-100k median human-normalized score is 0.415 (a 55% gain over the prior state of the art), and the target encoder "has a large impact". This is the recipe for v1 latent prediction.

**CURL / RAD / DrQ / DrQ-v2 [16,17,10,18]; DrAC [19]**
- CURL uses InfoNCE with a momentum encoder. RAD and DrQ show that random shifts alone (pad 4, crop, ±4 px) are competitive without auxiliary losses, and DrQ also averages Q-targets over augmentations.
- DrAC shows that naively augmenting PPO's buffer changes the objective and can "completely fail". Its fix regularizes the policy and value toward invariance. Shifts also move the absolute screen edges, which matter for clamping, so skip augmentation in v0.

**SAC+AE (Yarats et al. 2021) [20]**
- Joint β-VAE + SAC training was unstable because of "the stochastic nature of a β-VAE and the non-stationary gradient from the actor".
- The fix is a deterministic regularized autoencoder (latent L2 λz=1e-6, decoder weight decay λθ=1e-7), with the encoder updated only by the reconstruction and critic gradients, never the actor's. Pixel decoders can work jointly, *if gradient routing is controlled*.

**ATC (Stooke et al. 2021) [21]**
- An encoder trained *only* by an unsupervised temporal-contrast loss "matches or outperforms end-to-end RL in most environments": all of DMControl and DMLab, and 5 of 8 Atari games. It uses PPO on Atari and DMLab and does better in sparse-reward settings. This supports a decoupled arm.

**PPG [22]; IDAAC [23]**
- With shared networks, "the optimization of one objective will interfere with the optimization of the other". The value function tolerates more sample reuse, and PPG distills value features into the policy in a separate auxiliary phase with a KL-to-old-policy term. Sharing one representation between policy and value can also cause overfitting [23].

**When auxiliary prediction helps or hurts**
- *Helps:* sparse reward or small data (UNREAL; auxiliary depth and loop-closure prediction [24]; self-supervised pretraining gave a "1.4× speed-up on average" [25]); memory transformers under PPO (reconstruction was key for TrXL [80]). Pixel-prediction features also probe better than VAE or random features [28].
- *Hurts or limited:* reconstruction spends capacity on irrelevant pixels [13], pixel-space forward models make poor curiosity features [55], joint stochastic autoencoders are unstable [20], shared trunks suffer interference [22], and naive augmentation fails [19]. Benefits depend on reward density, horizon and distractors [27].
- *Tool:* gate or weight the auxiliary loss by the cosine similarity between its gradient and the main gradient [26].

## 3. Learning what the agent controls (contingency / controllability)

**Contingent regions (Bellemare et al. 2012) [29]**
- Definition: contingent regions are the pixels whose next value differs for some pair of actions. They are computed exactly with emulator resets (counterfactual rollouts), then learned with a logistic regressor and used as features, tested on 46 held-out games. Our deterministic, resettable simulator gives these ground-truth maps for free (§10, P4).

**Contingency-aware exploration (Choi et al. 2019) [30]**
- Method: an attentive dynamics model (ADM), i.e. an inverse model that predicts a_{t−1} from (s_{t−1}, s_t) using per-cell logits on feature differences and a softmax/sparsemax spatial attention α, plus an entropy regularizer on α. The argmax of α gives the agent's location, which drives count-based exploration. It scores more than 11,000 on Montezuma's Revenge with no demonstrations or RAM.
- **Metric:** mean Euclidean distance between the argmax-attention cell and the RAM ground-truth location on a 9×9 grid, "typically ... 1 or 2 grid cells away". Tiny sprites are hard to see at 84×84, which argues for a cursor of at least 5 px.

**Independently controllable features (Thomas, Bengio et al.) [31–33]**
- Each feature f_k is paired with a policy π_k, and *selectivity* sel(s,a,k) = E_{s′}[|f_k(s′)−f_k(s)| / Σ_{k′}|f_{k′}(s′)−f_{k′}(s)|] is maximized together with an autoencoder loss. It disentangles controllable factors with no extrinsic reward (toy domains). It can serve as a metric: actions should move cursor features but not target features.

**ICM (Pathak et al. 2017) [34]; poking (Agrawal et al. 2016) [35]**
- ICM learns features through an inverse model, so they keep "only the information relevant to the action", and predicts forward in that space (β=0.2 forward vs inverse, λ=0.1 policy). In "very sparse" VizDoom, A3C and ICM-pixels never succeed, while ICM reaches a perfect score in 66% of runs. ICM-pixels also degrades with input noise.
- Agrawal et al. jointly train forward and inverse models on 400 h and 100k pokes. "The forward model ... regularize[s] the feature space for the inverse model." An inverse head is a cheap regularizer for us.

**AC-State (Lamb et al. 2023) [36]**
- A multi-step inverse model (k ~ U(1, K)) plus an information bottleneck provably recovers the control-endogenous state and discards exogenous noise. The paper gives a counterexample where one-step inverse models fail, and demonstrates the method on a robot arm with distractors. If we add distractors, use multi-step inverse probes.

**Empowerment [37–39]**
- Empowerment is the channel capacity from actions to future sensor states [37]. Mohamed & Rezende give a scalable variational bound [38], and VIC maximizes the mutual information between options and final states [39]. [our analysis] A cursor's empowerment drops at edges and corners because of clamping, so a model-estimated empowerment map is another controllability readout.

**Controllability metrics from world models [40–42]**
- Genie: Δ_tPSNR = PSNR(x_t, x̂_t) − PSNR(x_t, x̂′_t), comparing generation from inferred versus random actions (reported at t=4) [40].
- CAI: state-dependent conditional mutual information between the action and an entity's next state [41].
- C-SWM: evaluates latent dynamics with ranking metrics (H@1, MRR) instead of pixel losses [42].

## 4. World models and latent dynamics

- **Ha & Schmidhuber 2018 [43]:** a VAE (V), an MDN-RNN (M) and a linear controller (C, 867 parameters, trained with CMA-ES), learned from random-rollout data; CarRacing score 906±21. Policies trained "in the dream" find "adversarial" exploits of model flaws, which the MDN temperature τ mitigates. The lesson is that random babbling data suffices to learn V and M in simple worlds, and that planners will exploit model errors (relevant to Phase 5).
- **PlaNet [44]:** a recurrent state-space model with deterministic and stochastic paths ("purely stochastic transitions make it difficult ... to remember information"), latent overshooting and CEM planning on 64×64 images. It used about 200× less interaction than A3C.
- **Dreamer V1–V3 [45–47]:** actor-critic in latent imagination (V1), discrete latents for Atari (V2), and in V3 a single configuration across 150+ tasks and Minecraft diamonds from scratch (published in Nature in 2025). V3's robustness tricks are symlog targets, "free bits" (KL clipped below 1 nat), a 1% uniform mix in categoricals, and returns divided by max(1, P95−P5). That last rule only scales large returns down and never scales small returns up, which is useful under sparse reward.
- **SimPLe [48]:** an action-conditioned video model with PPO trained inside it (γ=0.95), on Atari-100k. A per-pixel clipped loss max(L, C) (C=10 for L2) was "crucial": it removes gradients from large background areas so the model can "concentrate on small but important areas (e.g. the ball in Pong)". This applies directly to a small cursor.
- **Collapse (brief) [49–53, 15]:**
  - BYOL uses an EMA target and a predictor; SimSiam shows stop-gradient suffices; Tian et al. analyze non-contrastive dynamics; VICReg adds variance and covariance terms.
  - Ni et al. 2024: latent self-prediction alone "is trivial and can be achieved by employing a constant representation". Stop-gradient or EMA targets give non-collapse guarantees that online targets lack.
  - For v1: use an EMA or stop-gradient target, keep the pixel and drive losses as grounding, and monitor the per-dimension std of z and the probe scores.

## 5. Intrinsic motivation and homeostasis

**Curiosity**
- **RND [54]:** prediction error against a fixed random network, which is deterministic and so avoids the noisy-TV attraction. Intrinsic returns are non-episodic and extrinsic returns episodic, with two value heads. Intrinsic reward is divided by a running std of intrinsic returns. Observations are whitened and clipped to [−5, 5] for the predictor and target networks only.
- **Large-scale curiosity [55]:** 54 environments with no extrinsic reward. Random features are often enough, and inverse-dynamics features generalize better. Forward dynamics on raw pixels "does not work well across any environment". The noisy TV is demonstrated with a TV plus a "change channel" action in a maze. The "done" signal leaks reward information, so it was removed for pure curiosity.
- **Noisy-TV mitigations:** ensemble disagreement [56] and aleatoric-uncertainty estimation [57].

**Learning progress**
- Schmidhuber 1991/2010 [58,59] treats curiosity as model improvement, later formalized as compression progress.
- Oudeyer et al. 2007 (IAC) [60]: intrinsic reward is the decrease in prediction error within regions of similar sensorimotor situations. The robot "will not stay in front of white noise for a long time", and stage-like development emerged on a real robot.
- Kim et al. 2020 [61]: γ-Progress is the loss of an old (slow EMA) model minus the loss of the new model. It overcomes the "white noise problem" and beats RND and disagreement in their setting.
- **Dark-room problem [62]:** minimizing surprise can favor static, predictable states. This is the formal version of the plan's "do not reward prediction accuracy".

**Homeostatic and drive-based RL**
- **Keramati & Gutkin 2014 [63]:** drive D(H_t) = (Σ_i |h*_i − h_{i,t}|^n)^{m/n}, and reward is the drive reduction r_t = D(H_t) − D(H_{t+1}). Maximizing discounted reward is then equivalent to minimizing discounted drive, which **requires γ<1**. The agent's state spans both internal and external variables.
- **Yoshida et al. [64–66]:** a deep-RL "neural homeostat" with continuous motor control, where drive-difference rewards worked best among the definitions compared [64]. A 2024 Neural Networks follow-up exists [65] (I could not access the abstract, so its content is [unverified]). A 2025 HRRL perspective reports risk aversion and anticipatory regulation [66].
- **Singh et al. 2010 [67]:** the optimal-reward framework, in which intrinsic rewards emerge as the rewards that maximize fitness across environments. This supports treating drives as designable reward and observation channels.

**Active inference (brief) [68–70]**
- The free-energy principle [68] and active inference as a process theory [69] cast goal-seeking as minimizing expected free energy, which includes an epistemic term; an RL implementation is in [70]. In that framing, hunger setpoints become prior preferences. This is not recommended for v0, but it matches the D(H) view above.

## 6. Developmental and sensorimotor learning (brief)

- **O'Regan & Noë 2001 [71]:** seeing is the exercise of mastery over sensorimotor contingencies, the lawful ways sensation changes with action. Our counterfactual sweeps (§10, P3) operationalize this.
- **Philipona et al. 2003 [72]:** an agent can infer the dimensionality of space from sensorimotor dependencies, with no prior model of its body [recollection of content].
- **Goal babbling, Rolf et al. 2010 [73]:** samples are generated along paths toward goals using the current inverse estimate, which learns inverse kinematics of redundant systems directly. Random motor babbling scales poorly with motor dimension [recollection/secondary]. SAGG-RIAC adds goal selection by learning progress [74].
- **Rovee & Rovee 1969 [75]:** infants kick more when kicking moves an overhead mobile ("conjugate reinforcement"), a classic contingency-learning paradigm [content recollection].
- Relevance: our action space has only three dimensions, so Gaussian/Bernoulli motor babbling covers it well. Goal babbling only matters for richer effectors.

## 7. Transformers in on-policy RL

- **GTrXL [76]:**
  - It combines "identity map reordering" (pre-LayerNorm) with GRU-type gates in place of residual connections, using gate bias b_g=2 (a "gated identity initialization" that greatly speeds learning).
  - At initialization the agent is effectively Markov: "an agent needs to learn how to walk before it can learn how to remember".
  - It matches LSTMs on reactive DMLab levels, beats them on memory levels, and is less sensitive to seeds and hyperparameters (trained with V-MPO).
- **POPGym [77]:** 13 memory models, PPO only. GRU had the best aggregate score (MMER 0.349, vs LSTM 0.255 and MLP 0.067), and "classical RNNs outperformed modern memory methods". On several games, memory models did not beat the MLP.
- **Ni et al. [78,79]:** tuned recurrent model-free RL is a strong POMDP baseline. Transformers improve memory (up to 1,500 steps back) but "do not improve long-term credit assignment".
- **Memory Gym [80]:**
  - TrXL+PPO uses a sliding window over cached per-layer hidden states, with **no gradient through the episodic memory**.
  - GRU beats TrXL on endless tasks. TrXL wins on finite tasks only with an observation-reconstruction auxiliary loss.
  - A ground-truth-estimation head serves as a diagnostic.
- **Recurrent PPO bookkeeping [81]:** reset memory at episode end, use sequential per-environment minibatches, and reconstruct the memory state from what was saved at rollout start.
- **Relevance:** our task is nearly fully observable from one frame plus the drives. Memory matters mainly for in-context inference of the action mapping (§10, P9). Start Markov, make the transformer start near-identity, and treat memory as a hypothesis to test.

## 8. PPO implementation practice

**The 37 Implementation Details of PPO (Huang et al. 2022) [81]**
- **Core details:**
  - Orthogonal init (√2 for hidden layers, 0.01 for the policy head, 1.0 for the value head, zero biases); Adam ε=1e-5; linear learning-rate annealing (Atari 2.5e-4, MuJoCo 3e-4); GAE; advantages normalized *per minibatch*.
  - Value-loss clipping is implemented, but there is "no evidence that [it] helps" [83] and it "even hurts" [82].
  - Global gradient-norm clip 0.5. Log approx_kl, clipfrac and explained variance.
- **Pixels:** divide by 255, because "without it, the first policy update results in the Kullback–Leibler divergence explosion". On Atari, policy and value share a CNN trunk.
- **Continuous actions:**
  - Gaussian with a **state-independent log-std initialized to 0**, and independent components (probabilities multiply).
  - Clip actions to the valid range **but store the unclipped sample**.
  - Observation normalization clipped to [−10, 10]; reward scaled by the running std of the discounted return (no mean subtraction) and clipped to [−10, 10].
- **Default configs:**
  - Atari: 128 steps, 4 minibatches, 4 epochs, γ .99, λ .95, clip 0.1, entropy 0.01.
  - MuJoCo: 2048 steps, 32 minibatches, 10 epochs, clip 0.2, entropy 0, separate networks.
  - **Procgen (64×64 RGB, the closest to our setting) [86]:** 64 envs × 256 steps, 8 minibatches, 3 epochs, γ .999, λ .95, lr 5e-4, clip 0.2, entropy 0.01, value coefficient 0.5, gradient clip 0.5, IMPALA-CNN [16, 32, 32] [87], reward normalization only, 25M steps.

**What Matters in On-Policy RL (Andrychowicz et al. 2021) [82]** (dense-reward MuJoCo; transfer with care)
- **Policy loss:** use PPO with a clip threshold starting at 0.25.
- **Policy head:**
  - Initialize the last policy layer 100× smaller.
  - Produce the std through a softplus with an offset; the **initial std ≈0.5 was best on 4 of 5 tasks**, and "performance is very sensitive to the initial action standard deviation".
  - tanh squashing beat clipping; tanh also "bounds the maximum possible policy entropy".
- **Value network:** separate and wider. Always normalize observations.
- **Advantages and value loss:** GAE with λ=0.9, and **neither Huber loss nor value clipping**.
- **Optimization:** start at γ=0.99 and tune. Adam with β1=0.9 and lr 3e-4. Entropy and KL regularizers did not help; gradient clipping helps slightly. Recompute advantages on every pass over the data.

**Other practice papers**
- **Engstrom et al. 2020 [83]:** code-level optimizations "are responsible for most of PPO's gain in cumulative reward over TRPO".
- **Hybrid Gaussian + Bernoulli actions:** factorize π = π_c·π_d with separate KL bounds for each part [88], or, as in H-PPO, "update separately ... their respective clipped surrogate objective" with separate ratios r^d and r^c [89].
- **Bounded actions:** Beta policies avoid the boundary bias of clipped Gaussians [90]. CAPG gives an unbiased, lower-variance gradient for clipped actions [91].
- **Time limits:** at a time-limit truncation, bootstrap from the value (partial-episode bootstrapping), or include remaining time in the observation [92].
- **Statistics:** report interquartile means with stratified-bootstrap confidence intervals, plus performance profiles and probability of improvement. These stay reliable with few runs [93].

## 9. Implications for our design (in priority order)

**9.1 Base rates [our sim]**

Assumptions follow the `circle_world` defaults: a 64×64 field, the circle fully inside it, the cursor starting outside the circle, move-then-hit-test, and dx, dy = 4·clip(N(0, σ²), −1, 1) with clamping. Each row uses 20k episodes, and success means clicking while inside the circle. The untrained `dev/` policy has σ = e^−0.5 ≈ 0.61 and p_click ≈ 0.5. `random_baseline.py` can confirm these numbers.

| Radius | T | Policy | Random success | Median steps |
|---|---|---|---|---|
| 12–16 (default) | 200 (default) | σ=0.61, p=0.5 (untrained) | **0.60** | 56 |
| 12–16 | 200 | σ=1.0, p=0.5 | 0.73 | 51 |
| 12–16 | 200 | dx, dy ~ U[−1,1], p=0.5 | 0.63 | 57 |
| 12–16 | 100 | σ=0.61, p=0.5 | 0.42 | 32 |
| 4 | 200 | σ=0.61, p=0.5 | 0.24 | 77 |
| 3 | 100 | σ=0.61, p=0.1 | **0.045** | 49 |

An oracle (move straight at 4 px/step, then click) needs a median of 5 steps (p95 10) at radius 12–16, and 8 steps at radius 4. A random p=0.5 policy makes about 30 clicks per successful 200-step episode.

Pixel statistics (using a bright square as a stand-in for the crosshair; `train.py`'s `copy_changed_mse` gives the real values): a 3, 5 or 7 px cursor changes 0.6 / 1.3 / 1.9% of pixels per step at 64² (1.0 / 1.7 / 2.6% at 32²). The copy-last-frame MSE is about 0.003 / 0.006 / 0.010 on a [0, 1] scale.

What follows from these numbers:
1. The "first success" milestone is trivial, and success rate at v0 will saturate. Make the **primary metrics steps-to-success (about 56 → 5), click precision P(inside | click), and click recall P(click | inside)**.
2. Exploration and drive claims need a *hard* condition where random success is 2–10% (for example radius 3–4, T=100, low initial p_click).
3. Keep `cursor_start_outside=True` (the default); inside starts would add about 15% free successes.
4. Clicks are free, so **always-clicking is optimal**. Either add a tiny cost per empty click after the bonus budget runs out (−0.01 to −0.05 before scaling), or drop "clicks per success" as a measure of skill.

**9.2 PPO settings: literature vs `dev/` defaults (2026-10-23 snapshot)**

| Knob | Literature start | `dev/` default | Verdict |
|---|---|---|---|
| Envs × rollout | 8×128 (Atari), 64×256 (Procgen) [81,86] | 128×128 (16k per update) | fine; very large batches can cost sample efficiency [82] |
| Epochs / minibatches | 3–4 / 4–8 | 4 / 4 | ✓ |
| γ, λ, clip ε | 0.99; 0.95 (0.9 in [82]); 0.2 (0.25 in [82]) | 0.99, 0.95, 0.2 | ✓ |
| Adam | lr 2.5e-4 to 5e-4, ε=1e-5, linear decay [81] | 2.5e-4, ε=1e-5, annealed | ✓ |
| Gradient clip; value loss | 0.5; coefficient 0.5, no value clipping [82,83] | 0.5; 0.5; `clip_vloss=0` | ✓ |
| Reward scale | fixed and identical across arms (running-std scaling would reweight the +10 differently with and without bonuses) | `reward_scale=0.1` | ✓ keep it fixed |
| Advantages | GAE + per-minibatch normalization [81] | `norm_adv=1` | add a scale floor for the hard condition (9.7) |
| Entropy | discrete 0.01, continuous 0 [81,82] | one `ent_coef=0.01` on the Gaussian + Bernoulli sum | split it (9.3) |
| Log-std | state-independent; init 0 [81] or σ≈0.5 [82] | init −0.5, clamp [−5, 1] | consider an upper clamp of 0 |
| Timeouts | bootstrap V at truncation [92] | `done` (success or timeout) is terminal in GAE | bootstrap timeouts (9.7) |

**9.3 Action head**
- Already matches the literature: a tanh-bounded mean, a state-independent log-std, the stored *unclipped* sample used for log-probabilities [81], the clamped executed action fed to the predictor, and the summed joint log-probability [81,88].
- [our analysis] With clipping, the Gaussian part of the entropy bonus pushes log σ up by a constant per step, while the extra spread only produces more saturated, bang-bang moves. Under Adam that constant push drifts log σ by up to about lr per optimizer step whenever it outweighs the policy gradient, however small `ent_coef` is.
  - Fix: apply `ent_coef` to the Bernoulli term only (or a separate, smaller movement coefficient), and/or lower `LS_MAX` from 1 to about 0 (σ ≤ 1). Alternatively use a tanh-squashed Gaussian with the log-det correction, since tanh "bounds the maximum possible policy entropy" [82].
  - Watch `std_x`.
- Initial σ: [82] favors 0.5. A larger σ covers the screen faster (in our sim, random success at T=200 is 0.73 with σ=1 vs 0.60 with σ=0.61), so `logstd_init` between −0.5 and 0 is reasonable.
- Log KL and clipfrac per component. If the click component dominates clipping, use separate clipped surrogates for movement and click [89].

**9.4 Encoder and memory**
- Use a cursor of at least 5×5 px with a distinctive color. Avoid a stride-4 first convolution; use a stride-2 stack (64→32→16→8) or the IMPALA-CNN [86].
- Optional priors that leak no environment state, to run as ablations: CoordConv channels (CNNs "fail spectacularly" at coordinate transforms) [94] and spatial-softmax keypoints [95,96].
- **Run a Markov baseline (context 1, or CNN+MLP) first** [76,77]. Make the transformer start near-identity (gated residual branches with b_g≈2, or zero-initialized residual branches) [76].
- The `dev/` "context = block" design restarts the context at every episode start and every 32 steps, so acting and training are bitwise identical. That avoids both the state-reconstruction bookkeeping of [81] and the stale cached memory of [80].
  - The costs: memory never spans more than 32 steps, it resets mid-episode, and the first step of each block is Markov.
  - Evaluate P9 (in-context adaptation) *within* blocks, and log performance by position within the block.
- The previous executed action is an input. Make it an ablation, since WoB found past-action inputs caused "meander[ing]" [1].

**9.5 Prediction head**
- Already in place and supported by the literature:
  - `pred_changed_weight`, which up-weights pixels that change between consecutive observed frames ([13,48]).
  - Pre-reset `terminal_image`/`terminal_drives` as targets (SPR also truncates at boundaries [15]).
  - The circle vanishing on success, which gives clicks a visible consequence.
- Next options:
  - Predict the *residual* from the downsampled current frame, so the default output is "copy". The decoder currently outputs a full sigmoid frame.
  - Condition multiplicatively (factored or FiLM) on a_t [14].
  - Try SimPLe's per-pixel clipping max(L, C) as an alternative to the changed-pixel weight [48].
  - Oversample rare hunger-relief transitions for the drive head, as UNREAL did for rewarding events [13].
  - Add an inverse-dynamics head [34,35].

**9.6 Gradient routing and experimental arms**
- `pred_backbone_scale` already implements β, the gradient scale between the trunk and the prediction head. Arms:
  - β=0: the head is trained on detached features and measures what RL alone encodes.
  - β=0.1 and β=1.
  - **Decoupled:** the trunk is trained only by the prediction and inverse losses, and policy/value read stop-gradient features [21].
- Adam's per-parameter scaling makes loss weights hard to interpret. Log the gradient norm and cosine similarity of each loss at the trunk [26]; aim initially for ‖∇L_pred‖/‖∇L_PPO‖ between 0.1 and 1.
- Also note that the global gradient clip is shared by all losses.
- If approx_kl spikes when prediction is on, lower β or move the prediction updates into a PPG-style phase [22].
- Add a "babble then act" comparison: reward-free pretraining followed by PPO, versus PPO from scratch [21,25].

**9.7 Failure modes: symptom → mitigation**
- **Auxiliary loss dominates:** the trunk gradient-norm ratio is ≫1, KL spikes, and the arm learns slower than the no-prediction arm → lower β, or switch to the decoupled or PPG-style setup.
- **Copy-frame minimum:** skill on changed pixels is about 0 → residual target, changed-pixel weighting, loss clipping.
- **Action ignored:** ΔPSNR between true and shuffled actions is about 0 (§10, P2) → multiplicative conditioning.
- **Always-click:** p_click → 1 → click cost or refractory period, and report precision.
- **Stillness:** mean |Δcursor| → 0 → no movement entropy, but keep a σ floor. Ensure no reward term correlates with inaction, and never reward prediction accuracy (dark room [62]).
- **σ blow-up:** more than 30% of samples clipped → clamp log σ or use tanh.
- **Edge hugging:** the cursor histogram piles up on the borders → bounded mean; reward only actual displacement (already in the plan).
- **Reset leakage:** prediction-loss spikes at done steps → mask them.
- **Reward aliasing:** low explained variance because bonuses depend on hidden counters → make restlessness the observable budget state (rewards should depend only on observed drives, as in [63]).
- **Timeouts treated as terminal:** GAE zeroes the bootstrap on every `done`. About 40% of untrained episodes time out, and no-drive arms cannot see time [92] → on `timeout & ~success`, bootstrap from V of the terminal observation (or of the last in-episode state). At minimum, log explained variance near t=200. [82] found special handling unnecessary only for *long* time limits.
- **Hard-setting advantage noise:** batches with no successes still get unit-variance advantages → floor the normalization scale, max(1, S) as in DreamerV3 [47].
- **Bonus farming:** keep the plan's bounded budgets. The success → reset transition is "noisy-TV-like" [55], so exclude it from any future intrinsic reward.

**9.8 Drives and statistics**
- Hunger rises steadily, so it acts as a partial clock (time-awareness [92]). Initial drives are randomized, though, so still bootstrap at truncation.
- For a homeostatic arm, use r = D(H_t) − D(H_{t+1}) with a convex D in place of the +10, not in addition to it [63,64].
- For a curiosity follow-up, use learning progress [60,61], not raw error.
- Use ≥5 seeds and report IQM with 95% stratified-bootstrap confidence intervals and performance profiles [93].

## 10. How to demonstrate learned dynamics without reward

Already in `dev/`:
- `analyze_dynamics.py` covers parts of P3 (action retrieval, displacement error), P5 (ridge probes plus a random-init control) and P7 (circle vanishing and hunger drop, inside vs outside).
- `train.py` logs `copy_changed_mse` for P1.

Still missing are P2, P4, P6, P8 and P9, plus AUROC and confidence intervals.

- **P0. Reward-free regime** (the `P_babble*` configs). Train the trunk and the prediction, drive and inverse heads on data from a fixed babbling policy with no PPO and no reward, as in Ha & Schmidhuber's random rollouts and Agrawal's random pokes [43,35]. Run P1–P9 on held-out seeds. Repeat on PPO checkpoints with β=0 to show what RL alone encodes.
- **P1. Skill against trivial baselines.** Compute on reset-masked held-out transitions:
  - Metrics: MSE over all pixels, MSE on the changed-pixel mask M, and MSE in the cursor box (ground truth used for evaluation only).
  - Baselines: copy-last-frame; the mean frame; an action-blind twin trained with a_t zeroed (the naFf control [14]); and the renderer itself as the oracle.
  - Report skill S = 1 − MSE/MSE_copy on M. Copy gets S=0 and a perfect model S=1. Global MSE alone says little, since copy already reaches about 0.003–0.010.
- **P2. Action dependence (Genie-style [40]).** ΔPSNR = PSNR(I_{t+1}, Î(z_t, a_t)) − PSNR(I_{t+1}, Î(z_t, ã)), with ã taken from another time step. Compute it on all pixels and on M. A value of 0 means the model ignores its action.
- **P3. Counterfactual action sweeps.** For a held-out state, decode Î(z_t, a) over a 9×9 grid of (dx, dy) × click ∈ {0, 1}. The simulator renders the true counterfactual frame I^a_{t+1} from the same state. Report:
  - (i) counterfactual MSE against the truth;
  - (ii) predicted cursor displacement d̂(a), taken as the centroid of cursor-colored mass in Î(a) or of |Î(a)−Î(0)|, regressed on the commanded 4a: the gain (ideal 1.0), R² and angular error;
  - (iii) saturation of d̂ at walls.
  - A quiver plot of d̂(a) is the clearest single figure for "it knows what its actions do".
- **P4. Controllable-entity localization.**
  - The model's contingency map is C_model(x) = Var_a[Î(x | z_t, a)]. The ground truth is C_true(x) = 1[∃ a, a′: I^a_{t+1}(x) ≠ I^{a′}_{t+1}(x)], computed by counterfactual rendering [29].
  - Report the distance in px from argmax C_model to the cursor center (Choi et al. reported 1–2 cells on a 9×9 grid [30]), the AUROC of C_model against C_true, and the contingency mass on the red circle (should be about 0).
  - **Distractor variant:** add a cursor-like sprite that does its own random walk. Error maps will flag both sprites, but the contingency map should flag only the real cursor [34,36]. An ADM-style attention inverse model gives a second localizer [30].
- **P5. Linear probes [28].** Freeze the model and collect CNN and transformer states on held-out episodes, split by episode.
  - Fit ridge or logistic probes for: cursor (x, y), target (x, y), the target−cursor vector, the inside flag, distance and ΔH.
  - Report R², MAE in px and AUROC.
  - Controls: the same network at random init (a random CNN reached 0.44 mean F1 in Atari, against 0.58 for pixel prediction and 0.68 for ST-DIM [28]), a linear probe on raw pixels, the β=0 trunk, and the majority class. Add an MLP probe to separate "linearly accessible" from "present".
- **P6. Inverse-dynamics probe.** Fit a small head on frozen (z_t, z_{t+1}) to regress the executed (dx, dy), and report R² (VPT's IDM reaches 0.97 on mouse movement [8]). Stratify out wall-clamped steps, where the action cannot be identified. Clicks can only be recovered if they are visible, so consider rendering a 1-frame click ring: it is feedback about the agent's own action, not target information. Add a multi-step variant with k ~ U(1, K) [36].
- **P7. Interoceptive consequences.** Compute s = ΔĤ(click=1) − ΔĤ(click=0) on held-out states. Report the AUROC of s for "cursor inside circle", and plot E[s] against the cursor−target offset; it should trace a disk of radius r. If this comes out of P0, it shows "clicking the red disc relieves hunger" was learned without any reward. Build the same offset map for the predicted fraction of the circle that vanishes after a click (`vanish_*` currently gives only inside/outside averages). Also check restlessness predictions against actual displacement.
- **P8. Model-only control.** With no policy learning, pick actions by random shooting or CEM over imagined rollouts (re-encoding predicted frames) that maximize predicted hunger relief. The 1→3→5-step curriculum [14] helps, and beware of exploits [43]. Compare success and steps against the random policy and PPO.
- **P9. Contingency remapping.** Rotate or flip the action→motion mapping per episode (0°, 90°, 180°, or swapped axes). Plot P1 and P3 against steps since reset. A memory model should adapt within a few steps and a Markov model cannot. This tests O'Regan & Noë's "mastery" [71] and is the cleanest justification for the transformer [79].

## References

1. Shi, Karpathy, Fan, Hernandez, Liang (2017). World of Bits: An Open-Domain Platform for Web-Based Agents. ICML, PMLR 70:3135–3144. https://proceedings.mlr.press/v70/shi17a.html
2. Liu, Guu, Pasupat, Shi, Liang (2018). Reinforcement Learning on Web Interfaces Using Workflow-Guided Exploration. ICLR. https://arxiv.org/abs/1802.08802
3. Jia, Kiros, Ba (2019). DOM-Q-NET: Grounded RL on Structured Language. ICLR. https://arxiv.org/abs/1902.07257
4. Gur, Rueckert, Faust, Hakkani-Tur (2019). Learning to Navigate the Web. ICLR. https://arxiv.org/abs/1812.09195
5. Humphreys, Raposo, Pohlen, Thornton, Chhaparia, Muldal, Abramson, Georgiev, Goldin, Santoro, Lillicrap (2022). A data-driven approach for learning to control computers. ICML, PMLR 162. https://arxiv.org/abs/2202.08137
6. Toyama, Hamel, Gergely, Comanici, Glaese, Ahmed, Jackson, Mourad, Precup (2021). AndroidEnv: A Reinforcement Learning Platform for Android. arXiv technical report. https://arxiv.org/abs/2105.13231
7. Shaw, Joshi, Cohan, Berant, Pasupat, Hu, Khandelwal, Lee, Toutanova (2023). From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces. NeurIPS. https://arxiv.org/abs/2306.00245
8. Baker, Akkaya, Zhokhov, Huizinga, Tang, Ecoffet, Houghton, Sampedro, Clune (2022). Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos. NeurIPS. https://arxiv.org/abs/2206.11795
9. Tassa, Doron, Muldal, Erez, Li, de Las Casas, Budden, Abdolmaleki, Merel, Lefrancq, Lillicrap, Riedmiller (2018). DeepMind Control Suite. arXiv. https://arxiv.org/abs/1801.00690
10. Kostrikov, Yarats, Fergus (2021). Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels. ICLR. https://arxiv.org/abs/2004.13649
11. Ikkala, Fischer, Klar, Bachinski, Fleig, Howes, Hämäläinen, Müller, Murray-Smith, Oulasvirta (2022). Breathing Life Into Biomechanical User Models. UIST. https://doi.org/10.1145/3526113.3545689
12. Selder, Fischer, Kristensson, Fleig (2025). What Makes a Model Breathe? Understanding RL Reward Function Design in Biomechanical User Simulation. CHI Late-Breaking Work. https://arxiv.org/abs/2503.02571
13. Jaderberg, Mnih, Czarnecki, Schaul, Leibo, Silver, Kavukcuoglu (2017). Reinforcement Learning with Unsupervised Auxiliary Tasks. ICLR. https://arxiv.org/abs/1611.05397
14. Oh, Guo, Lee, Lewis, Singh (2015). Action-Conditional Video Prediction using Deep Networks in Atari Games. NeurIPS. https://arxiv.org/abs/1507.08750
15. Schwarzer, Anand, Goel, Hjelm, Courville, Bachman (2021). Data-Efficient Reinforcement Learning with Self-Predictive Representations. ICLR. https://arxiv.org/abs/2007.05929
16. Laskin, Srinivas, Abbeel (2020). CURL: Contrastive Unsupervised Representations for Reinforcement Learning. ICML, PMLR 119:5639–5650. https://arxiv.org/abs/2004.04136
17. Laskin, Lee, Stooke, Pinto, Abbeel, Srinivas (2020). Reinforcement Learning with Augmented Data. NeurIPS. https://arxiv.org/abs/2004.14990
18. Yarats, Fergus, Lazaric, Pinto (2022). Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning. ICLR. https://arxiv.org/abs/2107.09645
19. Raileanu, Goldstein, Yarats, Kostrikov, Fergus (2021). Automatic Data Augmentation for Generalization in Reinforcement Learning. NeurIPS. https://arxiv.org/abs/2006.12862
20. Yarats, Zhang, Kostrikov, Amos, Pineau, Fergus (2021). Improving Sample Efficiency in Model-Free Reinforcement Learning from Images. AAAI. https://arxiv.org/abs/1910.01741
21. Stooke, Lee, Abbeel, Laskin (2021). Decoupling Representation Learning from Reinforcement Learning. ICML. https://arxiv.org/abs/2009.08319
22. Cobbe, Hilton, Klimov, Schulman (2021). Phasic Policy Gradient. ICML. https://arxiv.org/abs/2009.04416
23. Raileanu, Fergus (2021). Decoupling Value and Policy for Generalization in Reinforcement Learning. ICML. https://arxiv.org/abs/2102.10330
24. Mirowski, Pascanu, Viola, Soyer, Ballard, Banino, Denil, Goroshin, Sifre, Kavukcuoglu, Kumaran, Hadsell (2017). Learning to Navigate in Complex Environments. ICLR. https://arxiv.org/abs/1611.03673
25. Shelhamer, Mahmoudieh, Argus, Darrell (2017). Loss is its own Reward: Self-Supervision for Reinforcement Learning. Listed under ICLR 2017 on mlanthology; track [unverified]. https://arxiv.org/abs/1612.07307
26. Du, Czarnecki, Jayakumar, Farajtabar, Pascanu, Lakshminarayanan (2018). Adapting Auxiliary Losses Using Gradient Similarity. arXiv; venue [unverified]. https://arxiv.org/abs/1812.02224
27. Tomar, Mishra, Zhang, Taylor (2023). Learning Representations for Pixel-based Control: What Matters and Why? TMLR. https://arxiv.org/abs/2111.07775
28. Anand, Racah, Ozair, Bengio, Côté, Hjelm (2019). Unsupervised State Representation Learning in Atari. NeurIPS. https://arxiv.org/abs/1906.08226
29. Bellemare, Veness, Bowling (2012). Investigating Contingency Awareness Using Atari 2600 Games. AAAI 26(1):864–871. https://doi.org/10.1609/aaai.v26i1.8321
30. Choi, Guo, Moczulski, Oh, Wu, Norouzi, Lee (2019). Contingency-Aware Exploration in Reinforcement Learning. ICLR. https://arxiv.org/abs/1811.01483
31. Thomas, Pondard, Bengio, Sarfati, Beaudoin, Meurs, Pineau, Precup, Bengio (2017). Independently Controllable Factors. arXiv. https://arxiv.org/abs/1708.01289
32. Bengio, Thomas, Pineau, Precup, Bengio (2017). Independently Controllable Features. arXiv (RLDM submission). https://arxiv.org/abs/1703.07718
33. Thomas, Bengio, Fedus, Pondard, Beaudoin, Larochelle, Pineau, Precup, Bengio (2018). Disentangling the independently controllable factors of variation by interacting with the world. NIPS 2017 Workshop. https://arxiv.org/abs/1802.09484
34. Pathak, Agrawal, Efros, Darrell (2017). Curiosity-driven Exploration by Self-supervised Prediction. ICML. https://arxiv.org/abs/1705.05363
35. Agrawal, Nair, Abbeel, Malik, Levine (2016). Learning to Poke by Poking: Experiential Learning of Intuitive Physics. NeurIPS. https://arxiv.org/abs/1606.07419
36. Lamb, Islam, Efroni, Didolkar, Misra, Foster, Molu, Chari, Krishnamurthy, Langford (2023). Guaranteed Discovery of Control-Endogenous Latent States with Multi-Step Inverse Models. TMLR. https://arxiv.org/abs/2207.08229
37. Klyubin, Polani, Nehaniv (2005). Empowerment: A Universal Agent-Centric Measure of Control. IEEE CEC, 1:128–135. https://doi.org/10.1109/CEC.2005.1554676
38. Mohamed, Rezende (2015). Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning. NeurIPS. https://arxiv.org/abs/1509.08731
39. Gregor, Rezende, Wierstra (2016). Variational Intrinsic Control. Listed under ICLR 2017 on mlanthology; track [unverified]. https://arxiv.org/abs/1611.07507
40. Bruce, Dennis, Edwards, Parker-Holder, Shi, Hughes, et al. (2024). Genie: Generative Interactive Environments. ICML, PMLR 235. https://arxiv.org/abs/2402.15391
41. Seitzer, Schölkopf, Martius (2021). Causal Influence Detection for Improving Efficiency in Reinforcement Learning. NeurIPS. https://arxiv.org/abs/2106.03443
42. Kipf, van der Pol, Welling (2020). Contrastive Learning of Structured World Models. ICLR. https://arxiv.org/abs/1911.12247
43. Ha, Schmidhuber (2018). Recurrent World Models Facilitate Policy Evolution. NeurIPS (arXiv "World Models": https://arxiv.org/abs/1803.10122). https://arxiv.org/abs/1809.01999
44. Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, Davidson (2019). Learning Latent Dynamics for Planning from Pixels. ICML, PMLR 97. https://arxiv.org/abs/1811.04551
45. Hafner, Lillicrap, Ba, Norouzi (2020). Dream to Control: Learning Behaviors by Latent Imagination. ICLR. https://arxiv.org/abs/1912.01603
46. Hafner, Lillicrap, Norouzi, Ba (2021). Mastering Atari with Discrete World Models. ICLR. https://arxiv.org/abs/2010.02193
47. Hafner, Pasukonis, Ba, Lillicrap (2025). Mastering diverse control tasks through world models. Nature 640:647–653. https://doi.org/10.1038/s41586-025-08744-2 (preprint: https://arxiv.org/abs/2301.04104)
48. Kaiser, Babaeizadeh, Milos, Osinski, Campbell, Czechowski, Erhan, Finn, Kozakowski, Levine, Mohiuddin, Sepassi, Tucker, Michalewski (2020). Model-Based Reinforcement Learning for Atari. ICLR. https://arxiv.org/abs/1903.00374
49. Grill, Strub, Altché, Tallec, Richemond, et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. NeurIPS. https://arxiv.org/abs/2006.07733
50. Chen, He (2021). Exploring Simple Siamese Representation Learning. CVPR. https://arxiv.org/abs/2011.10566
51. Tian, Chen, Ganguli (2021). Understanding Self-Supervised Learning Dynamics without Contrastive Pairs. ICML. https://arxiv.org/abs/2102.06810
52. Bardes, Ponce, LeCun (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR. https://arxiv.org/abs/2105.04906
53. Ni, Eysenbach, Seyedsalehi, Ma, Gehring, Mahajan, Bacon (2024). Bridging State and History Representations: Understanding Self-Predictive RL. ICLR. https://arxiv.org/abs/2401.08898
54. Burda, Edwards, Storkey, Klimov (2019). Exploration by Random Network Distillation. ICLR. https://arxiv.org/abs/1810.12894
55. Burda, Edwards, Pathak, Storkey, Darrell, Efros (2019). Large-Scale Study of Curiosity-Driven Learning. ICLR. https://arxiv.org/abs/1808.04355
56. Pathak, Gandhi, Gupta (2019). Self-Supervised Exploration via Disagreement. ICML. https://arxiv.org/abs/1906.04161
57. Mavor-Parker, Young, Barry, Griffin (2022). How to Stay Curious while Avoiding Noisy TVs using Aleatoric Uncertainty Estimation. ICML. https://arxiv.org/abs/2102.04399
58. Schmidhuber (1991). Curious model-building control systems. IJCNN, vol. 2:1458–1463. https://doi.org/10.1109/IJCNN.1991.170605
59. Schmidhuber (2010). Formal Theory of Creativity, Fun, and Intrinsic Motivation (1990–2010). IEEE TAMD 2(3):230–247. https://doi.org/10.1109/TAMD.2010.2056368
60. Oudeyer, Kaplan, Hafner (2007). Intrinsic Motivation Systems for Autonomous Mental Development. IEEE Trans. Evol. Comput. 11(2):265–286. https://doi.org/10.1109/TEVC.2006.890271
61. Kim, Sano, De Freitas, Haber, Yamins (2020). Active World Model Learning with Progress Curiosity. ICML. https://arxiv.org/abs/2007.07853
62. Friston, Thornton, Clark (2012). Free-Energy Minimization and the Dark-Room Problem. Frontiers in Psychology 3:130. https://doi.org/10.3389/fpsyg.2012.00130
63. Keramati, Gutkin (2014). Homeostatic reinforcement learning for integrating reward collection and physiological stability. eLife 3:e04811. https://doi.org/10.7554/eLife.04811
64. Yoshida, Daikoku, Nagai, Kuniyoshi (2021). Embodiment Perspective of Reward Definition for Behavioural Homeostasis. NeurIPS Deep RL Workshop. https://mlanthology.org/neuripsw/2021/yoshida2021neuripsw-embodiment/
65. Yoshida, Daikoku, Nagai, Kuniyoshi (2024). Emergence of integrated behaviors through direct optimization for homeostasis. Neural Networks 177:106379. https://doi.org/10.1016/j.neunet.2024.106379
66. Yoshida, Sprekeler, Gutkin (2025). Linking Homeostasis to Reinforcement Learning: Internal State Control of Motivated Behavior. arXiv. https://arxiv.org/abs/2507.04998
67. Singh, Lewis, Barto, Sorg (2010). Intrinsically Motivated Reinforcement Learning: An Evolutionary Perspective. IEEE TAMD 2(2):70–82. https://doi.org/10.1109/TAMD.2010.2051031
68. Friston (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience 11(2):127–138. https://doi.org/10.1038/nrn2787
69. Friston, FitzGerald, Rigoli, Schwartenbeck, Pezzulo (2017). Active Inference: A Process Theory. Neural Computation 29(1):1–49. https://doi.org/10.1162/NECO_a_00912
70. Tschantz, Millidge, Seth, Buckley (2020). Reinforcement Learning through Active Inference. arXiv. https://arxiv.org/abs/2002.12636
71. O'Regan, Noë (2001). A sensorimotor account of vision and visual consciousness. Behavioral and Brain Sciences 24(5):939–973. https://doi.org/10.1017/S0140525X01000115
72. Philipona, O'Regan, Nadal (2003). Is There Something Out There? Inferring Space from Sensorimotor Dependencies. Neural Computation 15(9):2029–2049. https://doi.org/10.1162/089976603322297278
73. Rolf, Steil, Gienger (2010). Goal Babbling Permits Direct Learning of Inverse Kinematics. IEEE TAMD 2(3):216–229. https://doi.org/10.1109/TAMD.2010.2062511
74. Baranes, Oudeyer (2013). Active learning of inverse models with intrinsically motivated goal exploration in robots. Robotics and Autonomous Systems 61(1):49–73. https://doi.org/10.1016/j.robot.2012.05.008
75. Rovee, Rovee (1969). Conjugate reinforcement of infant exploratory behavior. Journal of Experimental Child Psychology 8(1):33–39. https://doi.org/10.1016/0022-0965(69)90025-3
76. Parisotto, Song, Rae, Pascanu, Gulcehre, Jayakumar, Jaderberg, Lopez Kaufman, Clark, Noury, Botvinick, Heess, Hadsell (2020). Stabilizing Transformers for Reinforcement Learning. ICML, PMLR 119:7487–7498. https://arxiv.org/abs/1910.06764
77. Morad, Kortvelesy, Bettini, Liwicki, Prorok (2023). POPGym: Benchmarking Partially Observable Reinforcement Learning. ICLR. https://arxiv.org/abs/2303.01859
78. Ni, Eysenbach, Salakhutdinov (2022). Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs. ICML. https://arxiv.org/abs/2110.05038
79. Ni, Ma, Eysenbach, Bacon (2023). When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment. NeurIPS. https://arxiv.org/abs/2307.03864
80. Pleines, Pallasch, Zimmer, Preuss (2025). Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents. JMLR 26:1–40. https://arxiv.org/abs/2309.17207
81. Huang, Dossa, Raffin, Kanervisto, Wang (2022). The 37 Implementation Details of Proximal Policy Optimization. ICLR Blog Track. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
82. Andrychowicz, Raichuk, Stańczyk, Orsini, Girgin, Marinier, Hussenot, Geist, Pietquin, Michalski, Gelly, Bachem (2021). What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study. ICLR (arXiv title: "What Matters In On-Policy Reinforcement Learning?"). https://arxiv.org/abs/2006.05990
83. Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. ICLR. https://arxiv.org/abs/2005.12729
84. Schulman, Wolski, Dhariwal, Radford, Klimov (2017). Proximal Policy Optimization Algorithms. arXiv. https://arxiv.org/abs/1707.06347
85. Schulman, Moritz, Levine, Jordan, Abbeel (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR. https://arxiv.org/abs/1506.02438
86. Cobbe, Hesse, Hilton, Schulman (2020). Leveraging Procedural Generation to Benchmark Reinforcement Learning. ICML, PMLR 119:2048–2056. https://arxiv.org/abs/1912.01588
87. Espeholt, Soyer, Munos, Simonyan, Mnih, et al. (2018). IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. ICML, PMLR 80. https://arxiv.org/abs/1802.01561
88. Neunert, Abdolmaleki, Wulfmeier, Lampe, Springenberg, Hafner, Romano, Buchli, Heess, Riedmiller (2019). Continuous-Discrete Reinforcement Learning for Hybrid Control in Robotics. CoRL. https://arxiv.org/abs/2001.00449
89. Fan, Su, Zhang, Yu (2019). Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space. IJCAI, 2279–2285. https://doi.org/10.24963/ijcai.2019/316
90. Chou, Maturana, Scherer (2017). Improving Stochastic Policy Gradients in Continuous Control with Deep RL using the Beta Distribution. ICML, PMLR 70:834–843. https://proceedings.mlr.press/v70/chou17a.html
91. Fujita, Maeda (2018). Clipped Action Policy Gradient. ICML, PMLR 80. https://arxiv.org/abs/1802.07564
92. Pardo, Tavakoli, Levdik, Kormushev (2018). Time Limits in Reinforcement Learning. ICML, PMLR 80:4042–4051. https://arxiv.org/abs/1712.00378
93. Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS. https://arxiv.org/abs/2108.13264
94. Liu, Lehman, Molino, Such, Frank, Sergeev, Yosinski (2018). An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution. NeurIPS. https://arxiv.org/abs/1807.03247
95. Levine, Finn, Darrell, Abbeel (2016). End-to-End Training of Deep Visuomotor Policies. JMLR 17(39):1–40. https://arxiv.org/abs/1504.00702
96. Finn, Tan, Duan, Darrell, Levine, Abbeel (2016). Deep Spatial Autoencoders for Visuomotor Learning. ICRA. https://arxiv.org/abs/1509.06113
