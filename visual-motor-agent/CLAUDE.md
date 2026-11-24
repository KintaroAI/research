# Visual–Motor Agent

A small, randomly initialized agent learns to move a cursor and click a red circle from pixels only.
The question: what does it learn about its own actions without being reinforced for it, and does
that help it learn the task? Plan: `../visual_motor_learning_plan.md`.

## Repository Structure

- **`dev/`**: active development (work here)
  - `src/vm_agent.cpp`, `src/vm_agent.h`: the model, in C++. Covers the CNN encoder, causal
    Transformer, policy/value/prediction heads, manual backward, acting with KV cache, rollout
    memory, GAE, PPO, AdamW and checkpoints. Built as `libvmagent.so`.
  - `agent_lib.py`: the ctypes wrapper (numpy pointers in, no copies).
  - `world/`: the Python world. `circle_world.py` is the vectorized numpy simulator; `record.py`
    records episodes; `browser_demo/` is the HTML/JS replica (play + replay) and is checked
    pixel-exact against Python by `check_js_equivalence.py`.
  - `train.py`: PPO loop (Python world + C++ agent). `run_queue.py`: job queue with 2 slots.
    `evaluate.py`: task metrics.
  - `analyze_dynamics.py`: counterfactual tests of the learned dynamics (plus
    `dynamics_over_training.py` and `compare_dynamics.py`). `plot_runs.py`: learning curves.
    `record_episodes.py`: GIFs, replay JSON and imagination strips. `make_video.py`: MP4s and
    screenshot sheets across checkpoints (needs dense checkpoints via `train.milestones`).
    For the universal model: **`dev/ANALYSIS.md`** — how to run lifetimes and the frozen tests, the dream curve,
    `tables.py` (README tables), `forage_video.py` / `dream_video.py` / `dream_measures.py`, and what repeats.
  - `plan.py`: CEM planner on the latent dynamics (acting without reward). `analyze_imagination.py`:
    imagined vs real rollouts, read out by retrieval. `analyze_planner.py`: where the planner fails.
    `train.py --set train.actor=planner`: the model learns from the planner's own experience.
  - `online.py`: one lifetime in one world, with a gradient step after every step from its own
    (replayed) experience. `model.one_frame=1`: each prediction uses only the current frame.
    `online.memory=gated` (vm-00007): 8-step chunks enter long-term memory by relief and surprise
    (`GatedMemory`). The default `stream` is the plain ring buffer, unchanged.
    `analyze_hazards.py` (vm-00012): whether a planner agent's hunger model imagines the hazards' cost; `contact`
    also probes patch-token models (per kind of token, and the token of the patch under the cursor).
    `analyze_self_actor.py`: why the self actor does or does not aim (aim, probe, relief, inputs,
    momentum; `--device cuda` runs its forward passes on the GPU, 30× faster). `ppo.pi_detach=1`: the move loss
    trains the policy head only (vm-00008);
    `model.pi_hidden`: a hidden layer in the policy head.
  - `model.patch=16` (vm-00010): patch tokens instead of the CNN. Each step is 16 patch tokens plus one
    token each for drives, previous move and command; each predicts its own part of the next state.
    `env.distractors=K`: grey dots that move on their own (separate random streams: the episodes stay
    the same). vm-00011: `env.respawn` (with `end_on_success=false`, a large `max_steps`: a world that never
    resets), `env.target_speed`, `env.hazards`, `env.wrap` (a torus) and `env.egocentric` (the view centered
    on the cursor). In a continual world with walls the basic agent starves at a wall: use `wrap`.
    Hunger is capped at 1: at the cap a hazard step leaves no trace; use slow hunger (`env.h_rate=0.001`, vm-00015).
    `env.hazard_click=1` (vm-00017): a click inside a hazard sets hunger to 1 (poison); `analyze_hazards.py click`
    tests whether a model predicts what each kind of click does (placed one-step windows); `click-real` (task 00027,
    universal model) tests it on real windows: the frozen agent lives as in the forage test, each chosen move is tried
    with and without a click on copies of the world, and the prediction is read from its last 1, 2, 4 and ctx steps.
    `env.touch` (task 00042, needs `respawn`): food and poison act on contact, no click, and a touched poison is
    replaced away from the cursor (`configs/U_touch.json`, smaller food; a random walk touches 4.7 food and 11.8 poison
    per 1k; `forage_chance.py`: blind walks, food : poison at most 0.78; vm-00035). `configs/U_touch_ego.json` adds
    `env.egocentric` (vm-00036): ~2× the food and less poison in every arm.
  - **Hunger ahead** (task 00082, vm-00071): `online.future_k = k`, the hunger a move is scored on is the mean over the next k ticks, labelled in hindsight; k = 4 gives about 1.7 times the food with no more poison, k = 16 stops seeing poison; refused with self-forcing.
  - **A bootstrapped value** (task 00096, vm-00085): `online.value_gamma = g` (with `future_k`), the label is (1 − g) Σ_{i<n} g^i h_{j+i} + g^n vh_{j+n}: n = `online.value_n` (default future_k) real ticks, then the model's own predicted label of the move taken at tick j + n (memory field `vh`, stored while acting: the transformer whose move it was). The newest window ends n ticks back. `eval_hunger.py value`: predicted against realized label, frozen. `tests/test_value_labels.py`. `online.value_refresh = 1` (task 00098, vm-00087): the tail made fresh before every update, the mean over the transformers of `forward_uni(..., next_body=True)` on the windows n steps later (`boot_values` in `lifetime.py`); memory hands out `pr_real`, `tw` and `b_img`/`b_drv`/`b_act`/`b_es` (`GatedMemory.keys`), long-term memory keeps n steps past each chunk (`ChunkStore(extra=n)`).
  - **The meal label** (task 00108, vm-00097): `online.value_target = meal` (default `hunger`, unchanged), with `value_refresh = 1`, `value_n = 1` and `value_gamma` (0.95 there).
    - A step whose hunger falls is a meal, V = 1. Any other step is worth minus the hunger it added, plus γ × its transformer's own fresh prediction for the next step (`online.meal_parts`, `meal_target`; `boot_values(mean=False)`; each transformer's label is written into the rollout before its update).
    - Clipped to [−1, 1] and stored as T = (1 − V)/2 in the hunger channel, so the instinct is unchanged.
    - `online.need = 1`: the instinct's tolerance divided by max(hunger, `NEED_FLOOR` .05), the label weighted by hunger when choosing (also in `eval_hunger.Life` and `forage_video`).
    - `online.MealMonitor` logs each life's predicted against real V by ticks to the next meal (metrics `meal`, bins `MEAL_BIN_NAMES`). `eval_hunger.py value` gives meal runs the realized meal label and a profile by ticks to the next real meal (row field `meal`).
    - `tests/test_meal_label.py`.
  - **The label as a body value, stored once** (task 00109, vm-00098; the owner: re-labeling at every replay is a cheat):
    - `model.body_label = 1`: the body carries a third value, the label, with its own channel (`uni_chan` needed; `VC` one larger, the move's channel after it; header slot 62; refused with `state`). Its input is blank (0.5). The move token predicts it, with its target from the frame's prev_reward slot (the GPU uploads that slot only with a body label). `next_body` is 3 wide, and `uni_candidates` returns the label, so the instinct reads it; hunger stays plain hunger.
    - `online.label = meal` (`label_gamma`; only with `body_label`, without `value_gamma`, `future_k`, `need`, `state`): after the move is chosen each transformer scores the move made (memory field `vhs`). One tick later step t's label is written once (`GatedMemory.write_label`, `online.stored_labels`, field `lab`): a meal 1, otherwise γ × that transformer's prediction for the next step's move, less the hunger a poison added (a rise above `POISON_RISE` .01; no drift cost). It is never re-labeled. Each transformer trains on its own labels, written into the rollout's prev_reward slot before its update.
    - Metrics `meal` (predictions) and `meal_stored` (the labels in memory) against what happened, by ticks to the next meal.
    - `online.full_dose = 1` (task 00110, vm-00099; only with `label = meal`): a poison costs its full dose at any hunger. The world reports the hunger the clip at 1 cut off (`info["hunger_cut"]`, 0 below the cap, `CircleWorld` only), memory keeps it (field `cut`), and the stored label, the monitors and `eval_hunger.py value` count it in the rise. What the model senses is unchanged. `tests/test_full_dose.py`.
    - `online.label_refresh = 1` (task 00112, vm-00100; only with `label = meal` and several transformers): the body label re-labeled at every update instead of stored once. `GatedMemory` keeps one step past each chunk and gives the label's parts (`meal_parts(..., drift=False, cut=...)`, the stored label's rule) and the bootstrap windows; `lifetime.py` makes each transformer's labels from its fresh prediction for the next move (`boot_values(..., chan=2)`, the label channel) and trains it on them. The stored labels are still written (monitor `meal_stored`) but not trained on. `tests/test_label_refresh.py`.
    - `online.instinct_temp = T` (task 00113, vm-00101; default 1): the instinct's candidates after the first drawn at temperature T (p^(1/T) per value, extra random numbers); the first stays the model's own move at temperature 1, so T = 1 repeats earlier runs exactly. Passed by `lifetime.py`, `eval_hunger.Life` (and so `eval_candidates.py`) and `forage_video.py` (and so `compare_video.py`); older tools of other worlds use 1. `tests/test_instinct_temp.py`.
    - `tests/test_body_label.py`, `tests/test_stored_label.py`.
    - **Result (vm-00098):**
      - Without inertia it works: the front moves back as designed, the predictions end unbiased, and S0 eats 1.4 times C8r's food (0.80 of M0's), still rising at 60k.
      - With inertia SB never forages (6.4 food per 1k). A third of the lives starve at the hunger cap with the instinct silent. Frozen, the candidates' labels are practically equal (spread .0001 or less). One push moves the next position by at most 1 px, and at the cap a poison costs nothing.
      - **The instinct's candidates are samples of the model's own move prediction** (`uni_act`). A model sure of its move proposes the same move 16 times: in SB's starving seeds all 16 are one move on .41–.79 of ticks, and the instinct has no choice. `eval_candidates.py` measures it, frozen: distinct moves, the share of ticks with all the same move, and the label's spread.
  - **Several foods** (task 00083, vm-00072): `env.foods = F`, still circles beside the food; `target` is always the nearest, so the step and the tools work on it. `configs/U_big.json`: 128 x 128 with 4 times as much of everything.
  - **Clogged channels** (task 00085, vm-00074): `online.clog = p`, `online.clog_long`, each transformer its own damaged input, black or speckled, whole window or one step, against clean targets. `lifetime.py` hands it to `update_rollout(inputs=...)`. `online.clog_act = 1` (task 00086, vm-00075): each transformer also acts on its own damaged view, memory stays clean. Both off by default. How it all works: ARCHITECTURE.md section 14.
  - **White noise** (task 00087, vm-00076): `env.noise = q`, each pixel of each frame replaced by a random brightness with probability q, fresh each tick, in the world: input, targets and memory all carry it. `env.clean_frame()` for scoring only. `eval_noise.py`: any model at any test level. `online.noise_in = q` (task 00088, vm-00077): the same noise on the model's input only (acting, and fresh at each update, one draw for all transformers), the world, memory and targets clean; needs the ring, not with `online.clog`.
  - **Hidden food** (task 00092, vm-00081): `env.food_show = v`, the food drawn only in its first v frames after it appears (touching it still feeds; 0 = always, the old world); `food_age`, `drawn()`, `info["gt_food_seen"]` only when v > 0. `eval_hidden.py`: aim by ticks since the food vanished, walkers (blind, forgetful, perfect).
  - **World schedule** (task 00094, vm-00083): `online.env_sched = {"key": [[tick, value], ...]}` changes world settings mid-life through `env.reconfigure`, before that tick's move; only keys in the world's `RUNTIME` (CircleWorld: `food_show`, `inertia`, `inertia_drag`); refused otherwise at the start; metrics carry `env_<key>`. The saved `env_config` is the starting world; the frozen tests (`eval_hunger.Life`) and videos (`forage_video`, `compare_video`) take `inertia` and `inertia_drag` at their last scheduled value (`lifetime.env_at_end`), other keys as the run began. With `online.mem_score` the blind world follows the schedule too.
  - **A world larger than the view** (task 00102, vm-00091): `env.view = V` (needs `egocentric`; even, at most `size`, `(size − V) / 2` a multiple of `2 · hazard_checker_cell`), the frame is the V x V center of the full egocentric frame; food, poison and hunger are the world's. `CircleWorld.foods_in_view()` (scoring only). `configs/U_view.json` (128 world, 2 foods, view 64, `img` 64). `eval_hunger.py chain` then scores aim at unseen food by ticks since food was last in view, against a blind baseline, also without the ticks after a meal (`cos_out_nomeal`: a just-eaten food is out of view too, which doubled the first figure). `eval_food_gap.py --view`. `compare_video.py` draws the whole world beside each view (`--cols 2`). Task 00103 (vm-00092): `chain`'s out-of-view bins are 1–8, 9–16, 17–32, 33–64 and 65+ ticks (each row names them in `since_names`; vm-00091's rows print under 1–8, 9–32, 33+), and `show --base ARM` pairs the memory measure by bin; a context of 32 gave no memory past 8 ticks and half the food.
  - **A state carried across windows** (task 00104, vm-00093): `model.state = 1`, two more tokens per step after the body, a write (no values, no target; its expected output values, 64 numbers from the shared head's patch channel, are the state) and a read (the state written ctx steps earlier, given from outside: `CudaModel.set_uni_state` or `forward_uni(state=...)`; none given: blank, 0.5, as the C++ model always reads). Acting: `CudaModel` keeps each body's last ctx writes in its ring (`st_q`; `last_read` for memory; `state_blank` the ablation). Training (`lifetime.py`): memory chunks of 2 ctx steps (field `st`, half precision; `b_st` for the refreshed value), the batch as pairs of consecutive windows, `CudaModel.update_pairs`: the second window reads the first's fresh writes, and `online.state_grad = 1` passes its loss back through them (`vmc_set_uni_carry`, dE/dlogit = p (c - E)); 0 reads them as given. `eval_hunger.py --state blank` (rows ARM-blank_sN) or `--state shuffle` (body b reads body b - 1's note: the usual values without the information; the right ablation for a learned input, vm-00093: a blank read is unfamiliar input). `tests/test_state.py`. Needs the GPU and the ring; refused with self-forcing, clog, noise_in, off, delays, env.thought and surprise gating.
  - **The flexible layer** (task 00105, vm-00094): `online.fast = λ` (0: off) and `online.fast_k`: the parameters hold W = S + F. S takes the usual AdamW step on the batch; then F ← λ F − k · lr · clamp(g / √v̂, −1, 1), g the newest window's gradient alone and v̂ the slow update's Adam second moment (`CudaModel.newest_fast`, `vmc_fast_step`); the parameters change by F's change, so S is untouched. `ckpt_final_*_fast.npy` holds F (S = the checkpoint − F); metrics `fast_ratio`, `fast_ratio_max` (F against S). At k = 10 F sat at a third of S, so vm-00094 runs k = 1. `online.mem_score = 1` (needs `env.view`): the chain's memory measure scored in life (`eval_hunger.MemoryScore`, the same code as `chain`) against a blind world stepped with the same moves; each metrics row carries `mem`, its sums and counts by bin since the last row. A frozen test cannot see F. `tests/test_fast.py`.
  - **Inertia** (task 00106, vm-00095): `env.inertia = k` (0: the old world, unchanged), the move is a push: vel ← clip(vel × (1 − `env.inertia_drag`) + move × max_step_px / k, ±max_step_px per axis), the cursor moves by vel; with no push the body keeps moving (k ticks from rest to full speed, 2k to reverse; a wall takes the blocked axis's velocity). `CircleWorld.vel` is the last displacement in either mode (state key `vel` when inertia > 0), `info["gt_vel"]` (scoring only) when inertia > 0. Both keys are in `RUNTIME`, so `online.env_sched` can switch inertia on mid-life (the body keeps moving as it last moved). `CircleWorld.push_of(moves)` is where moves would take the body from the current state (`step()` moves by it); scoring code that asks where a move lands must use it (`eval_hunger.would_touch`, `in_reach`, `eval_pieces.touches_poison` do), not `move * max_step_px`. `eval_hunger.py --env inertia=4` tests any run in its world with a setting changed (rows `ARM-inertia4_sN`), `--step N` its checkpoint at tick N (rows `ARM-stepN_sN`); `eval_hunger.py move`: speed, top-speed share, braking, and speed and aim (of the move and of the motion) by the gap to food and poison. Under inertia `chain`'s aim scores the push, not the motion. `compare_video.py RUN@30000+inertia=4:LABEL` gives each panel its own checkpoint and world settings (labels without ':'). The JS replica (`browser_demo/`) does not implement inertia. `tests/test_inertia.py`.
  - **Delayed copies** (task 00091, vm-00080): `env.delays = [0, d1, d2, d3]` (`world/delay_world.py`): the frame is a 2 x 2 composite of the view now and d ticks ago, twice the world's size on each side; in training the composite is the input and the present, tiled, the target (`online.delay_targets`). Memory is sized by the model's frame (`model.img`), not `env.size`. `eval_delay.py`: error by copy, chain with the past blanked; `eval_hunger.Life(view=...)`.
  - **Forecast panels** (task 00115, vm-00102): `online.panels = 1` (`world/panel_world.py`, `PanelWorld`): with `model.img` twice `env.view` and the four quarter transformers, the frame is a 2 x 2 composite: panel 1 the view sensed `online.panel_lag` ticks late (drives, terminal drives and `hunger_cut` delayed with it), panels 2–4 transformers 0–2's forecasts of their own panel from the tick before (`forward_uni_step`, `panel_decode` pair or mean, Gaussian `panel_noise`; `panel_blank = 1`: noise instead, the baseline). Panel i is trained against the real frame i ticks after panel 1's (`online.panel_targets`, memory field `f_img`, `GatedMemory(fut=4)`); the composites are the inputs (`update_rollout(inputs=...)`). Strips `panels_TICK.png` every `panel_snap` ticks; `online.panel_record = N` saves body 0's composites of the first N ticks (`panel_frames.npy`) for `panel_video.py` (the panels as seen beside the real frames they aim at). `lifetime.py --init STEM` loads the pieces' `STEM_pK.bin` (with `ppo.lr = 0`: the frozen test). `tests/test_panels.py`. vm-00102: frozen, live forecasts predict panels 2–4 better than noise (−.027 nats) and lived poison is a quarter lower, but training batches show the opposite (memory replays stale forecasts), and the 128 × 128 frame forages worse than TT's 64 × 64. `readme_images.py` skips runs with `ppo.lr = 0` (frozen tests) in the curves.
  - **Outage** (task 00089, vm-00078): `online.off = q`, `online.off_span`: each transformer off at random, its own chain; no forward, move or update, and no catching up. The ring's skipped ticks (`uni_ring_skip`, seen flags per slot, ages in ticks) and a training step mask (`set_step_mask`, `update_rollout(step_mask=...)`: a missed step is no key and has no loss); memory keeps the on-mask per tick (`Replay(pieces=True)`, field `on`). `eval_outage.py`: chain at test off shares, restore, alone.
  - **Two senses** (task 00081, vm-00070): `env.sense` (`world/sense_world.py`, `configs/U_sense.json`): a view and a text sense in one wide frame, a scripted teacher, each sense going off by itself; `eval_sense.py` scores a sense's return against `sense.cut`, `eval_steer.py` who steers with the eyes closed.
  - **One half teaches another** (task 00078, vm-00068): `reference/mimic.py` (PyTorch, one letter per tick; arms Z, N, A, B, S, E): half 2 learns only from half 1's distribution and is heard where words are missing.
  - **Two stacked boxes, backprop cut between them** (task 00045, vm-00038): `reference/stacked_local.py` (PyTorch,
    text world; arms e2e / frozen / local / dtp / dni, `--tie`, `--cut-norm`), `reference/stacked_summary.py`.
  - **Attention in regions** (task 00044, vm-00037; `vm_attn_allow` in `vm_agent.h`, `torch_reference.attn_allow`):
    `model.attn_split` S bands of the patch grid's rows (`attn_cols=1`: columns), each reaching `attn_overlap` rows
    past its edges; body and move in every region, `attn_pred_global=1` also the last patch (it predicts the move). A
    mask (weights 0), not a compute saving. Prediction survives halves; the move's predictor must see the whole view.
    `attn_checker=1` (task 00046, vm-00039): regions by (row + column) % S, a checkerboard; no overlap or columns.
    Task 00047, vm-00040: `attn_win=W` (patches attend within W rows and columns) and `uni_sum=R` (a summary token after
    every R rows, no values; the token before it has no target). **The slot layout is no longer [body, patches, move]:**
    use `agent_lib.uni_layout(cfg)` (patch slots, summary slots, move) in any tool; a patch is predicted by the token
    before its slot (`layout["patch"] - 1`), the move by slot nt - 2, the next body by the move token.
    Task 00048, vm-00041: `uni_pyr=L`, a pyramid of summaries (one per 2^l x 2^l block, l = 1..L); the layout is a table
    built once (`vm_uni_layout`, a device array in CUDA). Task 00049: `sum_thumb=1`, summaries carry their block's
    thumbnail as values (then every token has a target). Task 00052: the 2-D position codes added to `uni_slot` at birth
    follow the slot table (`vm_uni_center`: a patch its cell, a summary its block's centre); before, slot k + 1 got cell
    k's code, which scrambled every summary model of vm-00040–00044 (`tests/test_slot_positions.py`). Any new slot kind
    or layout must place its code there too. Task 00054 (vm-00046): `sum_only=1`, the patches are not tokens (a step is [body, summaries,
    move]); `configs/T_text_B2.json` is the text world at twice the scale (a B1 thumbnail is one letter). Task 00055
    (vm-00047): `attn_split` with `uni_pyr`, halves that share the summaries (a patch reads its band's patches and every
    summary). Task 00057 (vm-00049): `band_n` / `band_k` / `band_cols`, a model that is one of S separate transformers (only its band's patches, every summary; each band its own lifetime on the same text; `letter_ce.py --bands` scores a pair). Task 00058 (vm-00050): `band_win` (only the summaries near the piece at each level), `band_cols = 2` (square blocks). Task 00059 (vm-00051): `online.pieces = 1` runs all pieces on one world and one body, the move by `online.piece_rule` (`pieces_move` in `lifetime.py`; gain is the one that works), checkpoints `ckpt_NNNNNNNN_pK.bin`; `eval_pieces.py` is the frozen test (aim, wins, `--touch`). Task 00060 (vm-00052): `env.no_overlap`, `online.piece_rule=council`. Task 00072 (vm-00063): `piece_rule` also `gain_sharp`, `soft` (`online.piece_temp`), `pessimism` (`online.piece_lambda`), `rank` (ties at random) and `rank_first` (a tie between sums goes to the lowest piece); `council` has the best poison on one build, and win shares under a scoring rule are mostly bookkeeping (ties among identical moves credit the lowest piece). Task 00061 (vm-00053): `env.turn_cost`. Task 00062: `model.age_key` (`wpe` holds keys read at attention by age, no step embedding in tokens; `vm_attn_*`, `k_attn_age`) and `online.ring` (`CudaModel.uni_ring_*`: acting computes only the new step's tokens on a key and value buffer; `tests/test_ring.py` pins it to a streaming torch version). Task 00064 (vm-00056): `model.img_w` (frames wider than high), `env.thought` (`world/thought_world.py`, `configs/U_thought.json`), `eval_thought.py`. Task 00065 (vm-00057): `env.cue` (`world/cue_world.py`, `configs/U_cue.json`), the delayed-cue world; `online.thought_decode` (`pair`, or `mean`, which stops the black and white collapse; thought still drifts dark, and `thought_extreme` does not see it). Task 00067 (vm-00059): `env.kind = car` (`world/car_world.py`, `configs/U_car.json`), the default model as a differential-drive car in MuJoCo, with move value 0 turning and value 1 driving; `car_chance.py` (blind drivers), `car_video.py` (the camera and a view from above); task 00068 (vm-00060): `env.dense`, `dense_hold`, `dense_end`, a dense start thinned over the life (`configs/U_car_dense.json`); task 00071 (vm-00062): the car only goes forward by default (`env.reverse = true` for the old car, `--reverse` in `car_chance.py` and `car_video.py`), and vm-00061 and later use 5 ticks/s (`env.frame_skip = 40`); set `MUJOCO_GL=egl` before the first import of mujoco, and pick the card with `MUJOCO_EGL_DEVICE_ID`. Many lifetimes on one GPU: start MPS and export its pipe (see below); it does not survive a reboot. Compare text runs of different layouts with `letter_ce.py` (next-letter cross-entropy), at their best Tasks 00073, 00074 (vm-00064, vm-00065): `model.band_n=16` and `64` with no code change; 64 pieces take about 2.8 GB per lifetime, so at most 8 per GPU. Task 00075 (vm-00066): `model.uni_chan`, on in `U_default.json` since 2026-11-01 (970,402 parameters per piece; runs before that date had it off), a channel per value: VC = V + 5 positions in `W_in`, `W_out` and the bias; value tensors stay V wide, `k_uni_pick` maps the head's rows. Task 00076 (vm-00067): `online.tie_tables` (`in`, `inout`), `cuda_lib.tie`, the mean of the pieces' tables after every update.
    checkpoint (~30k; text lifetimes overfit after it). GPU buffers (`Buf`) free themselves: never list them in
    `vmc_free` again (that list leaked 14 GB in the test file).
  - **Breakout** (task 00111, vm-00103): `env.kind = breakout` (`world/breakout_world.py`, `configs/U_breakout.json`: `U_agent.json` with only the `env` block replaced): a 64 x 64 court, the move's dx pushes a paddle, a hit is a meal and a miss a poison (with `hunger_cut`), restlessness as `CircleWorld`'s; `info["gt_ball"]` etc. for scoring only. `breakout_walkers.py` (random, still, tracker), `online.frame_record = N` (body 0's first N frames in any world, `frames.npy`) and `frames_video.py`. vm-00103: the default agent unchanged learns to play, hit share .24 (random) to .68 lived at 50–60k, .64 frozen. `readme_images.py` plots a Breakout run's hit share and hits.
    - **Stage 2** (task 00117, vm-00104; `configs/U_breakout2.json`): `paddle_relief` (a paddle hit's food; 0 here), `bricks = 1` (4 x 6 bricks of 3 looks, solid, checker and hollow, feeding `brick_food` .1, .25, .5 by rank; `brick_shift` rotates which look feeds how much, the Makefile sets seed mod 3), a cleared wall comes back with no slot keeping its type (a derangement), the serve from below the bricks; `hunger_cut` is signed (negative where the floor at 0 cut a meal short; earlier label code adds it only on steps whose hunger did not fall). Stats `bo_bricks`, `bo_brick_t0..2`, `bo_food`, `bo_clears`, `bo_pref_hit`, `bo_pref_stand` (the preference: the rank hit minus the standing mean, per hit). `breakout_walkers.py` adds the aimer (the tracker meeting the ball off-centre by a fresh offset each return: the chance level of the preference; the plain tracker's ball goes straight up one column). vm-00104: with a ball at 2.25 px and food only from bricks the default agent barely plays (28% of balls at 50–60k, random 23%) and never prefers better bricks.
    - **A meal worth its size** (task 00117): `online.meal_size = M` (with `label = meal` and `full_dose`; 0 off): a meal is a step whose full change (rise + signed cut) is negative, worth min(1, relief / M), in `meal_parts`, `stored_labels` and `MealMonitor`. `tests/test_meal_size.py`. vm-00104: +.036 paddle hit share over the plain label, no taste for bricks.
    - **Paddle food** (task 00119, vm-00107; `configs/U_breakout2p.json`, stage 2 with `paddle_relief` .25): play comes back, PK (default) .43 and PS (`meal_size` .5) .47 hit share at 50–60k against random .23 (vm-00104 .28, .33), still rising at 60k, frozen at 60k about .45; still no preference for better bricks: the return is one tick from its meal, the aim 15+ ticks from the brick's. vm-00109: one PS life of 1M ticks returns .87 at 250k and .95 at 1M (frozen .95), plays like the aimer, still no taste; label, hunger and memory stable.
  - **The blind spot** (task 00118): `online.blind = [x, y, w, h]`, those pixels black in everything the model sees (acting, update inputs, the label's bootstrap windows; the same frame for every transformer), memory and targets whole; `online.pred_record = N` saves body 0's predicted current frame (each piece its block, decoded pair) and the true one (`pred_frames.npy`, `pred_truth.npy`); `eval_blind.py` scores the gap (error, background and previous-frame baselines, objects by ticks since entry, hits and false alarms, the surround; `--video`). `online.blind_target` (vm-00106): `black` (the gap in the targets too) or `motion` (`GapTeacher`: each step's gap filled from a canvas of the world painted only from the visible part, the body's position from the world as a training signal; memory field `timg`). `tests/test_blind.py`. vm-00105: trained against the whole frame, the default agent fills the gap (1.56 grey levels off at 60k; empty gap 5.85, seen 1.22), holds objects that slid in for 15+ ticks, and does worse when the gap is opened (4.21). vm-00106: with the gap black in the targets too it predicts black (G0, 27.0); the motion teacher teaches it to fill (GB 3.67 against the empty guess's 6.00, 12/12 seeds; about half of BL's gain).
  - **The default agent (since 2026-11-22, the owner; task 00113): `configs/U_agent.json`**, vm-00101's TT: `U_view.json`'s world with inertia 4, C8's model (4 quarter transformers of 8 layers, tolerance .002, context 8, the ring, gain), RD's label (`body_label`, `label = meal`, `label_gamma` .95, `full_dose`, `label_refresh`) and `instinct_temp = 2`. New experiments in the food worlds start from it; the code defaults and the older configs are unchanged, so earlier Makefiles still run as they did.
  - **Learned baselines** (task 00120, vm-00108): `dev/baselines/` — `vm_env.py` (a one-body gym-style wrapper around `make_world`: the grey frame, drives and previous move; reward = the meal label's immediate part, +1 a meal, minus a poison's full dose; no episode end; `VecVMEnv` for PPO's bodies; `tests/test_baselines_env.py`), `run_ppo.py` (CleanRL-style PPO, Nature CNN, `~/venvs/vm-gpu`) and `run_dreamer.py` (DreamerV3, the PyTorch port `NM512/dreamerv3-torch` at 6ef8646 in `~/src/dreamerv3-torch`, venv `~/venvs/dreamer`; JAX can't run on this CPU); both write our `metrics.jsonl` rows. vm-00108, seeds 1–6, 60k ticks: in Breakout stage 1 Dreamer returns .99 at 50–60k against the default agent's .65 (both near random until 25–30k); in the food world the default agent eats 71.7 food and 8.4 poison per 1k against Dreamer's 44.8 and 22.9; one-body PPO is far behind both; 16-body PPO needs 3–6 times our ticks (Breakout) and 20–30 times (food). Dreamer: about 3.6 h per life alone, about 8 times our replayed steps per tick (29.5M against 3.84M in 60k ticks).
  - **The universal model** (task 00023, `../ARCHITECTURE.md`; as built, with every parameter and buffer: `../ARCHITECTURE_REVIEW.md`): `model.universal=1` with `model.patch`: one token
    format, fully causal next-token prediction, one head, two-hot cross-entropy over `model.bins`; `model.gray=1` reads one
    channel per patch (task 00024). Start from `configs/U_default.json` (vm-00054's QNA, the default since 2026-10-31: four
    quadrant transformers, gain, age key, ring, the egocentric touch world without overlaps, log summaries `band_win = 1` since 2026-11-02, vm-00073; `lifetime.py`);
    `configs/U_universal.json` is the one-model form (grayscale world, checkerboard poison, dots, the instinct). Its forward is `Agent.forward_uni` / `CudaModel.forward_uni`; `vm_forward` refuses it.
    `online.click_bias` is skipped for models without a policy head (it used to write into the next parameter).
    Surprise gating (task 00028): `online.store_surprise` with the universal model on the GPU; a step's surprise is its
    largest token cross-entropy relative to that slot's recent mean (`GatedMemory.note_tokens`, read one tick late from
    `CudaModel.uni_token_ce`, after acting). Once long-term memory is full, `online.ltm_evict` (task 00031) says what a
    stored chunk replaces: `oldest` (the ring, default) or `reservoir` (a uniform sample of the life); vm-00025: in a
    life longer than the memory, surprise gating with the ring is the rule that avoids traps. The text world (task 00032):
    `env.kind = text` (`world/text_world.py`, made by `world.make_config` / `make_world`, which `lifetime.py` uses):
    Tiny Shakespeare as grey pages, configs `T_text_A.json` (64 letters in one 8x8 patch token) and `T_text_B.json`
    (one 2x2 letter per token), `bins = 256`, `pred_img = 8` (unused by the universal model but validated: it must
    divide `img`). `eval_predict.py` scores any world's held-out prediction (nats per value; text: per letter);
    `reference/char_gpt.py` (run with `~/venvs/vm-gpu/bin/python`) is the classical character-level GPT on the same
    windows. `model.in_bins = 1` (task 00036, vm-00030): bin-embedding input, `W_in` d x (V * bins) over each used value's
    two-hot bins (init 0.02), in C++, CUDA and the reference; header slot 43. Set in every universal config since task 00037 (vm-00031: with surprise-gated memory no 500k-tick life was trapped). The corpus is read from `../llm-fundamentals/dev/data/tinyshakespeare/input.txt` (`./venv/bin/python prepare_tinyshakespeare.py` there downloads it). The instinct's horizon (task 00029): `online.instinct_h` (1: one step)
    holds each candidate for H steps, imagined by the model (`online.uni_lookahead`, Jacobi decoding); it costs
    (H − 1)·J + 1 forward passes per tick. Task 00107 (vm-00096): with the quarter pieces each imagines only its own block,
    the rest of the frame kept as the current one; H > 1 needs `online.ring = 0` (the ring refuses it; ring and full windows
    are not the same, `tests/test_ring.py`), and `eval_hunger.Life` then acts on full windows of the last ctx steps. In the
    inertia world H = 4 costs 3.6 times H = 1 with J = 2, 2.4 times with J = 1 (one life on the 4090). `online.instinct_decode=mode` (GPU) decodes imagined pixels by their most probable
    bin instead of the mean (its bin centres are not the world's grey levels: use `pair`, task 00038, the most probable
    bin and its more probable neighbour, weighted: sharp, and exact for two-hot values); Self-forcing (task 00039, lifetime.py on the GPU): `online.sf_windows` replayed windows per update
    get their last 1..`sf_steps` steps' inputs (frame and body) from the model's own predictions (`online.dream_inputs`:
    `sf_jacobi` Jacobi iterations, pair decoding; moves stay real), the targets stay real (`CudaModel.set_uni_inputs`);
    8 windows of up to 3 steps cost ~70% of the speed (29 ticks/s per lifetime). `eval_dream.py` measures the dream:
    the frozen agent forages, then the model dreams 16 steps with the real moves that followed (pixels exactly right,
    cursor lost / distance, hunger error per dream step). `reference/text_dream.py` (task 00040, `~/venvs/vm-gpu`): the text world's dream is generation;
    the universal model and the classical GPT continue the same val contexts, greedy and sampled, scored by both (a letter's
    probability: the mean of its glyph values' log-probabilities; eval_predict's `nll_glyph` sums them, a temperature of 1 / V). imagined frames lose the moving cursor either way (task 00029). `online.py --eval-tests forage --eval-worlds 16 --eval-horizon 4` keeps
    the frozen test affordable.
    Speed (task 00026): the instinct's candidates reuse the policy pass's keys and values (`CudaModel.uni_candidates`),
    the universal model's batches carry no next frames, and `lifetime.py` reads the update's stats every
    `online.stats_every` ticks: 178 ticks/s for a gray lifetime with the instinct alone on the RTX 4090.
  - `lifetime.py` (task 00020, `../ARCHITECTURE.md`): the self actor with N parallel bodies early in life
    (`online.bodies`, `bodies_hold`, `bodies_halve`; one shared long-term memory, `ChunkStore`), the signed hindsight
    command (`online.goal_mode=delta`) and exploration decay. With one body it reproduces `online.py` bit for bit.
    `model.act_token=1` (patch models): a learned query token per step that the policy reads, trained by its own
    predictions of the next drives and the coarse next frame (`ppo.act_coef`). `analyze_patch.py`: coupling (attention between patches), action sensitivity and probes
    per kind of token.
  - `src/vm_cuda.cu`, `src/vm_cuda.h`, `cuda_lib.py` (tasks 00011, 00012): the basic agent's forward pass and
    update on the GPU (`make libvmcuda.so`, `online.device=cuda`).
  - `reference/torch_reference.py`: float64 PyTorch re-implementation, used by the tests (and by
    `analyze_patch.py tokens` for all token outputs).
  - `configs/*.json`: runs from the plan's experiment matrix (A–F) plus reward-free babbling (P_*).
- **`NNNNN-experiment-name/`**: experiment snapshots (README.md, Makefile, and `images/` with small PNGs for the README, at most 960 px wide and 200 KB each, made
  by `dev/readme_images.py`; nothing else, and no other binaries; the git tag `exp/vm-NNNNN` preserves the source).
- **`tasks/`**: numbered task plans (write one before implementing a plan).

## Build & Test (from dev/)

```bash
make setup          # venv: numpy, pillow, matplotlib, pytest, CPU torch (torch: tests only)
make                # libvmagent.so
make test           # C++ unit tests (GEMM, acting == training bitwise, determinism, checkpoints)
make pytest         # world/reward/leak tests, JS equivalence, PyTorch gradient equivalence
make bench          # acting / update throughput
make libvmcuda.so   # optional: the GPU update (nvcc, cuBLAS); tests/test_cuda_model.py skips without it
```

**MuJoCo** (`dev/venv`, installed by hand, not by `make setup`): this VM's CPU has no AVX, so the PyPI wheel dies with `Illegal
instruction` on import. MuJoCo 3.17.0 is built from source: the library in `~/src/mujoco` with `-DMUJOCO_ENABLE_AVX=OFF
-DMUJOCO_ENABLE_AVX_INTRINSICS=OFF`, installed in `~/opt/mujoco-3.17.0`; the bindings from PyPI's sdist in `~/src/sdist/mujoco-3.17.0`, patched in
`mujoco/CMakeLists.txt` (`set(AVX_COMPILE_OPTIONS)` in place of `get_avx_compile_options`; `GLFW_BUILD_WAYLAND` and `GLFW_BUILD_X11` OFF after
`project()`), built with `pip wheel . --no-deps --no-build-isolation`, `MUJOCO_PATH`, `MUJOCO_PLUGIN_PATH` and the real cmake binary on PATH; the
wheel is in `~/src/sdist/wheel`. Headless only, no viewer: render with `MUJOCO_GL=egl` and `CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_EGL_DEVICE_ID=N`.

## Run

```bash
./venv/bin/python train.py --config configs/A_baseline.json --seed 1 --total-steps 1000000 --set train.threads=64
./venv/bin/python analyze_dynamics.py RUN_DIR --ckpt ckpt_final.bin
./venv/bin/python plot_runs.py RUN1 RUN2 -o curves.png --labels A C
```

Outputs go to `~/data/research/visual-motor-agent/exp_NNNNN/RRR_name_sS/`, not into the repo.
`output_name.py` picks the next run number.

## Key Design Decisions

- **The agent runs on the CPU** (QEMU VM, 128 vCPUs, SSE4.2 only): C++17 with OpenMP and a hand-written
  SSE GEMM. Since 2026-10-24 the VM also has an RTX 3090 (24 GB, sm_86, driver 595, nvcc 12.0), and since
  2026-10-25 an RTX 4090 (24 GB, sm_89). `libvmcuda.so` carries native code for both.
  - **Pick a GPU with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N`.** CUDA numbers devices fastest-first by
    default, so without the order variable `CUDA_VISIBLE_DEVICES=0` is the 4090, while `nvidia-smi`'s GPU 0 is the
    3090. The 4090 is about 1.5× faster (8×8-patch update with the act token 3.8 ms against 5.6); three 8×8-patch lifetimes per card at
    once run at 113 ticks/s each on the 4090 and 74 on the 3090. One MPS daemon serves both cards.
  - **Profiling:** Nsight Compute works for this user (`NVreg_RestrictProfilingToAdminUsers=0` in
    `/etc/modprobe.d/nvidia-profiling.conf`): `ncu --metrics gpu__time_duration.sum --csv ./venv/bin/python SCRIPT`.
  - **The basic agent's lifetime can run on it** (tasks 00011, 00012): `src/vm_cuda.cu` → `make libvmcuda.so`,
    wrapped by `cuda_lib.py`; `online.py --set online.device=cuda`. Forward passes and updates run on the GPU;
    the weights come back to the C++ agent only for checkpoints. The update takes 1.8 ms against 31 ms in C++,
    and the online loop runs 202 steps/s against 24 (a 50k lifetime in 6 minutes). It supports the CNN model
    with coord, residual frames and U-Net skips; the frame, drives, move, inverse-dynamics and latent-dynamics
    losses, the planner's `encode`/`imagine`/`decode`, and patch tokens (tasks 00013, 00015): DMI, the planner
    agent L and vm-00010's patch-token agents run there. Windows longer than 32 tokens (patch tokens: `model.patch=8`
    gives 536 per window) use batched cuBLAS attention, any length (task 00018). Not PPO (only `train.py` uses it). GPU lifetimes are not
    bitwise reproducible against CPU ones, nor run to run (compare arms over seeds; frozen tests repeat exactly). `online.py --eval RUN --eval-device cuda` runs the frozen tests'
    forward passes on the GPU (about 10 s per checkpoint).
  - **Many lifetimes on one GPU: start the CUDA multi-process service** (`nvidia-cuda-mps-control -d`, with
    `CUDA_MPS_PIPE_DIRECTORY` set the same for the daemon and the runs). Twelve basic-agent lifetimes then run at
    69 steps/s each instead of 20.
  - **Keep it equivalent:** `tests/test_cuda_model.py` checks the loss and every gradient against
    `vm_debug_grads` (about 1e-7) and the forward against `vm_forward`. Re-run it after touching either side. The same parameter layout is shared,
    so the CUDA code's `ParamId` order must follow `vm_agent.cpp`'s.
  - CUDA PyTorch (for experiments only) is in `~/venvs/vm-gpu`; the project venv keeps CPU torch.
  - Use 64 threads per run. Acting is about 2× faster than with 128, and two runs fit on the
    machine at once.
  - Expect about 1.7–2.0k env-steps/s per run with 4 PPO epochs.
- **Information hygiene.** The model receives only: the image (uint8), the drives (zeros unless
  `env.drives`), the previous executed action, and optionally the previous reward. `info["gt_*"]`
  and the env state are for evaluation and scoring only.
- **Context = block.** Contexts restart at episode starts and every `ctx` (32) steps. Training
  sequences are exactly those blocks, so acting and training give bitwise-identical outputs (tested).
- **Every gradient is verified** against the PyTorch reference (`tests/test_model_reference.py`,
  about 1e-7 relative error per tensor). Re-run it after touching any op.
- **Prediction is auxiliary only.** It is never rewarded. Plain pixel MSE from a global latent
  ignores the 1%-of-pixels cursor. The recipe that learns action consequences reward-free (exp
  vm-00001) is:
  - `model.coord=1`, `model.pred_residual=1`, `model.pred_skip=1`
  - `ppo.pred_changed_weight=10`
  - enough updates: `model.ctx=8` with 32 minibatches, lr 1e-3
  - optionally the inverse-dynamics head: `model.inv_hidden=128`, `ppo.inv_coef=0.1`, about 2–4×
    faster. It predicts the own executed (dx, dy) from consecutive *visual features*, never from
    tokens, which contain the previous action.
  See `configs/P_babble_inv.json`.
- **Latent dynamics heads (`model.dyn_hidden`) must not train the encoder.** Matching imagined to
  encoded latents with gradients into the encoder collapses the features. Even a decoder on real
  latents alone shrank them until the cursor was no longer decodable (exp vm-00002). The heads
  read a detached v.
- **The self actor (`online.actor=self`) needs replay.** With the newest window only (batch 1), its
  encoder features blow up (norm 10²–10⁵ by 100k steps in 5 of 6 runs, vm-00006), and the drives
  and command vanish from the tokens. There is no weight decay (`ppo.weight_decay=0`). Even with
  replay, its features lose the cursor over training (vm-00007); check with
  `analyze_self_actor.py probe`. The cause is the move loss: detach it (`ppo.pi_detach=1`). To aim, the
  head needs a hidden layer (`model.pi_hidden=128`), so the hindsight command can steer it: with inverse
  dynamics the basic agent then reaches test success 0.63–0.82 at 50k (vm-00008 DMI).
- **A detached head still shares the gradient-norm clip** with the rest (`ppo.max_grad_norm`): its large
  move-loss gradient scales every other update down. `ppo.pi_own_clip=1` clips the heads by their own norm
  (found in vm-00009: a detached move loss slowed the planner agent's early learning).
- **Frozen checkpoints understate an online learner** (vm-00005). L's frozen planner misses 1–19%
  of episodes, while the same weights, still learning, miss almost none: errors are corrected
  within steps. To measure the agent, continue it with learning on (`online.py --init CKPT`; lr 0
  is the frozen control). `plan.py --explore-gain/--explore-eps` adds online.py's acting rule.
- **Run names:** `train.py --set name=X` overrides the config's name. Experiment Makefiles use
  it so their analysis globs find the runs.
- **Checkpoint header.** Config ints sit in slots 2–19 and n_params/has_opt/step in 20–22.
  Extension fields go at 32+ (`pred_residual` 32, `pred_skip` 33, `inv_hidden` 34, `dyn_hidden` 35,
  `one_frame` 36, `pi_hidden` 37, `patch` 38, `act_token` 39). Never append into slot 20.
- **Patch tokens act through `vm_forward`** on the context window (the basic agent already does), so
  they need no KV cache: `vm_act` and `vm_encode` refuse patch mode, and so does the planner (it imagines
  with the CNN model's latent dynamics). Attention is block-causal: a token sees all tokens of its own
  step and of earlier steps.
- `make` builds `libvmagent.so` under a temporary name and renames it, so running trainers are not
  disturbed by a rebuild.
- **Small updates are frame-parallel.** With fewer sequences than threads, `loss_and_grads_frames`
  and `vm_forward` spread per-frame encoder and decoder work over threads, for any `ctx`. Larger
  minibatches use one thread per sequence. Both paths are covered by the reference gradient tests
  (run at 4 and 16 threads).
- **Testing a new build while experiments run:** build it to another path, run
  `VM_AGENT_LIB=that.so pytest`, and only then `make`.

## Conventions

- No co-authored-by lines in commit messages.
- Don't kill runs with `pkill -f <pattern>`: the pattern also matches the calling shell. Kill by PID.
- Experiment READMEs are technical reports (hypothesis, method, results, analysis).
- Large logs and checkpoints stay in `~/data/...`; READMEs quote key numbers only.
