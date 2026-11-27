# Task 00132: Worlds to Watch — the World Model Measured Without Moves (experiment vm-00119)

**Date:** 2026-11-26
**Status:** Complete (2026-11-26; pending tag `exp/vm-00119`): the two worlds and the watching option, the 24 lives, the measures tool and the first reference table (vm-00119: `00119-watching-worlds/README.md`). The owner's word for the lives and the measures was "lets go according to the plan then"; the word to commit, tag and push came the same evening.

## Context

- **The decision** (the owner, 2026-11-26; `ARCHITECTURE.md` section 15): the world model, its dream and its memory come before anything else, measured by numbers that do not depend on the model's parts.
- **Watching first** (the owner, 2026-11-26): "I think we need to exclude own movement and decision making for now - it should observe the world, moving objects and stuff - for example in foraging world we can have food and poison to move and collide - maybe poison move slowly and food faster and they do not overlap and the idea is to see how it predicts, and for arcanoid (what was that called?) we can write simple script that plays it and maybe score - so it can predict not only movements (paddle movements should be predictable, not random) but the score as well".
- **What existed:**
  - the default agent's world already moves its poison (.75 px a tick) and lets no two circles overlap; its food stands still (more than one food had to), and two moving circles that meet swap the normal parts of their velocities, so speeds mix;
  - the Breakout court (the owner's "arcanoid": here it is called Breakout, task 00111) has scripted players in `dev/breakout_walkers.py`: random, still, tracker, aimer. The aimer draws a random offset.

## The two worlds

### Moving food and poison: `env.kind = watch` (`dev/world/watch_world.py`, `configs/W_discs.json`)

- A 64 x 64 torus in the food world's looks: food a solid disc, poison a disc showing the view's checkerboard (2 px cells), both grey 94 on 21, dots under them.
- **2 food discs at 1.5 px a tick, 2 poison discs at .75**, radii as in the default agent's world (8–11 and 5–8), 8 dots at .5–2 px a tick.
- **They never overlap, and every disc keeps its speed.** Two discs that meet are pushed apart along the line of their centres, half the overlap each, and each one that was moving toward the other has its velocity mirrored off that line. An exchange of momentum between equal masses would mix the speeds, and food would stop being the faster kind.
- No body is in the frame, nothing is eaten, the body's values are 0, the move does nothing.
- **Nothing is left to chance after the first frame,** so a perfect predictor is exact at any horizon.
- `get_state` / `set_state` copy a world. For scoring only: `gt_pos`, `gt_vel`, `gt_rad`, `gt_kind`, `gt_bounce`, `gt_dots`.
- **Measured without a model:**

  | Food, poison discs | Share of the frame covered | Ticks with a bounce, per 1k | Median ticks between a disc's bounces: food, poison |
  |---|---|---|---|
  | 2, 4 | .27 | 234 | 8, 13 |
  | 2, 3 | .23 | 165 | 10, 16 |
  | **2, 2** | .20 | 108 | 13, 20 |
  | 1, 3 | .16 | 78 | 16, 26 |
  | 1, 2 | .13 | 42 | 22, 32 |

  8 worlds, 6,000 ticks each. Two and two is the config: flights long enough to predict, and a bounce in about one tick of nine. Over 5,000 ticks the speeds drift by less than 1e-14 and the smallest gap between two discs is 0 (never an overlap).

### The court with a script: `env.auto_lag = d` (`dev/world/breakout_world.py`, `configs/W_breakout.json`)

- Breakout stage 1 as the agent plays it (`U_breakout.json`), with the paddle given to a script and the move ignored.
- **The script sees the court as it was d ticks ago.** If a ball was in play then, it pushes the paddle's centre toward that ball's centre, as far as the paddle's speed allows (4.5 px a tick); if not, the paddle stays. Nothing is random, and with d = 6 every move follows from frames inside the model's 8-tick context.
- **Why a delay.** A tracker without one is no world to watch: it centres the ball, the ball then goes straight up one column, and the paddle stops moving. Limited to .5 px a tick, a ninth of the paddle's speed, it still returns .99 of the balls. With a delay the paddle trails a slanting ball, meets it off-centre, and the ball leaves at an angle; with a long delay it also misses.
- **Measured without a model** (4 seeds, 20,000 ticks each):

  | Delay in ticks | Hits per 1k | Misses per 1k | Hit share | Mean angle off the paddle, degrees from vertical |
  |---|---|---|---|---|
  | 4 | 11.4 | 0.0 | 1.000 | 27.7 |
  | 5 | 10.2 | 1.1 | .907 | 28.5 |
  | **6** | 9.4 | 2.3 | .804 | 28.8 |
  | 7 | 9.2 | 3.0 | .757 | 27.1 |
  | 8 | 8.5 | 4.0 | .678 | 27.5 |

  Six ticks is the config: both events happen (about 560 hits and 140 misses in 60k ticks) and the script's rule stays inside the context.
- **The score** is the body's hunger, as in the court the agent plays: +.001 a tick, a hit sets it to 0, a miss adds .5. It is a value the model already predicts, so no new channel is needed. A score drawn on the court as pixels would be a different test (reading a number) and is not built.
  - **Where it comes from:** the court's code computes it at every tick from what happened to the ball (`BreakoutWorld.step`). It is not in the picture. The model reads it as the first of the two numbers of the body token and predicts its next value, as it predicts the next frame's pixels.
  - **How it looks under the script** (4 courts, 20,000 ticks each: 733 hits, 190 misses): it sits low (median .05, below .2 on .88 of the ticks). A hit's fall is small, median .09 (the drift since the last hit; under .1 for .69 of the hits). A miss's rise is the full .5 for .94 of the misses; the cap at 1 cuts the rest (a miss soon after a miss), none wholly.
  - **It is a level that a hit resets, not a count of points.** Ten hits in a row and one leave the same value. For the event forecast the events are taken from the world (`gt_paddle_hit`, `gt_hazard_click`), and a hit is the harder one to see in the value.
  - **The body's second value is a constant here** (found while checking this, 2026-11-26): restlessness is 1 after the first 100 ticks or so. Movement relieves it only within a budget of 200 px, and only an episode reset refills the budget; no life here resets. The same holds in the court the agent plays (1 from tick 113–123 on with random moves) and in the default food world (1 on every tick after the first 1,000, random moves). A count of points, if the owner wants one, could take that place without a change to the model.
- The only chance left in this world is the serve after a miss (where, and at what angle).

## The model only watches: `online.watch = 1` (`dev/lifetime.py`)

- The move made is the null move (no push, no click) at every tick, whatever the model's own move prediction says. The world gets it, the ring and memory hold it, and it is the move token's target.
- Needs the universal model with `instinct_m = 0`; refused with a label, a value, `need`, `actor_pg` or `act_every`.
- **The model** in both configs is the default agent's (`U_agent.json`) without the label: 4 quarter transformers of 8 layers, context 8, the ring, `piece_rule = mean`.
- **A first look** (one life of 2,000 ticks per world, alone on the 4090; not a result): 40 ticks per second. The patches' cross-entropy falls from 1.38 to .96 (discs) and from .93 to .67 (the court); the move's settles at .463, the floor for a null move (dx and dy lie midway between two bins).

## The measures in this setting (`ARCHITECTURE.md` section 15)

- **Set aside while the model only watches:** measure 3 (what a move does), measure 12 (the worth to a decision), the anchor "another move".
- **Kept:** 1 next-frame skill, 2 event forecast, 4 the dream's curve and horizon, 5 object hold, 6 dream realism, 7 honesty, 9 warm-up, 10 learning, 11 cost; 8 (memory span) once a gap is put in front of the discs and the ball (`online.blind`).
- **Events:** in the court, the jumps of the body's hunger (a hit, a miss); in the disc world, which has no body, the bounces (`gt_bounce`), scored by the next-frame skill on the ticks around a bounce against free flight.
- **The readers:** the discs' centres by kind (solid or checker), the ball, the paddle.
- **Other dice:** only the court's serve. The disc world has none.

## The order of work (the owner's word, 2026-11-26: "lets go according to the plan then")

1. **The lives that watch** (vm-00119): WD (`W_discs`) and WB (`W_breakout`), seeds 1–12, 60k ticks, a checkpoint every 10k. Started 14:28, the last ended 18:57. Seeds 1–8 on the RTX 4090, six lives at a time (14.5 ticks per second each); seeds 9–12 on the RTX 3090, four at a time (11.1 each). A life's frames can be made again from its seed (the recorded frame of tick t equals the world stepped t times with null moves, 400 of 400 ticks in both worlds), so nothing more is recorded.
2. **The measures** (`dev/wm_measures.py`), built while the lives ran. How each number is counted is set in the next section, before any of these lives was scored. Checkpoints of the running lives were used before that only to time the passes and to count the decoding passes until a frame stops changing; the adapter was tried on throwaway models of 3,000 ticks (seed 99).
3. **The reference table** from the checkpoints, with the floors and ceilings: done, vm-00119. Each life was scored as it ended, beside the lives still running; all scoring ended 19:45. DreamerV3's world model through the same contract: not done.
4. Then the changes of section 15, each judged on the same table: for the owner's word.

## The measures as built (set 2026-11-26, before any life was scored)

### The contract and the adapters

- **Three calls.** `reset(n)` starts n streams with nothing carried; `observe(frame, body, move)` takes one real tick of every stream (the frame as grey levels, the body's values, the move made on seeing them); `dream(H, moves)` returns the next H ticks as frames in grey levels and body values. Nothing else of a model is read.
- **The four transformers** (`Pieces`): the last 8 real ticks are the window. A dreamed tick: the body is the mean of the four move tokens' expected next body; each transformer decodes its own quarter, the quarters make one frame, and that frame is the input of all four. The decoding is repeated from the previous frame until the frame stops changing, which is decoding token by token exactly (on an untrained model: 45 passes). The same weights give two models: `pair` (each value the most probable bin and its neighbour: sharp; task 00038, `eval_dream.py`'s default) and `mean` (the expected value: the least squared error, blurred).
- **The anchors:** `copy` (the last real frame and body, held); the mean frame (the mean of all the exam's real frames, per pixel); `linear` (for positions only: each thing kept moving as it last moved); the world itself (the recorded continuation: error 0); in the court also other dice (the world copied at the start and stepped with another chance for the serves).

### The exam

- **Working set:** one world of 16 copies, seed 1001, null moves, 2,300 ticks from birth. **Final set:** seed 2001, not opened until a final comparison.
- **Regular starts:** ticks 200, 325, ..., 2,075 of every copy: 256 stretches. Before each the model observes the 64 real ticks up to it; then it dreams 32.
- **Event starts in the court:** for hits and for misses, and for each lead d of 1, 2, 4, 8 and 16 ticks, up to 64 events taken in turn from the 16 copies; the dream starts d ticks before the event and lasts d ticks.
- **Event starts in the disc world:** 128 starts whose next tick has a bounce, 128 with no bounce from 2 ticks before to 2 after; one dreamed tick each.
- **Home:** the life's own world (its seed), ticks 60,000 to 62,300, with the same 16 regular starts. Away is the working set. A life saw one world only (in the disc world the same four discs for ever), so the two are read side by side.

### The numbers

Errors are squared differences in grey levels. A **skill** is 1 − the model's summed error ÷ the copy's summed error over the same pixels: 0 is no better than copying, 1 is exact.

1. **Next-frame skill:** the first dreamed tick of the regular starts. Whole frame, and by kind of thing over the pixels the thing covers in the last real frame or in the true next one: food discs, poison discs, dots, the rest; the ball, the paddle, the rest. Where the copy's error is 0 (the rest) the model's error is given as it is.
2. **Event forecast.**
   - **The court, next tick:** over every tick of the working set, the predicted change of the hunger. For hits the score is the predicted fall, for misses the predicted rise; the area under the curve of ticks with the event against ticks without (.5 is chance). The size: the summed predicted change ÷ the summed real change on the event's ticks.
   - **The court, within h ticks:** the score is the largest fall (rise) between two successive ticks of the dreamed hunger within the first h dreamed ticks; event starts with the lead h against regular starts with no such event within h ticks.
   - **The disc world:** next-frame skill on the discs that turn (bounce starts) against discs in free flight (free starts), and their distance in px (measure 5) beside `linear`, which is exact in free flight and wrong at a bounce.
3. (waits) the move's effect.
4. **Dream curve and horizon:** for each dreamed tick h = 1..32 of the regular starts, the skill against the copy and against the mean frame. The horizon is the number of ticks from the first on during which the dream's error stays below both; 32+ if it always does.
5. **Object hold.** For each thing at each dreamed tick: the frame without it B (the other things in their true places), its paint P at the shift d, and the gain of drawing it there, the sum over its painted pixels of abs(F − B) − abs(F − P). The shift with the largest gain within 8 px (the paddle: along its row) is where the dream has the thing. **Lost:** no shift has a gain above 0, that is, the dreamed frame is nowhere nearer to the thing drawn than to the frame without it. **Held:** found within 2 px of its true place. Per kind and tick: the shares held and lost, the median distance of the found. Floors: `linear` and `copy`, held by the same 2 px. The horizon by objects is the number of ticks from the first on during which at least half are held. On real frames the reader must give distance 0 for every thing (a test).
6. **Dream realism,** without the truth:
   - the distance between the grey-level distributions of the dreamed and the real frames at each tick (the mean absolute difference of their sorted grey levels, in grey levels);
   - **valid frames:** the count of bright pixels in each zone (the disc world: pixels above 57 anywhere; the court: above 110 in the paddle's rows, above 137 above them, above 55 on the walls) lies between the smallest and the largest count of the same copy's real frames. The share of valid dreamed frames at 8 ticks goes on the board;
   - a free dream of 200 ticks, on seeds 1–3 only (its cost): the tick by which half of the dreams have shown an invalid frame.
7. (later) honesty: this adapter gives one guess, not versions.
8. (later) the memory span: needs a gap (`online.blind`).
9. **Warm-up:** from a reset, shown the last k real ticks, k = 1, 2, 3, 4, 6, 8, 16: the next-frame skill of the regular starts and, in the court, the next-tick forecast of the event starts with lead 1 against the regular starts. Past 8 nothing changes for this model by construction.
10. **Learning:** the next-frame skill, the next-tick forecast and the first 8 dreamed ticks at the checkpoints of 0, 10k, ..., 60k (`pair`); the first checkpoint at which the next-frame skill is within a tenth of its value at 60k.
11. **Cost:** ticks per second of one life alone on the 4090, card memory, parameters.

**The statistics:** every number per life; the mean over the 12 seeds with its 95% interval. The anchors have no seeds.

**What I expect** (written 2026-11-26 15:00, before any life was scored; a guess to be checked, not a rule):
- `mean` scores better than `pair` on pixels, and worse on valid frames and on the things held, because it blurs.
- The dream's horizon by pixels is under 8 ticks in both worlds (vm-00114: the dream's score was noise at 4 ticks).
- `linear`, which needs no model, holds the things longer than the dream does: on the working set it holds half of them for 12 ticks (food), 20 (poison), 15 (the ball) and 22 (the paddle).
- The next-tick forecast of a hit is above .9.
- In the disc world home is better than away: a life saw the same four discs for 60k ticks.

**Decided on the way, before any life was scored:**
- `mean` is scored on the regular starts only; the events and the warm-up are scored with `pair` (the next-tick forecast reads the expected body and is the same for both).
- The learning curve leaves out the court's dreams toward an event (the next tick only), for its cost.
- A disc pushed by a disc that turned moves off its line without turning; only the discs whose velocity changed count as turning.

**Added after the first lives were scored** (2026-11-26; each is marked so in the write-up, and none changes a number set above):
- **A third decoder, `instinct`:** the same weights dreamed as the agent's instinct and vm-00114's lookahead dream (`online.uni_lookahead` with its defaults: the expected value, 2 passes, each quarter alone, the rest of the frame left as the last real one). Seed 1 showed `mean` losing the ball within 2 dreamed ticks, and vm-00114's negative result was read off such a dream.
- **`instinct_pair`:** the instinct's way of dreaming with `pair` in place of the expected value (`online.instinct_decode = pair`), to tell the expected value's part from that of the 2 passes and the quarters alone, and to see what the existing lookahead option would dream.
- **A fair straight line.** The floor set above (`linear`) is drawn through each thing's last two true places. In the court a thing is drawn at whole pixels, so no frame shows those places: a line through the ball's place in the last 2 frames as drawn holds it for 2 ticks, through 4 frames for 7, through 8 for 11, where `linear` holds it for 15. Lines through the places as the last 2, 4 and 8 frames show them are added (`hold_seen`; the disc world: the centroid of a disc's pixels, .07 px from its true centre on average). A floor must know no more than a model can see; `linear` stays in the tables as what the true places would allow.
- **The dream after a short warm-up** (`warmdream`): from a reset, shown the last 2, 4 or 8 real ticks, 8 dreamed ticks, to see whether the dream uses more of the window than the next frame does.
- **What makes a frame invalid:** the valid share and the bright counts by zone (`real.zones`), in the regular starts and in the dream of 200 ticks. The first invalid frame of the long dream comes at tick 1 or 2 in every life, so that number says nothing; the counts by zone show what the dream keeps.
- **After a bounce:** at the next tick a bounce moves a disc about .4 px off its line, too little to read. The discs of the regular starts are split by when they first turn in the dreamed stretch (within 2 ticks, or not within 4) and read at tick 4, from the saved distances.
- **The court's next-tick forecasts by tick** are kept (`_events.npz`) for later readings.
- **The learning curve's first point:** the untrained weights (checkpoint 0) are scored for seed 1 only. They are far below 0 in every life (next-frame skill −155 in the court, sharp noise), and they cost the most to score (45 decoding passes a tick). The checkpoints of 10k to 50k are scored for every seed; 60k is read from the full run.
- **How much the floors move with the exam's draw** (two spare seeds, 3001 and 3002; the final set stays unopened): the copy's error within 3%, the horizons of `linear` within 3 ticks (food 12, 9, 10; poison 20, 18, 19; the ball 15, 16, 18; the paddle 22, 19, 21). A horizon by objects is not to be read finer than that between two exams; on one exam two models are compared stretch by stretch.

**The board** (per world, at 60k, `pair`): next-frame skill; the next-tick forecast (the court) or the skill on turning discs (the disc world); the dream's horizon by pixels and by objects; the share of valid frames at 8 ticks; each beside its floors. The memory span and the move retrieval are empty until their turn.

## Results (vm-00119; 12 seeds a world at 60k, the working set)

The write-up is `00119-watching-worlds/README.md`. In short:

- WB (court), `pair`: next-frame skill .667 ± .010; hit forecast next tick .996, miss .987; 8 ticks ahead .939 / .913; dream horizon by pixels 22.7; ball held 9.1 ticks (a straight line through 2 / 4 / 8 frames as seen: 2 / 7 / 11; through the true places 15), paddle 22.9 (seen 8: 15); valid frames at 8 ticks .752.
- WD (discs), `pair`: next-frame skill .337 ± .042 (home .558, +.221 ± .047); horizon by pixels 6.2; food held 3.0 ticks, poison 3.9 (a straight line through the last 2 frames: 10, 13); valid at 8 ticks .095.
- Decoders, ball held at ticks 2 / 4: pair .928 / .818; mean .255 / .010; instinct .234 / .008; instinct_pair .866 / .654.
- The window: next-frame skill from 1 / 2 / 4 / 8 real ticks, court .533 / .639 / .667 / .667, discs −.118 / .308 / .334 / .337; the ball held at dreamed tick 8 from 2 / 4 / 8 real ticks: .592 / .582 / .591.
- Learning: next-frame skill in the court .403 at 10k, .559 at 20k, .639 at 40k, .667 at 60k; in the disc world .265, .299, .323, .337.

What it says:
- The model knows what comes next (the court's events almost surely); the dream is weakest in where the things are.
- In the disc world a straight line through the last two frames holds a disc three times longer than the dream; a bounce ends a disc in the dream.
- In the court the dream holds the ball nearly as long as a straight line through all 8 frames of the window, from the last 2 frames alone.
- The decoder decides: with the expected value (the instinct's default, vm-00114's lookahead) the ball is gone by the fourth dreamed tick.
- Half of the 8-tick window is not used.
- What I expected: 3 of 5 right as set, 2 half right (the task's "What I expect"; the README's table has each).

## Verification

- `tests/test_watch.py`, 18 tests: the looks, speeds kept and no overlap over 3,000 ticks, the mirror rule by hand, the torus, no chance after birth and the move ignored, the state; the script off by default and the court unchanged, the move ignored, the paddle going where the ball was (delays 1 and 6), a varied game with hits and misses; a watching life never moves the paddle, both configs live, the refusals.
- **The 14 test files around the change** (worlds, the life loop, the transformers' lifetime, Breakout, the baselines' wrapper): 206 passed, 4 skipped, 1 failed. The failure was a stale list of keys in `tests/test_env.py` (it lacked `hunger_cut`, added in task 00110), corrected; its file then passes (33).
- **The old court is unchanged:** vm-00103's BD at 60k, seed 1, recorded again with the changed code (`analyze_misses.py record`, 16 courts, 1,500 ticks) equals vm-00118's recording in all 36 arrays.
- **The measures tool:** `tests/test_wm_measures.py`, 29 tests (with `tests/test_watch.py`: 47 passed on the RTX 3090 after the last change). Predictors made by hand get the answers known beforehand: the world itself 1 at every horizon and the copy 0; a frame shifted by 4 px has every disc 4 px off; a blank frame has lost everything; a blurred ball beats a ball 2 px off on pixel error and is lost to the reader; the things painted back on the frame without them give the real frame, and the reader finds each at shift 0 with a gain above 0. The adapter's decoding gives the same frames from another first guess, and dreaming changes nothing it carries.
- **Every life's files:** a check of 1,015 items (each file there, its parts, their sizes, the life and the seed it names, no frame whose decoding did not settle) passed before the write-up was built.
- **A slip, and its repair:** the helper script that ran the readings added later (it is not in the repository) used the loop's variables inside a function, so for half an hour it scored the same two lives over and over and marked every job done; one of those runs overwrote the main results file of WB seed 12 while that life was being scored. Seen when the tables showed those readings for two seeds only. The script was corrected, the markers reset to the files that existed, WB seed 12 scored again, and the check above written. Lesson: a job is done when its output is there and names the right life, not when its command returned 0.
- **The README's numbers** are filled from the lives' files by the same reader as the tables (the prose cannot disagree with them); every table row has its head's width.
- **The whole dev test suite** after the last change (2026-11-26 20:15, RTX 4090): 2,119 passed, 34 skipped. One failure met on the way was older than this task and is corrected: `tests/test_record.py::test_json_and_npz_round_trip` had failed since vm-00080 added `EnvConfig.delays`, a tuple that came back from JSON as a list (`EnvConfig.from_dict` now makes it a tuple again).

