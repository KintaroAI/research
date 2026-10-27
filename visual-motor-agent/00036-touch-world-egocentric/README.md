# vm-00036: The Touch World, Egocentric

**Date:** 2026-10-27
**Status:** Complete
**Source:** `exp/vm-00036` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00036/`
**Task plan:** [`../tasks/00043-touch-world-egocentric.md`](../tasks/00043-touch-world-egocentric.md)

## Goal

In the touch world (vm-00035) the agent collects food and avoids poison unrewarded; self-forcing with the 4-step
`pair` lookahead in life (TSF4) cut poison to 3.0 per 1k at ~20 food. Its view was allocentric: the cursor moves over a
fixed scene. Here the view is egocentric (`env.egocentric`): the cursor is always drawn at the center and every move
shifts the scene. Prediction (the user's): poison contacts drop further. A contact always happens at the same place in
the view, and the dream cannot lose the cursor; against that, every move changes every patch.

## Method

- **World** (`configs/U_touch_ego.json`): vm-00035's `U_touch.json` with `egocentric = true`. The dynamics are
  unchanged (hit tests in world coordinates; a test checks the events are identical to the allocentric world's for the
  same seed and moves), so vm-00035's chance numbers hold: no blind walk exceeds food : poison 0.78.
- **Arms** (100k ticks, seeds 1–3), each the egocentric twin of a vm-00035 arm:
  - **EB:** no self-forcing, the 1-step instinct (vm-00035's TB). 3090, 62 ticks/s.
  - **ESF4:** self-forcing (8 windows, 1–3 steps, J = 2) and the 4-step pair lookahead in life (vm-00035's TSF4).
    s1–2 on the 4090, s3 on the 3090 after EB, 13.4 ticks/s each.
- **Frozen at 100k** (`online.py --eval-tests forage`, 16 fresh worlds × 1,000 steps, the same worlds as vm-00035):
  instinct off, 1-step, 4-step mean, 4-step pair. The dream curve (`eval_dream.py`).
- **Videos** (`forage_video.py`, `dream_video.py`; paths and touch rings now drawn relative to the centered cursor in an
  egocentric view).

## Log

### 2026-10-27
- All six lives ran without incident; none had a 1k window without food.
- `eval_dream.py`'s cursor distance is 0 by construction egocentric (the cursor is drawn at the center); only pixels
  exactly right and the hunger error compare across views.

## Results

### R1. Frozen at 100k: food / poison per 1k (food : poison)

| run | instinct off | 1-step | 4-step mean | **4-step pair** |
|---|---|---|---|---|
| EB s1 | 28.4 / 19.6 (1.45) | 32.2 / 8.1 (3.97) | 35.9 / 5.4 (6.60) | 38.7 / 4.1 (9.38) |
| EB s2 | 25.8 / 23.5 (1.10) | 29.4 / 11.9 (2.47) | 30.6 / 7.8 (3.92) | 34.1 / 5.2 (6.50) |
| EB s3 | 46.4 / 15.1 (3.08) | 48.2 / 6.4 (7.57) | 50.9 / 4.3 (11.80) | 60.4 / 3.0 (20.12) |
| ESF4 s1 | 25.6 / 14.5 (1.77) | 25.8 / 2.8 (9.36) | 36.1 / 2.4 (14.82) | 41.6 / 2.6 (15.83) |
| ESF4 s2 | 32.2 / 14.1 (2.29) | 38.6 / 6.4 (6.05) | 48.6 / 2.9 (16.89) | 49.9 / 3.1 (16.31) |
| ESF4 s3 | 25.8 / 17.8 (1.45) | 25.3 / 3.6 (6.98) | 45.9 / 5.9 (7.73) | 39.5 / 2.0 (19.75) |

Means against vm-00035's allocentric twins (food / poison per 1k):

| | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| TB (allocentric) | 15.5 / 16.7 | 16.9 / 13.7 | 16.5 / 13.3 | 19.9 / 10.7 |
| **EB** (egocentric) | 33.5 / 19.4 | 36.6 / 8.8 | 39.1 / 5.9 | 44.4 / 4.1 |
| TSF4 (allocentric) | 14.1 / 16.1 | 15.5 / 8.0 | 17.8 / 10.5 | 19.7 / 3.0 |
| **ESF4** (egocentric) | 27.9 / 15.4 | 29.9 / 4.2 | 43.5 / 3.8 | **43.7 / 2.6** |

### R2. In life: food / poison per 1k, by quarter of the life

| run | 0–25k | 25–50k | 50–75k | 75–100k |
|---|---|---|---|---|
| EB s1 / s2 / s3 | 35.0/10.0 · 32.0/13.4 · 41.5/9.7 | 34.4/8.9 · 31.3/11.8 · 47.2/6.8 | 33.5/8.4 · 28.0/11.3 · 48.9/7.6 | 30.8/8.3 · 29.0/9.6 · 49.6/5.9 |
| **ESF4 s1 / s2 / s3** | 36.1/8.6 · 31.1/8.0 · 21.0/5.8 | 32.4/2.8 · 34.0/2.8 · 28.5/2.4 | 37.1/2.8 · 36.3/2.1 · 33.9/2.0 | **35.0/2.5 · 38.0/1.9 · 35.1/1.6** |

vm-00035, last quarter: TB 12–19 food / 9.8–17.3 poison; TSF4 12–21 / 3.6–5.2.

### R3. The dream at 100k (16 dreamed steps with the real moves)

| run | pixels exactly right after 1 / 6 / 11 / 16 steps | hunger error after 16 |
|---|---|---|
| EB s1 / s2 / s3 | 0.98 / 0.92–0.95 / 0.90–0.92 / 0.882–0.892 | 0.062 / 0.062 / 0.103 |
| ESF4 s1 / s2 / s3 | 0.98 / 0.94–0.95 / 0.92 / 0.903–0.906 | 0.028 / 0.028 / 0.014 |
| TB (vm-00035) | | 0.128–0.178 (pixels 0.900–0.912) |
| TSF4 (vm-00035) | | 0.022–0.053 (pixels 0.904–0.932) |

### R4. Videos (in `exp_00036/`, not in git)

- **`video_ESF4_s3_vs_TSF4_s3/compare.mp4`** (1,000 steps, 15 fps): ESF4 seed 3 (egocentric, left) and vm-00035's
  TSF4 seed 3 (allocentric, right), both frozen at 100k with the 4-step pair lookahead, in the same world (the forage
  test's world 0: the same circles and dots, seen centered on the cursor on the left). In world 0: **ESF4 41 food, 1
  poison; TSF4 23 food, 2 poison**; over the run's 16 worlds 39.5 / 2.0 and 24.1 / 2.9, the frozen test's numbers.
  `forage.mp4` and `forage_sheet.png`: ESF4 alone (the sheet's paths are drawn relative to the cursor at the panel's
  last step).
- **`video_ESF4_s3/dream_pair.mp4`, `video_EB_s3/dream_pair.mp4`** (200 real steps, then 200 dreamed): the dream
  shifts the scene with the dreamed moves for a while (circles leave at one edge and come in at the other), then
  empties: fewer than 30 circle and dot pixels from dreamed step 115 (ESF4) and 93 (EB) on; the real frames hold
  390–500. The allocentric dreams (vm-00035) never empty (285–335 at step 200) but lose (TB) or misplace (TSF4) the
  cursor.

## Analysis

- **The prediction holds: poison falls further, and food doubles.** With the 4-step pair lookahead ESF4 touches 2.6
  poison per 1k at 43.7 food (frozen), against TSF4's 3.0 at 19.7; food : poison 16–20 against 4.7–8.2 (blind walks
  ≤ 0.78). In life ESF4 ends at 1.6–2.5 poison per 1k with 35–38 food, against TSF4's 3.6–5.2 with 12–21. Per food
  eaten, ESF4 touches poison 2.6× less often than TSF4.
- **Most of the gain is the view.** EB (no self-forcing, 1-step in life) is already near TSF4 in poison (4.1 against
  3.0 per 1k, pair) with 2.3× its food, and every egocentric policy forages ~2× better than its allocentric twin, even
  with the instinct off (33.5 against 15.5 food). Likely because a circle at the center is a contact whichever way
  the agent has moved; allocentric, the same event happens anywhere in the frame and must be learned per place.
- **Self-forcing still pays, less.** Egocentric it cuts the pair lookahead's poison 4.1 → 2.6 per 1k (−37%) at the
  same food, against 10.7 → 3.0 (−72%) allocentric, and cuts the dreamed hunger error by half or more (0.014–0.028
  against 0.062–0.103). The egocentric dream cannot lose the cursor, which was self-forcing's main gain allocentric.
  As in vm-00035, EB and ESF4 differ in two things (self-forcing, and the lookahead in life), so the in-life gain is
  not split; the frozen tests compare them under the same instinct.
- **Mean decoding now helps the lookahead** (EB: 4-step mean 5.9 poison against 1-step 8.8; ESF4 3.8 against 4.2),
  which it did not allocentric (vm-00035: mean ≈ 1-step). Pair is still best or equal.
- **The dream's weakness moved.** Allocentric the scene stays and the cursor drifts; egocentric the cursor stays and
  the scene must shift with every move. The 16-step pixel share is about the same (0.88–0.91 against 0.90–0.93), and
  the 4-step lookahead lives well within it, but over ~100 dreamed steps the egocentric dream forgets the world (it
  goes blank).

## Conclusions

1. Egocentric, the touch-world agent forages twice as well and touches poison less: with self-forcing and the pair
   lookahead, 43.7 food and 2.6 poison per 1k frozen (allocentric 19.7 / 3.0), 35–38 food and 1.6–2.5 poison in the
   last quarter of its life. The user's prediction holds.
2. Most of it comes from the view itself: without self-forcing (EB) the egocentric agent already eats 2.3× more than
   allocentric TSF4 with poison near its level.
3. Self-forcing still lowers poison egocentric (−37% with the pair lookahead) and makes the dreamed hunger the most
   accurate yet; its larger allocentric gain came from keeping the cursor, which egocentric gives for free.

## Next Steps

- The missing cells: ESF (self-forcing, 1-step in life) and EB4 (the pair lookahead in life without self-forcing),
  to split R2's in-life gain.
- The egocentric dream empties after ~100 steps: longer self-forced stretches (`sf_steps` > 3) may keep the scene.
- Moving food (`env.target_speed`) egocentric; a longer lookahead now that food is seen in the frame's coordinates.
