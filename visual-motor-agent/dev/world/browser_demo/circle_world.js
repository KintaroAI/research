/**
 * circle_world.js: browser / Node twin of world/circle_world.py.
 *
 * Same renderer, same step semantics, same float64 math in the same operation
 * order.  Reward and drive outputs are written to Float32Array, which matches
 * the float32 outputs of the Python simulator.  Resets draw from a small
 * seeded PRNG (mulberry32) rather than numpy's PCG64, so random episodes differ
 * from Python's.  Equivalence tests therefore pass explicit states
 * (see render_cli.js and ../check_js_equivalence.py).
 *
 * No dependencies.  Works as a classic <script> (global `CircleWorldJS`) from
 * file:// and as a CommonJS module in Node (`require('./circle_world.js')`).
 *
 * Conventions (identical to Python):
 *   - Config objects use the Python EnvConfig field names (snake_case).
 *   - Images are RGB Uint8Array of length n*size*size*3, laid out like a numpy
 *     (n, size, size, 3) C-order array: image[env][row = y][col = x][channel].
 *   - Pixel (col j, row i) has center (j + 0.5, i + 0.5); the circle covers it
 *     iff ((j+0.5)-cx)^2 + ((i+0.5)-cy)^2 <= r*r.
 *   - Cursor hotspot is (floor(x), floor(y)); the sprite is a 7-px white "+"
 *     with a 1-px black outline (3x3 dilation minus the "+").
 */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  } else {
    root.CircleWorldJS = api;
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /** Cursor is clamped to [0, size - CURSOR_EDGE_EPS] on both axes. */
  const CURSOR_EDGE_EPS = 1e-4;

  /** Mirrors Python's EnvConfig defaults. */
  const DEFAULT_CONFIG = deepFreeze({
    size: 64,
    max_steps: 200,
    max_step_px: 4.0,
    radius_min: 12.0,
    radius_max: 16.0,
    cursor_start_outside: true,
    end_on_success: true,
    auto_reset: true,
    bg_color: [20, 20, 30],
    target_color: [220, 40, 40],
    cursor_color: [255, 255, 255],
    cursor_outline: [0, 0, 0],
    success_reward: 10.0,
    step_cost: 0.001,
    bonuses: false,
    move_bonus: 0.01,
    move_budget_px: 200.0,
    click_bonus: 0.02,
    click_budget: 10,
    drives: false,
    h_init_min: 0.0,
    h_init_max: 0.5,
    h_rate: 0.005,
    h_relief: 1.0,
    hunger_cost: 0.0,
    b_init_min: 0.0,
    b_init_max: 0.5,
    b_rate: 0.02,
    b_relief: 0.1,
  });

  function deepFreeze(obj) {
    for (const v of Object.values(obj)) {
      if (v && typeof v === 'object') Object.freeze(v);
    }
    return Object.freeze(obj);
  }

  /** New config = defaults overridden by `overrides` (unknown keys throw). */
  function makeConfig(overrides) {
    const cfg = {};
    for (const k of Object.keys(DEFAULT_CONFIG)) {
      const v = DEFAULT_CONFIG[k];
      cfg[k] = Array.isArray(v) ? v.slice() : v;
    }
    if (overrides) {
      for (const k of Object.keys(overrides)) {
        if (!(k in DEFAULT_CONFIG)) throw new Error('unknown config key: ' + k);
        const v = overrides[k];
        cfg[k] = Array.isArray(v) ? v.slice() : v;
      }
    }
    return cfg;
  }

  // ---------------------------------------------------------------------------
  // Cursor sprite
  // ---------------------------------------------------------------------------

  /** (dx, dy) offsets of the white "+": {(0,0)} U {(+-k,0),(0,+-k) : k=1..3}. */
  const CURSOR_WHITE = (function () {
    const w = [[0, 0]];
    for (let k = 1; k <= 3; k++) w.push([k, 0], [-k, 0], [0, k], [0, -k]);
    return w;
  }());

  /** Outline offsets: (3x3 dilation of the white set) minus the white set. */
  const CURSOR_OUTLINE = (function () {
    const key = (x, y) => x + ',' + y;
    const white = new Set(CURSOR_WHITE.map(([x, y]) => key(x, y)));
    const seen = new Set();
    const out = [];
    for (const [x, y] of CURSOR_WHITE) {
      for (let a = -1; a <= 1; a++) {
        for (let b = -1; b <= 1; b++) {
          const k = key(x + a, y + b);
          if (!white.has(k) && !seen.has(k)) {
            seen.add(k);
            out.push([x + a, y + b]);
          }
        }
      }
    }
    return out;
  }());

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  function paintOffsets(out, base, size, hx, hy, offsets, rgb) {
    for (let k = 0; k < offsets.length; k++) {
      const px = hx + offsets[k][0];
      const py = hy + offsets[k][1];
      if (px >= 0 && px < size && py >= 0 && py < size) {
        const o = base + (py * size + px) * 3;
        out[o] = rgb[0];
        out[o + 1] = rgb[1];
        out[o + 2] = rgb[2];
      }
    }
  }

  /**
   * Render one state into `out` (RGB bytes) starting at byte offset `base`.
   * Background, then the circle (if alive), then the cursor sprite.
   */
  function renderInto(out, base, cursorX, cursorY, cx, cy, r, alive, config) {
    const size = config.size;
    const bg = config.bg_color;
    const fg = config.target_color;
    const r2 = r * r;
    let o = base;
    for (let i = 0; i < size; i++) {
      const ddy = (i + 0.5) - cy;
      const ddy2 = ddy * ddy;
      for (let j = 0; j < size; j++, o += 3) {
        const ddx = (j + 0.5) - cx;
        const c = (alive && ddx * ddx + ddy2 <= r2) ? fg : bg;
        out[o] = c[0];
        out[o + 1] = c[1];
        out[o + 2] = c[2];
      }
    }
    const hx = Math.floor(cursorX);
    const hy = Math.floor(cursorY);
    paintOffsets(out, base, size, hx, hy, CURSOR_OUTLINE, config.cursor_outline);
    paintOffsets(out, base, size, hx, hy, CURSOR_WHITE, config.cursor_color);
  }

  /** Flatten [[x, y], ...] or a flat array into a Float64Array of length 2n. */
  function flatPairs(xy, n) {
    if (xy.length === 2 * n && typeof xy[0] === 'number') return Float64Array.from(xy);
    const out = new Float64Array(2 * n);
    for (let e = 0; e < n; e++) {
      out[2 * e] = xy[e][0];
      out[2 * e + 1] = xy[e][1];
    }
    return out;
  }

  /**
   * Render n states.  Same contract as Python's render_state:
   *   cursorXY, targetXY: n pairs (nested or flat), radius: n numbers,
   *   targetAlive: n booleans/0-1.  Returns Uint8Array(n * size * size * 3).
   */
  function renderState(cursorXY, targetXY, radius, targetAlive, config) {
    const n = radius.length;
    const size = config.size;
    const cur = flatPairs(cursorXY, n);
    const tgt = flatPairs(targetXY, n);
    const out = new Uint8Array(n * size * size * 3);
    for (let e = 0; e < n; e++) {
      renderInto(out, e * size * size * 3, cur[2 * e], cur[2 * e + 1],
                 tgt[2 * e], tgt[2 * e + 1], radius[e], !!targetAlive[e], config);
    }
    return out;
  }

  /** Copy env `e` of an RGB batch into an RGBA buffer (e.g. ImageData.data). */
  function rgbToRgba(rgb, e, size, rgba) {
    const n = size * size;
    const base = e * n * 3;
    for (let p = 0; p < n; p++) {
      rgba[4 * p] = rgb[base + 3 * p];
      rgba[4 * p + 1] = rgb[base + 3 * p + 1];
      rgba[4 * p + 2] = rgb[base + 3 * p + 2];
      rgba[4 * p + 3] = 255;
    }
    return rgba;
  }

  // ---------------------------------------------------------------------------
  // Actions and RNG
  // ---------------------------------------------------------------------------

  /** Python sanitize_actions for one action: NaN -> 0, clip dx/dy, click > 0.5. */
  function sanitizeAction(dx, dy, c) {
    if (Number.isNaN(dx)) dx = 0;
    if (Number.isNaN(dy)) dy = 0;
    if (Number.isNaN(c)) c = 0;
    return [Math.min(Math.max(dx, -1), 1), Math.min(Math.max(dy, -1), 1), c > 0.5];
  }

  /** Small seeded PRNG returning doubles in [0, 1); state is one uint32. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    const next = function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    next.getState = () => a;
    next.setState = (s) => { a = s >>> 0; };
    return next;
  }

  function envSeed(seed, e) {
    const mix = mulberry32((Math.imul(seed >>> 0, 0x9E3779B1) ^ Math.imul(e + 1, 0x85EBCA77)) >>> 0);
    return Math.floor(mix() * 4294967296) >>> 0;
  }

  // ---------------------------------------------------------------------------
  // Simulator
  // ---------------------------------------------------------------------------

  /** State array name -> values per env (same names as Python STATE_SPEC). */
  const STATE_SPEC = {
    cursor: 2, target: 2, radius: 1, target_alive: 1, t: 1,
    hunger: 1, restlessness: 1, cum_move: 1, empty_clicks: 1, clicks: 1,
    ep_return: 1, path_length: 1, init_dist: 1, bonus_total: 1,
  };

  const EP_KEYS = ['ep_length', 'ep_return', 'ep_success', 'ep_clicks', 'ep_empty_clicks',
                   'ep_path_length', 'ep_init_dist', 'ep_bonus_total'];

  class CircleWorld {
    /**
     * @param {number} numEnvs  number of parallel envs
     * @param {object} config   EnvConfig-style overrides (copied)
     * @param {number} seed     uint32 seed for the reset PRNGs
     */
    constructor(numEnvs, config, seed) {
      this.numEnvs = Math.max(1, numEnvs | 0);
      this.config = makeConfig(config);
      this.seed = (seed || 0) >>> 0;
      this._rngs = [];
      for (let e = 0; e < this.numEnvs; e++) this._rngs.push(mulberry32(envSeed(this.seed, e)));
      for (const [k, w] of Object.entries(STATE_SPEC)) {
        this[k] = k === 'target_alive' ? new Uint8Array(this.numEnvs * w)
                                       : new Float64Array(this.numEnvs * w);
      }
      this._resetEnvs(range(this.numEnvs));
    }

    /** Start new episodes (all envs or the given indices); returns obs. */
    reset(envIds) {
      this._resetEnvs(envIds ? Array.from(envIds) : range(this.numEnvs));
      return this.observe();
    }

    _resetEnvs(indices) {
      const cfg = this.config;
      const size = cfg.size;
      for (const e of indices) {
        const rand = this._rngs[e];
        const u = [rand(), rand(), rand(), rand(), rand()];
        const r = cfg.radius_min + (cfg.radius_max - cfg.radius_min) * u[0];
        const cx = r + (size - 2 * r) * u[1];
        const cy = r + (size - 2 * r) * u[2];
        let x = 0;
        let y = 0;
        for (;;) {
          x = Math.min(rand() * size, size - CURSOR_EDGE_EPS);
          y = Math.min(rand() * size, size - CURSOR_EDGE_EPS);
          const dx = x - cx;
          const dy = y - cy;
          if (!cfg.cursor_start_outside || dx * dx + dy * dy > r * r) break;
        }
        this.cursor[2 * e] = x;
        this.cursor[2 * e + 1] = y;
        this.target[2 * e] = cx;
        this.target[2 * e + 1] = cy;
        this.radius[e] = r;
        this.target_alive[e] = 1;
        this.hunger[e] = cfg.h_init_min + (cfg.h_init_max - cfg.h_init_min) * u[3];
        this.restlessness[e] = cfg.b_init_min + (cfg.b_init_max - cfg.b_init_min) * u[4];
        this.init_dist[e] = Math.max(0, Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r);
        this.t[e] = 0;
        this.cum_move[e] = 0;
        this.empty_clicks[e] = 0;
        this.clicks[e] = 0;
        this.ep_return[e] = 0;
        this.path_length[e] = 0;
        this.bonus_total[e] = 0;
      }
    }

    /** Render the current state of all envs (RGB Uint8Array). */
    render() {
      const size = this.config.size;
      const out = new Uint8Array(this.numEnvs * size * size * 3);
      for (let e = 0; e < this.numEnvs; e++) this._renderEnv(out, e);
      return out;
    }

    _renderEnv(out, e) {
      const size = this.config.size;
      renderInto(out, e * size * size * 3, this.cursor[2 * e], this.cursor[2 * e + 1],
                 this.target[2 * e], this.target[2 * e + 1], this.radius[e],
                 this.target_alive[e] === 1, this.config);
    }

    _driveObs() {
      const d = new Float32Array(2 * this.numEnvs);
      if (this.config.drives) {
        for (let e = 0; e < this.numEnvs; e++) {
          d[2 * e] = this.hunger[e];
          d[2 * e + 1] = this.restlessness[e];
        }
      }
      return d;
    }

    /** {image: RGB Uint8Array, drives: Float32Array(2n)} for the current state. */
    observe() {
      return { image: this.render(), drives: this._driveObs() };
    }

    /**
     * Advance all envs by one action; mirrors Python CircleWorld.step.
     * @param actions  n actions [dx, dy, click] (nested) or a flat array of 3n
     * @returns {{obs, reward: Float32Array, done: Uint8Array, info: object}}
     */
    step(actions) {
      const cfg = this.config;
      const n = this.numEnvs;
      const a = new Float64Array(3 * n);
      if (actions.length === 3 * n && typeof actions[0] === 'number') {
        a.set(actions);
      } else {
        for (let e = 0; e < n; e++) for (let k = 0; k < 3; k++) a[3 * e + k] = actions[e][k];
      }
      const reward = new Float32Array(n);
      const done = new Uint8Array(n);
      const success = new Uint8Array(n);
      const timeout = new Uint8Array(n);
      const click = new Uint8Array(n);
      const emptyClick = new Uint8Array(n);
      const inside = new Uint8Array(n);
      const bonusOut = new Float64Array(n);
      const hi = cfg.size - CURSOR_EDGE_EPS;

      for (let e = 0; e < n; e++) {
        const [dx, dy, clk] = sanitizeAction(a[3 * e], a[3 * e + 1], a[3 * e + 2]);

        // 1. Move; displacement is the actual (clamped) motion.
        const ox = this.cursor[2 * e];
        const oy = this.cursor[2 * e + 1];
        const nx = Math.min(Math.max(ox + dx * cfg.max_step_px, 0), hi);
        const ny = Math.min(Math.max(oy + dy * cfg.max_step_px, 0), hi);
        const mx = nx - ox;
        const my = ny - oy;
        const disp = Math.sqrt(mx * mx + my * my);
        this.cursor[2 * e] = nx;
        this.cursor[2 * e + 1] = ny;

        // 2. Hit test at the new continuous position.
        const tx = nx - this.target[2 * e];
        const ty = ny - this.target[2 * e + 1];
        const r = this.radius[e];
        const ins = (tx * tx + ty * ty <= r * r) && this.target_alive[e] === 1;
        const suc = clk && ins;
        const emp = clk && !suc;

        // 3. Reward (budget factors use pre-step counters).
        const moveFactor = Math.max(0, 1 - this.cum_move[e] / cfg.move_budget_px);
        let rew = cfg.success_reward * (suc ? 1 : 0) - cfg.step_cost;
        let bonus = 0;
        if (cfg.bonuses) {
          const clickFactor = Math.max(0, 1 - this.empty_clicks[e] / cfg.click_budget);
          bonus = cfg.move_bonus * (disp / cfg.max_step_px) * moveFactor
                + (emp ? cfg.click_bonus * clickFactor : 0);
          rew = rew + bonus;
        }
        if (cfg.drives) rew = rew - cfg.hunger_cost * this.hunger[e];

        // 4. Counters and drives.
        this.cum_move[e] += disp;
        this.empty_clicks[e] += emp ? 1 : 0;
        this.clicks[e] += clk ? 1 : 0;
        this.path_length[e] += disp;
        this.hunger[e] = Math.min(Math.max(
          this.hunger[e] + cfg.h_rate - cfg.h_relief * (suc ? 1 : 0), 0), 1);
        this.restlessness[e] = Math.min(Math.max(
          this.restlessness[e] + cfg.b_rate - cfg.b_relief * (disp / cfg.max_step_px) * moveFactor,
          0), 1);
        this.t[e] += 1;
        this.ep_return[e] += rew;
        this.bonus_total[e] += bonus;

        // 5. A hit target disappears.
        if (suc) this.target_alive[e] = 0;

        // 6. Termination.
        const dn = (suc && cfg.end_on_success) || this.t[e] >= cfg.max_steps;

        reward[e] = rew;
        done[e] = dn ? 1 : 0;
        success[e] = suc ? 1 : 0;
        timeout[e] = dn && !suc ? 1 : 0;
        click[e] = clk ? 1 : 0;
        emptyClick[e] = emp ? 1 : 0;
        inside[e] = ins ? 1 : 0;
        bonusOut[e] = bonus;
      }

      // 7. Info from the post-step, pre-reset state.
      const terminalImage = this.render();
      const info = {
        terminal_image: terminalImage,
        terminal_drives: this._driveObs(),
        success: success,
        timeout: timeout,
        click: click,
        empty_click: emptyClick,
        first: new Uint8Array(n),
        bonus: bonusOut,
        // evaluation-only ground truth: never a model input
        gt_cursor: Float64Array.from(this.cursor),
        gt_target: Float64Array.from(this.target),
        gt_radius: Float64Array.from(this.radius),
        gt_target_alive: Uint8Array.from(this.target_alive),
        gt_inside: inside,
        gt_drives: new Float64Array(2 * n),
      };
      for (const k of EP_KEYS) info[k] = new Float64Array(n).fill(NaN);
      for (let e = 0; e < n; e++) {
        info.gt_drives[2 * e] = this.hunger[e];
        info.gt_drives[2 * e + 1] = this.restlessness[e];
        if (done[e]) {
          info.ep_length[e] = this.t[e];
          info.ep_return[e] = this.ep_return[e];
          info.ep_success[e] = this.target_alive[e] ? 0 : 1;
          info.ep_clicks[e] = this.clicks[e];
          info.ep_empty_clicks[e] = this.empty_clicks[e];
          info.ep_path_length[e] = this.path_length[e];
          info.ep_init_dist[e] = this.init_dist[e];
          info.ep_bonus_total[e] = this.bonus_total[e];
        }
      }

      // 8. Auto-reset: done envs return the first frame of their next episode.
      const image = Uint8Array.from(terminalImage);
      if (cfg.auto_reset) {
        const idx = [];
        for (let e = 0; e < n; e++) if (done[e]) idx.push(e);
        if (idx.length) {
          this._resetEnvs(idx);
          for (const e of idx) {
            this._renderEnv(image, e);
            info.first[e] = 1;
          }
        }
      }
      return { obs: { image: image, drives: this._driveObs() }, reward: reward, done: done, info: info };
    }

    /** Copy of the state (typed arrays) plus the PRNG states. */
    getState() {
      const s = {};
      for (const k of Object.keys(STATE_SPEC)) s[k] = this[k].slice();
      s.rng_state = Uint32Array.from(this._rngs.map((g) => g.getState()));
      return s;
    }

    /** Overwrite the given state arrays (flat or nested per-env values). */
    setState(state) {
      const n = this.numEnvs;
      for (const [k, v] of Object.entries(state)) {
        if (k === 'rng_state') {
          for (let e = 0; e < n; e++) this._rngs[e].setState(v[e]);
          continue;
        }
        if (!(k in STATE_SPEC)) throw new Error('unknown state key: ' + k);
        const w = STATE_SPEC[k];
        const flat = (w === 2) ? flatPairs(v, n) : Array.from(v, Number);
        if (flat.length !== n * w) throw new Error('state ' + k + ' has wrong length');
        for (let q = 0; q < n * w; q++) this[k][q] = flat[q];
      }
    }
  }

  function range(n) {
    return Array.from({ length: n }, (_, i) => i);
  }

  return {
    VERSION: 1,
    CURSOR_EDGE_EPS: CURSOR_EDGE_EPS,
    DEFAULT_CONFIG: DEFAULT_CONFIG,
    CURSOR_WHITE: CURSOR_WHITE,
    CURSOR_OUTLINE: CURSOR_OUTLINE,
    makeConfig: makeConfig,
    renderState: renderState,
    renderInto: renderInto,
    rgbToRgba: rgbToRgba,
    sanitizeAction: sanitizeAction,
    mulberry32: mulberry32,
    CircleWorld: CircleWorld,
  };
}));
