#!/usr/bin/env node
/**
 * render_cli.js: render and step fixture states with circle_world.js.
 *
 * Usage:  node render_cli.js fixtures.json results.json
 *
 * fixtures.json:
 *   {"fixtures": [{"id": "...",
 *                  "config": {EnvConfig overrides},
 *                  "state": {cursor: [x, y], target: [x, y], radius: r,
 *                            target_alive: bool, t, hunger, restlessness,
 *                            cum_move, empty_clicks, clicks, ep_return,
 *                            path_length, init_dist, bonus_total},
 *                  "actions": [[dx, dy, click], ...]}]}       (optional)
 *   Non-finite numbers in actions are given as the strings "NaN",
 *   "Infinity" and "-Infinity" (JSON has no literal for them).
 *
 * results.json:
 *   {"results": [{"id", "image0", "steps": [{image, cursor, reward, drives,
 *                 done, success, timeout, click, empty_click, first, bonus,
 *                 state: {...}, ep: {...}}]}]}
 *   Images are base64(zlib-deflate(raw RGB bytes, size*size*3)).  Episode
 *   stats that are NaN (env not done) are written as null.
 *
 * Each fixture is simulated as a one-env world.  Fixtures normally set
 * auto_reset=false, so no random resets happen and every step's image is
 * the post-step (terminal) frame.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const CW = require(path.join(__dirname, 'circle_world.js'));

const SCALAR_STATE = ['radius', 'target_alive', 't', 'hunger', 'restlessness', 'cum_move',
                      'empty_clicks', 'clicks', 'ep_return', 'path_length', 'init_dist',
                      'bonus_total'];
const EP_KEYS = ['ep_length', 'ep_return', 'ep_success', 'ep_clicks', 'ep_empty_clicks',
                 'ep_path_length', 'ep_init_dist', 'ep_bonus_total'];

function num(v) {
  return typeof v === 'string' ? Number(v) : v;
}

function encodeImage(u8) {
  return zlib.deflateSync(Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength)).toString('base64');
}

function finiteOrNull(x) {
  return Number.isFinite(x) ? x : null;
}

function runFixture(fx) {
  const world = new CW.CircleWorld(1, fx.config || {}, 0);
  const st = fx.state || {};
  const state = {};
  if (st.cursor) state.cursor = st.cursor;
  if (st.target) state.target = st.target;
  for (const k of SCALAR_STATE) if (k in st) state[k] = [k === 'target_alive' ? (st[k] ? 1 : 0) : st[k]];
  world.setState(state);

  const result = { id: fx.id, image0: encodeImage(world.observe().image), steps: [] };
  for (const action of fx.actions || []) {
    const out = world.step([action.map(num)]);
    const info = out.info;
    const s = world.getState();
    const ep = {};
    for (const k of EP_KEYS) ep[k] = finiteOrNull(info[k][0]);
    result.steps.push({
      image: encodeImage(info.terminal_image),
      obs_image: encodeImage(out.obs.image),
      cursor: [info.gt_cursor[0], info.gt_cursor[1]],
      reward: out.reward[0],
      drives: [info.terminal_drives[0], info.terminal_drives[1]],
      obs_drives: [out.obs.drives[0], out.obs.drives[1]],
      done: out.done[0] === 1,
      success: info.success[0] === 1,
      timeout: info.timeout[0] === 1,
      click: info.click[0] === 1,
      empty_click: info.empty_click[0] === 1,
      first: info.first[0] === 1,
      inside: info.gt_inside[0] === 1,
      bonus: info.bonus[0],
      state: {
        target_alive: info.gt_target_alive[0] === 1,
        hunger: info.gt_drives[0],
        restlessness: info.gt_drives[1],
        t: s.t[0],
        cum_move: s.cum_move[0],
        empty_clicks: s.empty_clicks[0],
        clicks: s.clicks[0],
        ep_return: s.ep_return[0],
        path_length: s.path_length[0],
        bonus_total: s.bonus_total[0],
      },
      ep: ep,
    });
  }
  return result;
}

function main(argv) {
  if (argv.length < 2) {
    process.stderr.write('usage: node render_cli.js fixtures.json results.json\n');
    process.exit(2);
  }
  const doc = JSON.parse(fs.readFileSync(argv[0], 'utf8'));
  const results = doc.fixtures.map(runFixture);
  fs.writeFileSync(argv[1], JSON.stringify({ version: CW.VERSION, results: results }));
}

main(process.argv.slice(2));
