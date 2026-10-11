// URL parameters: the web's launch arguments.
//
//   ?mode=cpu|gpu|direct   physics mode ("metal" is accepted for "direct")
//   ?count=N               particle count
//   ?size=F                size multiplier, 0.5 to 1.5
//   ?duration=S            timer length in seconds, 5 to 30
//   ?broadphase=grid|brute GPU collision search
//   ?colors                random colour per particle
//   ?autostart             flip two seconds after load
//   ?test                  10 s run, auto-start, results on window.shiftingSands.results
//   ?seed=N                seed the spawn/colour randomness
//   ?stats                 show the stats readout on load
//
// Every field is null when absent, so "no override" is distinguishable from a
// value. They are applied after the saved settings load, so the URL wins.
// Pure module.

import { MODES, BROADPHASES } from '../models/physicsConfig.js';

const MODE_ALIASES = { metal: 'direct', webgpu: 'direct', readback: 'gpu' };

function flag(params, name) {
  if (!params.has(name)) return null;
  const v = params.get(name).toLowerCase();
  return !(v === '0' || v === 'false' || v === 'off' || v === 'no');
}

function number(params, name) {
  if (!params.has(name)) return null;
  const raw = params.get(name).trim();
  if (raw === '') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function parseLaunchArgs(search = '') {
  const params = new URLSearchParams(search);

  let mode = params.get('mode')?.toLowerCase() ?? null;
  if (mode && MODE_ALIASES[mode]) mode = MODE_ALIASES[mode];
  if (!MODES.includes(mode)) mode = null;

  let broadphase = params.get('broadphase')?.toLowerCase() ?? null;
  if (broadphase === 'n2' || broadphase === 'bruteforce') broadphase = 'brute';
  if (!BROADPHASES.includes(broadphase)) broadphase = null;

  const seed = number(params, 'seed');

  return {
    mode,
    broadphase,
    count: number(params, 'count'),
    size: number(params, 'size'),
    duration: number(params, 'duration'),
    colors: flag(params, 'colors') ?? flag(params, 'multicolor'),
    autostart: flag(params, 'autostart') === true,
    test: flag(params, 'test') === true,
    stats: flag(params, 'stats') === true,
    seed: seed == null ? null : Math.trunc(seed),
  };
}

/**
 * Apply overrides to a SettingsStore. Mode goes first because changing mode
 * recalls that mode's saved count and size, which count/size must then beat.
 * URL values are not persisted for mode, so a test link doesn't change what
 * the next plain visit opens with.
 */
export function applyLaunchArgs(args, store) {
  if (args.broadphase) store.setBroadphase(args.broadphase);
  if (args.mode) store.setMode(args.mode, { persist: false });
  if (args.count != null) store.setCount(args.count);
  if (args.size != null) store.setSize(args.size);
  if (args.test) store.setDuration(10);
  else if (args.duration != null) store.setDuration(args.duration);
  if (args.colors != null) store.setRandomColors(args.colors);
}
