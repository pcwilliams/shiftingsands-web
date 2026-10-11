// Persisted settings: the web version of the app's UserDefaults.
//
// Each mode remembers its own particle count and size, because 200,000 is a
// sensible count for one mode and a frozen tab in another. Storage is
// injected so tests can use an in-memory map.

import {
  MODES, MODE_INFO, BROADPHASES, DURATION, SIZE,
  clampCount, clampSize, clampDuration,
} from '../models/physicsConfig.js';

const PREFIX = 'shiftingsands.';

export function memoryStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
  };
}

/** localStorage can throw in private modes, so probe it and fall back. */
export function defaultStorage() {
  try {
    const s = globalThis.localStorage;
    const probe = `${PREFIX}probe`;
    s.setItem(probe, '1');
    s.removeItem(probe);
    return s;
  } catch {
    return memoryStorage();
  }
}

export class SettingsStore {
  #storage;
  #listeners = new Set();

  constructor(storage = defaultStorage()) {
    this.#storage = storage;
    const mode = this.#read('mode');
    this.mode = MODES.includes(mode) ? mode : 'direct';
    const broadphase = this.#read('broadphase');
    this.broadphase = BROADPHASES.includes(broadphase) ? broadphase : 'grid';
    const duration = Number(this.#read('duration'));
    this.duration = duration >= DURATION.min && duration <= DURATION.max
      ? clampDuration(duration) : DURATION.default;
    this.randomColors = this.#read('randomColors') === 'true';
    this.count = this.#savedCount(this.mode);
    this.size = this.#savedSize(this.mode);
  }

  #read(key) {
    try { return this.#storage.getItem(PREFIX + key); } catch { return null; }
  }

  #write(key, value) {
    try { this.#storage.setItem(PREFIX + key, String(value)); } catch { /* best effort */ }
  }

  #savedCount(mode) {
    const saved = Number(this.#read(`count.${mode}`));
    return saved >= MODE_INFO[mode].min
      ? clampCount(mode, saved, this.broadphase)
      : clampCount(mode, MODE_INFO[mode].defaultCount, this.broadphase);
  }

  #savedSize(mode) {
    const saved = Number(this.#read(`size.${mode}`));
    return saved >= SIZE.min && saved <= SIZE.max ? clampSize(saved) : SIZE.default;
  }

  subscribe(fn) {
    this.#listeners.add(fn);
    return () => this.#listeners.delete(fn);
  }

  #emit(change) {
    for (const fn of this.#listeners) fn(this, change);
  }

  /** Switching mode recalls that mode's own count and size. */
  setMode(mode, { persist = true } = {}) {
    if (!MODES.includes(mode) || mode === this.mode) return;
    this.mode = mode;
    if (persist) this.#write('mode', mode);
    this.count = this.#savedCount(mode);
    this.size = this.#savedSize(mode);
    this.#emit('mode');
  }

  setBroadphase(broadphase) {
    if (!BROADPHASES.includes(broadphase) || broadphase === this.broadphase) return;
    this.broadphase = broadphase;
    this.#write('broadphase', broadphase);
    this.count = clampCount(this.mode, this.count, broadphase);
    this.#emit('broadphase');
  }

  setCount(count) {
    const next = clampCount(this.mode, count, this.broadphase);
    if (next === this.count) return;
    this.count = next;
    this.#write(`count.${this.mode}`, next);
    this.#emit('count');
  }

  setSize(size) {
    const next = clampSize(size);
    if (next === this.size) return;
    this.size = next;
    this.#write(`size.${this.mode}`, next);
    this.#emit('size');
  }

  setDuration(duration) {
    const next = clampDuration(duration);
    if (next === this.duration) return;
    this.duration = next;
    this.#write('duration', next);
    this.#emit('duration');
  }

  setRandomColors(on) {
    const next = Boolean(on);
    if (next === this.randomColors) return;
    this.randomColors = next;
    this.#write('randomColors', next);
    this.#emit('randomColors');
  }
}
