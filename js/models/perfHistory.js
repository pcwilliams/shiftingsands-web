// A short rolling history of per-frame timings, for the chart in Stats.
// Two series on one axis (milliseconds): physics and drawing. Pure module.

export class PerfHistory {
  constructor(capacity = 240) {
    this.capacity = capacity;
    this.physics = new Float32Array(capacity);
    this.drawing = new Float32Array(capacity);
    this.reset();
  }

  /** Forget everything, including the per-run maxima. */
  reset() {
    this.length = 0;
    this.head = 0;            // index the next sample is written to
    this.maxPhysics = 0;
    this.maxDrawing = 0;
  }

  push(physicsMs, drawingMs) {
    const p = Number.isFinite(physicsMs) ? Math.max(physicsMs, 0) : 0;
    const d = Number.isFinite(drawingMs) ? Math.max(drawingMs, 0) : 0;
    this.physics[this.head] = p;
    this.drawing[this.head] = d;
    this.head = (this.head + 1) % this.capacity;
    this.length = Math.min(this.length + 1, this.capacity);
    this.maxPhysics = Math.max(this.maxPhysics, p);
    this.maxDrawing = Math.max(this.maxDrawing, d);
  }

  /** Samples oldest first. */
  series(name) {
    const source = this[name];
    const out = new Array(this.length);
    const start = (this.head - this.length + this.capacity) % this.capacity;
    for (let i = 0; i < this.length; i++) out[i] = source[(start + i) % this.capacity];
    return out;
  }

  get latest() {
    if (this.length === 0) return { physics: 0, drawing: 0 };
    const i = (this.head - 1 + this.capacity) % this.capacity;
    return { physics: this.physics[i], drawing: this.drawing[i] };
  }

  /** Largest value currently in the window, across both series. */
  get windowMax() {
    let max = 0;
    for (const name of ['physics', 'drawing']) {
      for (const v of this.series(name)) max = Math.max(max, v);
    }
    return max;
  }
}

/**
 * Top of the chart's axis: a round number at or above the window's largest
 * value, never below `floor` so a quiet trace doesn't fill the chart with noise.
 */
export function niceCeiling(value, floor = 20) {
  const v = Math.max(value, floor);
  const magnitude = Math.pow(10, Math.floor(Math.log10(v)));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (step * magnitude >= v) return step * magnitude;
  }
  return 10 * magnitude;
}
