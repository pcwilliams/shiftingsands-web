// Per-frame timings for one whole run, for the chart in Stats.
//
// The chart's time axis is fixed: it spans the run from the moment Start is
// pressed (the flip, the countdown, and a little settling time after it) and
// fills in from the left as the run goes. Nothing scrolls, so afterwards the
// whole run can be read in one look. Two series share one axis
// (milliseconds): physics and drawing. Pure module.

/** Seconds recorded after the countdown ends, while the sand settles. */
export const SETTLE_TAIL = 3;

export class RunTimeline {
  constructor() {
    this.clear();
  }

  /** Empty, with no run under way. */
  clear() {
    this.span = 0;            // seconds covered by the axis
    this.marks = [];          // [{ t, label }]
    this.times = [];
    this.physics = [];
    this.drawing = [];
    this.maxPhysics = 0;
    this.maxDrawing = 0;
    this.startedAt = null;
  }

  /**
   * Begin a run.
   * @param {number} now       seconds, any monotonic clock
   * @param {number} flip      length of the flip, seconds
   * @param {number} duration  length of the countdown, seconds
   */
  begin(now, flip, duration) {
    this.clear();
    this.startedAt = now;
    this.span = flip + duration + SETTLE_TAIL;
    this.marks = [
      { t: flip, label: 'flip' },
      { t: flip + duration, label: '0:00' },
    ];
  }

  get recording() {
    return this.startedAt !== null;
  }

  /** Add this frame's timings. Ignored outside a run; ends the run at the span. */
  push(now, physicsMs, drawingMs) {
    if (!this.recording) return;
    const t = now - this.startedAt;
    if (t > this.span) {
      this.startedAt = null;
      return;
    }
    const p = Number.isFinite(physicsMs) ? Math.max(physicsMs, 0) : 0;
    const d = Number.isFinite(drawingMs) ? Math.max(drawingMs, 0) : 0;
    this.times.push(Math.max(t, 0));
    this.physics.push(p);
    this.drawing.push(d);
    this.maxPhysics = Math.max(this.maxPhysics, p);
    this.maxDrawing = Math.max(this.maxDrawing, d);
  }

  get length() {
    return this.times.length;
  }

  get latest() {
    const i = this.length - 1;
    return i < 0 ? { physics: 0, drawing: 0 } : { physics: this.physics[i], drawing: this.drawing[i] };
  }

  /**
   * One value per chart column: the largest sample that falls in the column,
   * so a single slow frame is never averaged away. Columns the run has not
   * reached yet (or where no frame landed) are null.
   */
  columns(name, count) {
    const out = new Array(count).fill(null);
    if (this.span <= 0) return out;
    const source = this[name];
    for (let i = 0; i < source.length; i++) {
      const column = Math.min(count - 1, Math.floor((this.times[i] / this.span) * count));
      if (out[column] === null || source[i] > out[column]) out[column] = source[i];
    }
    return out;
  }
}

/**
 * Top of the chart's axis: a round number at or above the largest value,
 * never below `floor` so a quiet trace doesn't fill the chart with noise.
 */
export function niceCeiling(value, floor = 20) {
  const v = Math.max(value, floor);
  const magnitude = Math.pow(10, Math.floor(Math.log10(v)));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (step * magnitude >= v) return step * magnitude;
  }
  return 10 * magnitude;
}

/**
 * Position (0 at the bottom, 1 at the top) of a value on a square-root axis.
 * Frame times are mostly a few milliseconds with rare spikes of hundreds; on
 * a straight axis either the spikes run off the top or the rest is a flat
 * line. The square root keeps both readable.
 */
export function sqrtPosition(value, top) {
  if (top <= 0) return 0;
  return Math.sqrt(Math.min(Math.max(value, 0), top) / top);
}
