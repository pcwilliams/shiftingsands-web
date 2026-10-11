// Timer state machine: a port of TimerViewModel.swift's timing half.
//
//   idle --startFlip--> flipping --completeFlip--> running --(elapsed)--> complete
//     ^                     |                         |                      |
//     +------- reset -------+-------------------------+----------------------+
//
// The start is two-phase because the 1.2 s flip has to finish before the
// countdown begins. There is no timer of its own here: the render loop calls
// tick(now) each frame. Pure module; time is passed in so tests control it.

export class TimerModel {
  #listeners = new Set();
  #startedAt = 0;

  constructor(duration = 5) {
    this.duration = duration;
    this.elapsed = 0;
    this.state = 'idle';
  }

  get isIdle() { return this.state === 'idle' || this.state === 'complete'; }
  get isFlipping() { return this.state === 'flipping'; }
  get isRunning() { return this.state === 'running'; }
  get isComplete() { return this.state === 'complete'; }

  get progress() {
    return this.duration > 0 ? Math.min(this.elapsed / this.duration, 1) : 0;
  }

  get remainingText() {
    const remaining = Math.max(this.duration - this.elapsed, 0);
    const whole = Math.floor(remaining);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
  }

  get buttonLabel() {
    return this.isFlipping || this.isRunning ? 'Reset' : 'Start';
  }

  subscribe(fn) {
    this.#listeners.add(fn);
    return () => this.#listeners.delete(fn);
  }

  #emit() {
    for (const fn of this.#listeners) fn(this);
  }

  setDuration(duration) {
    if (!this.isIdle) return false;
    this.duration = duration;
    this.elapsed = 0;
    this.state = 'idle';
    this.#emit();
    return true;
  }

  /** Phase 1: the hourglass starts turning. The clock is not running yet. */
  startFlip() {
    if (this.isFlipping) return false;
    this.state = 'flipping';
    this.elapsed = 0;
    this.#emit();
    return true;
  }

  /** Phase 2: the flip has landed, so the countdown starts now. */
  completeFlip(now) {
    if (!this.isFlipping) return false;
    this.state = 'running';
    this.#startedAt = now;
    this.elapsed = 0;
    this.#emit();
    return true;
  }

  /** Call once per frame with a monotonic time in seconds. */
  tick(now) {
    if (!this.isRunning) return;
    const elapsed = now - this.#startedAt;
    const before = this.remainingText;
    if (elapsed >= this.duration) {
      this.elapsed = this.duration;
      this.state = 'complete';
      this.#emit();
      return;
    }
    this.elapsed = elapsed;
    if (this.remainingText !== before) this.#emit();
  }

  reset() {
    this.state = 'idle';
    this.elapsed = 0;
    this.#emit();
  }
}
