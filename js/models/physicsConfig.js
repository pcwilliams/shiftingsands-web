// Physics constants and the per-mode rules. Pure module.
//
// The three modes mirror the iOS app's CPU / GPU / Metal:
//   cpu     JavaScript on the main thread
//   gpu     WebGPU compute, then positions are read back to JavaScript and
//           uploaded again for drawing (the round trip the app's GPU mode had)
//   direct  WebGPU compute feeding the draw call on the GPU (the app's Metal
//           mode): particle data never visits JavaScript

export const MODES = Object.freeze(['cpu', 'gpu', 'direct']);

export const MODE_INFO = Object.freeze({
  cpu:    { label: 'CPU',    min: 50, max: 2000,   defaultCount: 250,   needsWebGPU: false },
  gpu:    { label: 'GPU',    min: 50, max: 20000,  defaultCount: 5000,  needsWebGPU: true },
  direct: { label: 'Direct', min: 50, max: 200000, defaultCount: 10000, needsWebGPU: true },
});

/** Broadphase: how each particle finds the others it might touch. */
export const BROADPHASES = Object.freeze(['grid', 'brute']);

/** Above this, the all-pairs kernel is too slow to be worth offering. */
export const BRUTE_FORCE_MAX = 50000;

export const PHYSICS = Object.freeze({
  restitution: 0.02,
  friction: 0.6,
  neckHalfHeight: 0.10,
  flowDamping: 0.92,     // per second, applied while moving
  settleDamping: 0.05,   // per second, applied when nearly at rest
  settleSpeed: 0.15,     // speed at which damping is fully "flow"
  maxDt: 1 / 30,         // clamp so a dropped frame can't blow the pile up
  flipDuration: 1.2,
});

export const DURATION = Object.freeze({ min: 5, max: 30, step: 5, default: 5, presets: [5, 10, 30] });
export const SIZE = Object.freeze({ min: 0.5, max: 1.5, step: 0.05, default: 1 });

/** The hourglass is about one unit tall, so gravity is scaled to be watchable. */
export function gravityScale(duration) {
  return Math.max(5 / duration, 0.15);
}

/** Extra damping around the neck, which is what actually sets the flow rate. */
export function neckDamping(duration) {
  return duration * 0.02;
}

export function cpuSubsteps(count) {
  if (count > 1000) return 2;
  if (count > 500) return 3;
  return 4;
}

/**
 * The app trades substeps for count, because all-pairs costs N² per substep.
 * The grid could afford more, but both broadphases use the same table so that
 * switching between them changes the speed and nothing else.
 */
export function gpuSubsteps(count) {
  if (count > 10000) return 1;
  if (count > 3000) return 2;
  if (count > 1500) return 3;
  return 4;
}

export function maxCount(mode, broadphase = 'grid') {
  const max = MODE_INFO[mode].max;
  return mode !== 'cpu' && broadphase === 'brute' ? Math.min(max, BRUTE_FORCE_MAX) : max;
}

export function clampCount(mode, count, broadphase = 'grid') {
  const { min } = MODE_INFO[mode];
  const n = Math.round(Number(count) / 50) * 50;
  if (!Number.isFinite(n)) return MODE_INFO[mode].defaultCount;
  return Math.min(Math.max(n, min), maxCount(mode, broadphase));
}

export function clampSize(size) {
  const s = Number(size);
  if (!Number.isFinite(s)) return SIZE.default;
  return Math.min(Math.max(Math.round(s / SIZE.step) * SIZE.step, SIZE.min), SIZE.max);
}

export function clampDuration(duration) {
  const d = Number(duration);
  if (!Number.isFinite(d)) return DURATION.default;
  return Math.min(Math.max(Math.round(d / DURATION.step) * DURATION.step, DURATION.min), DURATION.max);
}

/**
 * Collision search work per frame, for the on-screen stats.
 * CPU: each pair once. Brute-force GPU: every thread scans everyone.
 * Grid: every thread looks in 27 buckets, whatever N is.
 */
export function searchWorkPerFrame(mode, count, broadphase) {
  if (mode === 'cpu') {
    return { value: (count * (count - 1) / 2) * cpuSubsteps(count), unit: 'pair tests' };
  }
  const substeps = gpuSubsteps(count);
  return broadphase === 'grid'
    ? { value: count * 27 * substeps, unit: 'bucket lookups' }
    : { value: count * count * substeps, unit: 'pair tests' };
}

/**
 * The count slider is logarithmic: 50 to 200,000 on a linear track would put
 * everything interesting in the first few pixels. Positions run 0..1000.
 */
export function sliderToCount(position, mode, broadphase = 'grid') {
  const { min } = MODE_INFO[mode];
  const max = maxCount(mode, broadphase);
  const t = Math.min(Math.max(position / 1000, 0), 1);
  const raw = min * Math.pow(max / min, t);
  // Snap to about two significant figures, and never finer than 50.
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)) - 1);
  const stepSize = Math.max(50, magnitude * 5);
  return clampCount(mode, Math.round(raw / stepSize) * stepSize, broadphase);
}

export function countToSlider(count, mode, broadphase = 'grid') {
  const { min } = MODE_INFO[mode];
  const max = maxCount(mode, broadphase);
  const c = Math.min(Math.max(count, min), max);
  return Math.round((Math.log(c / min) / Math.log(max / min)) * 1000);
}

export function formatCount(count) {
  if (count < 1000) return String(count);
  const k = count / 1000;
  return Number.isInteger(k) ? `${k}k` : `${k.toFixed(k < 10 ? 2 : 1).replace(/0+$/, '').replace(/\.$/, '')}k`;
}

/** Icosphere detail for the sand mesh: fewer triangles each as the count rises. */
export function sphereDetail(count) {
  if (count <= 2000) return 2;   // 320 triangles
  if (count <= 60000) return 1;  // 80 triangles, as in the app's Metal mode
  return 0;                      // 20 triangles
}

export function easeInOut(t) {
  const x = Math.min(Math.max(t, 0), 1);
  return x * x * (3 - 2 * x);
}
