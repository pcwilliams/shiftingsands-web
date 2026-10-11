// Particle sizing and spawn packing: a port of the static helpers on
// GranularSimulation.swift. Pure module.

import { CHAMBER_BOTTOM, WALL_CLEARANCE } from './hourglassProfile.js';

/** Floats per particle. Same layout as the GPU struct: see PARTICLE_LAYOUT. */
export const STRIDE = 8;

/**
 * One particle is 8 floats = 32 bytes:
 *   [0..2] position   [3] radius
 *   [4..6] velocity   [7] sleep counter
 * The CPU simulation, the WGSL struct and the instanced vertex buffer all use
 * this layout, so the same bytes can move between them without conversion.
 */
export const PARTICLE_LAYOUT = Object.freeze({
  X: 0, Y: 1, Z: 2, RADIUS: 3, VX: 4, VY: 5, VZ: 6, SLEEP: 7,
});

/**
 * Radius that keeps the settled pile at the same height for any count.
 *
 * r ∝ N^(-1/3) keeps the total sphere volume constant. Smaller spheres also
 * pack closer to the glass, so each layer holds more and the pile ends up
 * shorter; the (effR / refEffR)^(2/3) factor corrects for that. Calibrated at
 * 250 particles, as in the app.
 */
export function radiusForCount(count, sizeMultiplier = 1) {
  const baseR = 0.030 * Math.cbrt(100 / Math.max(count, 1));
  const nominalGlassR = 0.147;
  const effR = nominalGlassR - baseR;
  const refBaseR = 0.030 * Math.cbrt(100 / 250);
  const refEffR = nominalGlassR - refBaseR;
  if (effR <= 0.01) return baseR * sizeMultiplier;
  return baseR * Math.pow(effR / refEffR, 2 / 3) * sizeMultiplier;
}

/** Small seeded PRNG so tests and screenshots are reproducible. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Hex close-packed layers filling the lower chamber from the floor up.
 *
 * Starting near the resting state means no settling pass is needed. If the
 * lattice runs out of room below y = -0.10, the rest are scattered in the
 * lower chamber and left to settle; nothing is ever placed near the neck.
 *
 * Returns a Float32Array of STRIDE floats per particle, velocities zero.
 */
export function packedParticles(count, particleRadius, profile, random = Math.random) {
  const out = new Float32Array(Math.max(count, 0) * STRIDE);
  const spacing = particleRadius * 2;
  const vertSpacing = spacing * 0.866;
  const maxY = -0.10;
  let n = 0;
  const put = (x, y, z) => {
    const o = n * STRIDE;
    out[o] = x; out[o + 1] = y; out[o + 2] = z; out[o + 3] = particleRadius;
    n++;
  };

  let y = CHAMBER_BOTTOM + particleRadius;
  let layer = 0;
  while (n < count && y < maxY) {
    const glassR = profile.innerRadiusAt(y) - particleRadius - WALL_CLEARANCE;
    if (glassR > particleRadius) {
      const xOffset = layer % 2 === 0 ? 0 : spacing * 0.5;
      const steps = Math.ceil(glassR / spacing);
      for (let ix = -steps; ix <= steps && n < count; ix++) {
        for (let iz = -steps; iz <= steps && n < count; iz++) {
          const x = ix * spacing + xOffset;
          const z = iz * spacing;
          if (Math.hypot(x, z) < glassR) put(x, y, z);
        }
      }
    }
    y += vertSpacing;
    layer++;
  }

  const minY = CHAMBER_BOTTOM + particleRadius;
  while (n < count) {
    const ry = minY + random() * (maxY - minY);
    const rGlass = Math.max(profile.innerRadiusAt(ry) - particleRadius - WALL_CLEARANCE, 0.001);
    const rr = random() * rGlass;
    const angle = random() * Math.PI * 2;
    put(rr * Math.cos(angle), ry, rr * Math.sin(angle));
  }
  return out;
}

/** HSB (all 0..1) to RGB (0..1). */
export function hsbToRgb(h, s, b) {
  const c = b * s;
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
  const m = b - c;
  const sector = Math.floor(h * 6) % 6;
  const [r, g, bl] = [
    [c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x],
  ][sector];
  return [r + m, g + m, bl + m];
}

export const SAND_RGB = Object.freeze([0.76, 0.60, 0.28]);

/** One RGBA8 colour per particle: golden sand, or a random hue each. */
export function particleColors(count, randomColors, random = Math.random) {
  const out = new Uint8Array(Math.max(count, 1) * 4);
  for (let i = 0; i < count; i++) {
    const [r, g, b] = randomColors ? hsbToRgb(random(), 0.7, 0.85) : SAND_RGB;
    out[i * 4] = Math.round(r * 255);
    out[i * 4 + 1] = Math.round(g * 255);
    out[i * 4 + 2] = Math.round(b * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}
