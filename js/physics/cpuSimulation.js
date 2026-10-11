// CPU granular physics: a port of GranularSimulation.swift.
//
// Single-threaded and sequential. Each collision is resolved in place, so a
// correction made for pair (0,1) is already visible when pair (1,2) is tested.
// That "Gauss-Seidel" ordering is why a pile settles cleanly here with none of
// the sleep machinery the GPU kernel needs.
//
// State lives in one Float32Array in the shared 8-float layout, which is also
// the instanced vertex buffer, so there is nothing to copy before drawing.
// Pure module: no DOM, no three.js.

import { CHAMBER_BOTTOM, CHAMBER_TOP, WALL_CLEARANCE } from '../models/hourglassProfile.js';
import { STRIDE, packedParticles } from '../models/packing.js';
import { PHYSICS, cpuSubsteps } from '../models/physicsConfig.js';

export class CpuSimulation {
  /**
   * @param {object} opts
   * @param {number} opts.count
   * @param {number} opts.particleRadius
   * @param {object} opts.profile       from createProfile()
   * @param {Float32Array} [opts.target] write into this array instead of allocating
   * @param {() => number} [opts.random]
   */
  constructor({ count, particleRadius, profile, target = null, random = Math.random }) {
    this.count = count;
    this.particleRadius = particleRadius;
    this.profile = profile;
    this.gravity = 1;
    this.neckDamping = 0;
    this.containerEulerX = 0;
    const packed = packedParticles(count, particleRadius, profile, random);
    if (target) {
      target.set(packed);
      this.data = target;
    } else {
      this.data = packed;
    }
  }

  get substeps() {
    return cpuSubsteps(this.count);
  }

  setParticle(i, { position, velocity = [0, 0, 0], radius = this.particleRadius }) {
    const o = i * STRIDE;
    this.data.set([position[0], position[1], position[2], radius,
                   velocity[0], velocity[1], velocity[2], 0], o);
  }

  getParticle(i) {
    const d = this.data;
    const o = i * STRIDE;
    return {
      position: [d[o], d[o + 1], d[o + 2]],
      radius: d[o + 3],
      velocity: [d[o + 4], d[o + 5], d[o + 6]],
    };
  }

  step(dt) {
    const n = this.count;
    if (n === 0) return;
    const d = this.data;
    const substeps = this.substeps;
    const subDt = dt / substeps;

    // World gravity (0, -g, 0) seen from inside the rotating container.
    const theta = this.containerEulerX;
    const gy = -this.gravity * Math.cos(theta);
    const gz = this.gravity * Math.sin(theta);

    const flowDamp = Math.pow(PHYSICS.flowDamping, subDt);
    const settleDamp = Math.pow(PHYSICS.settleDamping, subDt);
    const { neckHalfHeight, settleSpeed } = PHYSICS;

    for (let s = 0; s < substeps; s++) {
      // 1 + 2. Gravity, then move.
      for (let o = 0, end = n * STRIDE; o < end; o += STRIDE) {
        d[o + 5] += gy * subDt;
        d[o + 6] += gz * subDt;
        d[o] += d[o + 4] * subDt;
        d[o + 1] += d[o + 5] * subDt;
        d[o + 2] += d[o + 6] * subDt;
      }

      this.#resolveWalls();
      this.#resolveFloorCeiling();
      this.#resolveSpheres();

      // 6 + 7. Neck friction, then speed-dependent damping.
      for (let o = 0, end = n * STRIDE; o < end; o += STRIDE) {
        let vx = d[o + 4], vy = d[o + 5], vz = d[o + 6];
        if (this.neckDamping > 0) {
          const neckDist = Math.abs(d[o + 1]);
          if (neckDist < neckHalfHeight) {
            const factor = 1 - neckDist / neckHalfHeight;
            const damp = Math.max(0, 1 - factor * this.neckDamping * subDt);
            vx *= damp; vy *= damp; vz *= damp;
          }
        }
        const speed = Math.hypot(vx, vy, vz);
        const t = Math.min(speed / settleSpeed, 1);
        const damp = settleDamp + (flowDamp - settleDamp) * t;
        d[o + 4] = vx * damp;
        d[o + 5] = vy * damp;
        d[o + 6] = vz * damp;
      }
    }
  }

  // The glass is rotationally symmetric, so wall collision is 2D: compare the
  // particle's distance from the Y axis with the glass radius at its height.
  #resolveWalls() {
    const d = this.data;
    const { restitution, friction } = PHYSICS;
    for (let o = 0, end = this.count * STRIDE; o < end; o += STRIDE) {
      const x = d[o], z = d[o + 2];
      const radial = Math.hypot(x, z);
      const maxR = this.profile.innerRadiusAt(d[o + 1]) - WALL_CLEARANCE - d[o + 3];

      if (maxR < 0.001) {
        if (radial > 0.001) {
          const scale = 0.001 / radial;
          d[o] *= scale; d[o + 2] *= scale;
          d[o + 4] = 0; d[o + 6] = 0;
        }
        continue;
      }
      if (radial > maxR && radial > 0.001) {
        const scale = maxR / radial;
        d[o] *= scale; d[o + 2] *= scale;
        const nx = x / radial, nz = z / radial;
        const radialVel = d[o + 4] * nx + d[o + 6] * nz;
        if (radialVel > 0) {
          d[o + 4] -= (1 + restitution) * radialVel * nx;
          d[o + 6] -= (1 + restitution) * radialVel * nz;
          const dotVN = d[o + 4] * nx + d[o + 6] * nz;
          const tx = d[o + 4] - dotVN * nx;
          const tz = d[o + 6] - dotVN * nz;
          d[o + 4] -= tx * friction;
          d[o + 6] -= tz * friction;
        }
      }
    }
  }

  #resolveFloorCeiling() {
    const d = this.data;
    const { restitution, friction } = PHYSICS;
    for (let o = 0, end = this.count * STRIDE; o < end; o += STRIDE) {
      const r = d[o + 3];
      if (d[o + 1] - r < CHAMBER_BOTTOM) {
        d[o + 1] = CHAMBER_BOTTOM + r;
        if (d[o + 5] < 0) d[o + 5] *= -restitution;
        d[o + 4] *= 1 - friction;
        d[o + 6] *= 1 - friction;
      }
      if (d[o + 1] + r > CHAMBER_TOP) {
        d[o + 1] = CHAMBER_TOP - r;
        if (d[o + 5] > 0) d[o + 5] *= -restitution;
        d[o + 4] *= 1 - friction;
        d[o + 6] *= 1 - friction;
      }
    }
  }

  // All pairs, O(N²). Fine for hundreds of particles; this loop is the reason
  // the other two modes exist.
  #resolveSpheres() {
    const d = this.data;
    const n = this.count;
    const { restitution } = PHYSICS;
    for (let i = 0; i < n; i++) {
      const a = i * STRIDE;
      for (let j = i + 1; j < n; j++) {
        const b = j * STRIDE;
        const dx = d[a] - d[b];
        const dy = d[a + 1] - d[b + 1];
        const dz = d[a + 2] - d[b + 2];
        const minDist = d[a + 3] + d[b + 3];
        const distSq = dx * dx + dy * dy + dz * dz;
        if (distSq >= minDist * minDist) continue;
        const dist = Math.sqrt(distSq);
        if (dist <= 0.0001) continue;

        const nx = dx / dist, ny = dy / dist, nz = dz / dist;
        const half = (minDist - dist) * 0.5;
        d[a] += nx * half; d[a + 1] += ny * half; d[a + 2] += nz * half;
        d[b] -= nx * half; d[b + 1] -= ny * half; d[b + 2] -= nz * half;

        const relVelNormal =
          (d[a + 4] - d[b + 4]) * nx + (d[a + 5] - d[b + 5]) * ny + (d[a + 6] - d[b + 6]) * nz;
        if (relVelNormal >= 0) continue;
        const impulse = -(1 + restitution) * relVelNormal * 0.5;
        d[a + 4] += nx * impulse; d[a + 5] += ny * impulse; d[a + 6] += nz * impulse;
        d[b + 4] -= nx * impulse; d[b + 5] -= ny * impulse; d[b + 6] -= nz * impulse;
      }
    }
  }

  /**
   * The container has just rotated by pi about X and is snapped back to 0.
   * The glass is symmetric top to bottom, so the picture doesn't change if the
   * particles are rotated instead: (x, y, z) -> (x, -y, -z).
   */
  snapAfterFlip() {
    const d = this.data;
    for (let o = 0, end = this.count * STRIDE; o < end; o += STRIDE) {
      d[o + 1] = -d[o + 1];
      d[o + 2] = -d[o + 2];
      d[o + 5] = -d[o + 5];
      d[o + 6] = -d[o + 6];
    }
    this.containerEulerX = 0;
  }
}

/** Summary of a particle array, used by ?test and by the unit tests. */
export function summarise(data, count) {
  let upper = 0, sleeping = 0, deepSleep = 0, totalSpeed = 0, maxSpeed = 0;
  let minY = Infinity, maxY = -Infinity, nonFinite = 0;
  for (let i = 0; i < count; i++) {
    const o = i * STRIDE;
    const y = data[o + 1];
    const speed = Math.hypot(data[o + 4], data[o + 5], data[o + 6]);
    if (!Number.isFinite(y) || !Number.isFinite(speed)) { nonFinite++; continue; }
    if (y > 0) upper++;
    if (data[o + 7] > 15) sleeping++;
    if (data[o + 7] > 30) deepSleep++;
    totalSpeed += speed;
    maxSpeed = Math.max(maxSpeed, speed);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  return {
    count, upper, lower: count - upper - nonFinite, sleeping, deepSleep, nonFinite,
    avgSpeed: count ? totalSpeed / count : 0, maxSpeed, minY, maxY,
    drainedPercent: count ? ((count - upper) / count) * 100 : 0,
  };
}
