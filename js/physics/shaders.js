// WGSL compute shaders: a port of ParticlePhysics.metal.
//
// WGSL is WebGPU's shading language. A compute shader is a function the GPU
// runs many times at once, each run with a different id; here one run handles
// one particle. The source is plain text handed to device.createShaderModule,
// which is why it lives in JavaScript strings.
//
// Metal                                WGSL
// -----------------------------------  ------------------------------------------
// kernel void f(...)                   @compute @workgroup_size(64) fn f(...)
// [[thread_position_in_grid]]          @builtin(global_invocation_id)
// device const T* p [[buffer(1)]]      @group(0) @binding(1) var<storage, read> p: array<T>;
// device T* p       [[buffer(2)]]      @group(0) @binding(2) var<storage, read_write> p: array<T>;
// constant U& u     [[buffer(0)]]      @group(0) @binding(0) var<uniform> u: U;
// float3 / float4 / uint               vec3f / vec4f / u32
//
// The physics kernel is generated in two flavours that differ only in how a
// particle finds its neighbours: 'brute' tests every other particle (the
// app's O(N²) loop) and 'grid' looks in a spatial hash.

import { PROFILE_TABLE_SIZE } from '../models/hourglassProfile.js';

export const WORKGROUP_SIZE = 64;

/** Byte size of the Uniforms struct below. */
export const UNIFORM_BYTES = 64;

const COMMON = /* wgsl */ `
// 32 bytes, and the same layout as the CPU simulation and the vertex buffer.
struct Particle {
  posRadius: vec4f,   // xyz = position, w = radius
  velSleep: vec4f,    // xyz = velocity, w = sleep counter
}

struct Uniforms {
  gravity: f32,
  restitution: f32,
  friction: f32,
  subDt: f32,
  containerEulerX: f32,
  chamberBottom: f32,
  chamberTop: f32,
  neckDamping: f32,
  neckHalfHeight: f32,
  cellSize: f32,
  particleCount: u32,
  tableMask: u32,
  gridOrigin: vec4f,   // xyz = grid corner, w = frame number (for sweepOrder)
}

@group(0) @binding(0) var<uniform> u: Uniforms;

fn cellOf(p: vec3f) -> vec3i {
  return max(vec3i(floor((p - u.gridOrigin.xyz) / u.cellSize)), vec3i(0));
}

// xyz: +1 or -1 per axis. w: which of three axis rotations to use.
fn sweepOrder(tid: u32) -> vec4i {
  var h = (tid * 2654435761u) ^ (u32(u.gridOrigin.w) * 2246822519u);
  h ^= h >> 15u;
  h *= 739982445u;
  h ^= h >> 12u;
  return vec4i(
    1 - 2 * i32(h & 1u),
    1 - 2 * i32((h >> 1u) & 1u),
    1 - 2 * i32((h >> 2u) & 1u),
    i32((h >> 3u) % 3u),
  );
}

fn hashCell(c: vec3i) -> u32 {
  let h = (u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u);
  return h & u.tableMask;
}
`;

// ---------------------------------------------------------------------------
// Grid build: one thread per particle adds itself to its bucket.
//
// Each bucket is a linked list. bucketHeads[b] holds the first particle in
// bucket b, and nextInBucket[i] holds the particle after i. Indices are stored
// plus one, so that 0 means "end of list" and clearing the heads to zero
// empties every bucket.
//
// Thousands of threads may pick the same bucket at the same instant, so the
// head is atomic. atomicExchange swaps my index in and hands back whoever was
// there before, who becomes my "next". No thread can lose that race: every
// particle ends up in exactly one list, however crowded the cell.
// ---------------------------------------------------------------------------
export const GRID_BUILD_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read_write> bucketHeads: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> nextInBucket: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn buildGrid(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= u.particleCount) { return; }
  let bucket = hashCell(cellOf(particlesIn[tid].posRadius.xyz));
  nextInBucket[tid] = atomicExchange(&bucketHeads[bucket], tid + 1u);
}
`;

// ---------------------------------------------------------------------------
// Neighbour loops. `body` is WGSL that can use `j` (the other particle's
// index) and `other` (its data), and can set `stop = true` to finish early.
// ---------------------------------------------------------------------------
function forEachNeighbour(broadphase, centre, body) {
  if (broadphase === 'brute') {
    return /* wgsl */ `{
      var stop = false;
      for (var j = 0u; j < u.particleCount && !stop; j++) {
        if (j == tid) { continue; }
        let other = particlesIn[j];
        ${body}
      }
    }`;
  }
  return /* wgsl */ `{
      var stop = false;
      let home = cellOf(${centre});
      // Two of the 27 cubes can hash to the same bucket. Remember the buckets
      // already walked so no particle is visited twice.
      // Each grain sweeps the 27 cubes in its own scrambled order, different
      // every frame. The collision loop moves a grain as it goes, so the
      // order in which neighbours are met nudges the result. All-pairs meets
      // them in index order, which is spatially arbitrary. A grid sweep is
      // not: when every grain met its -x neighbours first, every grain got
      // the same nudge and the pile crept sideways. So the sweep direction
      // on each axis, and which axis is outermost, come from a hash of the
      // grain's index and the frame number, and the nudges cancel.
      let order = sweepOrder(tid);
      var seen: array<u32, 27>;
      var seenCount = 0u;
      for (var dz = -1; dz <= 1 && !stop; dz++) {
        for (var dy = -1; dy <= 1 && !stop; dy++) {
          for (var dx = -1; dx <= 1 && !stop; dx++) {
            var step = vec3i(dx, dy, dz) * order.xyz;
            if (order.w == 1) { step = step.yzx; }
            if (order.w == 2) { step = step.zxy; }
            let cell = home + step;
            if (any(cell < vec3i(0))) { continue; }
            let bucket = hashCell(cell);
            var repeat = false;
            for (var k = 0u; k < seenCount; k++) {
              if (seen[k] == bucket) { repeat = true; }
            }
            if (repeat) { continue; }
            seen[seenCount] = bucket;
            seenCount++;

            // Walk the bucket's list. A bucket may also hold particles from
            // some far-off cube with the same hash; the distance tests in the
            // body reject those. The guard only matters if memory is corrupt.
            var link = bucketHeads[bucket];
            var guard = 0u;
            while (link != 0u && !stop && guard < u.particleCount) {
              let j = link - 1u;
              link = nextInBucket[j];
              guard++;
              if (j == tid) { continue; }
              let other = particlesIn[j];
              ${body}
            }
          }
        }
      }
    }`;
}

/**
 * The physics kernel. Steps 1 to 9 follow ParticlePhysics.metal line for line;
 * the comments there explain the tuning. The short version of why it looks
 * the way it does:
 *
 * Every thread reads the same snapshot (particlesIn) and writes only its own
 * particle (particlesOut). Nobody sees anybody else's correction until the
 * next substep. Corrections therefore overshoot and oscillate where the CPU
 * version converges, and impulses cancel gravity inside a packed pile. The
 * kernel deals with that by treating the two chambers differently and by
 * putting resting particles to sleep.
 */
export function physicsWGSL(broadphase) {
  const grid = broadphase === 'grid';
  const neighbours = (centre, body) => forEachNeighbour(broadphase, centre, body);

  return /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(3) var<storage, read> profileTable: array<f32, ${PROFILE_TABLE_SIZE}>;
${grid ? /* wgsl */ `
@group(0) @binding(4) var<storage, read> bucketHeads: array<u32>;
@group(0) @binding(5) var<storage, read> nextInBucket: array<u32>;` : ''}

// Inner glass radius at height y: O(1) lookup in a 256-entry table.
fn glassRadius(y: f32) -> f32 {
  let yNorm = clamp((y + 0.5) * ${PROFILE_TABLE_SIZE - 1}.0, 0.0, ${PROFILE_TABLE_SIZE - 1}.0);
  let i0 = u32(yNorm);
  let i1 = min(i0 + 1u, ${PROFILE_TABLE_SIZE - 1}u);
  return mix(profileTable[i0], profileTable[i1], yNorm - f32(i0));
}

// Deep sleepers cycle 31..60 so that (counter % 30) visits every value once
// per 30 steps, which is what staggers their support checks.
fn deeperSleep(counter: f32) -> f32 {
  let next = counter + 1.0;
  if (next > 60.0) { return 31.0; }
  return next;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn physicsStep(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= u.particleCount) { return; }

  let me = particlesIn[tid];
  var pos = me.posRadius.xyz;
  let radius = me.posRadius.w;
  var vel = me.velSleep.xyz;
  var sleepCounter = me.velSleep.w;

  // ── Sleep ────────────────────────────────────────────────────────────
  // counter 16..30  light sleep: frozen, but checks every step whether an
  //                 awake particle is hitting it
  // counter  > 30   deep sleep: frozen, and checks only one step in 30
  //                 whether it still has something to rest on
  if (sleepCounter > 30.0) {
    let myTurn = (u32(sleepCounter) % 30u) == (tid % 30u);
    var supported = !myTurn;
    if (myTurn) {
      supported = (pos.y - radius) < (u.chamberBottom + radius * 0.1);
      if (!supported) ${neighbours('pos', /* wgsl */ `
        if (distance(pos, other.posRadius.xyz) < (radius + other.posRadius.w) * 1.05) {
          supported = true;
          stop = true;
        }`)}
    }
    if (supported) {
      particlesOut[tid] = Particle(me.posRadius, vec4f(0.0, 0.0, 0.0, deeperSleep(sleepCounter)));
      return;
    }
    sleepCounter = 0.0;   // nothing underneath any more: wake and fall
  }

  if (sleepCounter > 15.0) {
    var woken = false;
    ${neighbours('pos', /* wgsl */ `
        let delta = pos - other.posRadius.xyz;
        let dist = length(delta);
        if (dist < radius + other.posRadius.w && dist > 0.0001 && other.velSleep.w <= 15.0) {
          if (dot(vel - other.velSleep.xyz, delta / dist) < -0.05) {
            woken = true;
            stop = true;
          }
        }`)}
    if (!woken) {
      particlesOut[tid] = Particle(me.posRadius, vec4f(0.0, 0.0, 0.0, min(sleepCounter + 1.0, 60.0)));
      return;
    }
    sleepCounter = 0.0;
  }

  // ── Awake ────────────────────────────────────────────────────────────
  let subDt = u.subDt;
  let theta = u.containerEulerX;

  // 1. Gravity, rotated into the container's frame during the flip.
  vel += vec3f(0.0, -u.gravity * cos(theta), u.gravity * sin(theta)) * subDt;

  // 2. Move.
  pos += vel * subDt;

  // 3. Glass wall. The glass is symmetric about Y, so this is a 2D test.
  let radial = length(pos.xz);
  let maxR = glassRadius(pos.y) - radius;
  if (maxR < 0.001) {
    // Narrower than the particle (the neck pinch): steer onto the axis.
    if (radial > 0.001) {
      let scale = 0.001 / radial;
      pos.x *= scale;
      pos.z *= scale;
      vel.x = 0.0;
      vel.z = 0.0;
    }
  } else if (radial > maxR && radial > 0.001) {
    let scale = maxR / radial;
    pos.x *= scale;
    pos.z *= scale;
    let n = pos.xz / max(maxR, 0.001);
    let radialVel = dot(vel.xz, n);
    if (radialVel > 0.0) {
      var v = vel.xz - (1.0 + u.restitution) * radialVel * n;
      let tangent = v - dot(v, n) * n;
      v -= tangent * u.friction;
      vel.x = v.x;
      vel.z = v.y;
    }
  }

  // 4. Floor and ceiling, with resting contact: a bounce smaller than what
  //    gravity adds in two steps is not a bounce, so zero it.
  if (pos.y - radius < u.chamberBottom) {
    pos.y = u.chamberBottom + radius;
    if (vel.y < 0.0) { vel.y *= -u.restitution; }
    if (abs(vel.y) < u.gravity * subDt * 2.0) { vel.y = 0.0; }
    vel.x *= 1.0 - u.friction;
    vel.z *= 1.0 - u.friction;
  }
  if (pos.y + radius > u.chamberTop) {
    pos.y = u.chamberTop - radius;
    if (vel.y > 0.0) { vel.y *= -u.restitution; }
    if (abs(vel.y) < u.gravity * subDt * 2.0) { vel.y = 0.0; }
    vel.x *= 1.0 - u.friction;
    vel.z *= 1.0 - u.friction;
  }

  // 5. Sphere against sphere.
  //    Upper chamber: push apart only. Impulses there cancel gravity and jam
  //    the pile, so gravity is left to drain it.
  //    Lower chamber: push apart and exchange velocity, so a pile can carry
  //    its own weight and come to rest.
  let inUpperChamber = pos.y > 0.0;
  var contactCount = 0u;
  ${neighbours('pos', /* wgsl */ `
        let delta = pos - other.posRadius.xyz;
        let dist = length(delta);
        let minDist = radius + other.posRadius.w;
        if (dist < minDist && dist > 0.0001) {
          contactCount++;
          let normal = delta / dist;
          pos += normal * ((minDist - dist) * 0.25);
          if (!inUpperChamber) {
            let relVelNormal = dot(vel - other.velSleep.xyz, normal);
            if (relVelNormal < 0.0) {
              vel += normal * (-(1.0 + u.restitution) * relVelNormal * 0.25);
            }
          }
        }`)}

  // 6. Neck friction: the flow-rate control.
  if (u.neckDamping > 0.0 && u.neckHalfHeight > 0.0) {
    let neckDist = abs(pos.y);
    if (neckDist < u.neckHalfHeight) {
      let neckFactor = 1.0 - neckDist / u.neckHalfHeight;
      vel *= max(0.0, 1.0 - neckFactor * u.neckDamping * subDt);
    }
  }

  // 7. Damping: gentle while flowing; in the lower chamber, hard when slow.
  let flowDamp = pow(0.92, subDt);
  if (inUpperChamber) {
    vel *= flowDamp;
  } else {
    let settleDamp = pow(0.05, subDt);
    let t = min(length(vel) / 0.15, 1.0);
    vel *= mix(settleDamp, flowDamp, t);
  }

  // 8. Snap tiny velocities to zero, but only for a particle that is resting
  //    on something. One in free fall must never be frozen in mid-air.
  let hasSupport = contactCount > 0u;
  if (!inUpperChamber && hasSupport && length(vel) < 0.01) {
    vel = vec3f(0.0);
  }

  // 9. Sleep counter. The upper chamber never sleeps: it has to drain.
  if (pos.y >= 0.0) {
    sleepCounter = 0.0;
  } else if (hasSupport && length(vel) < 0.08) {
    sleepCounter = min(sleepCounter + 1.0, 60.0);
  } else {
    sleepCounter = 0.0;
  }

  particlesOut[tid] = Particle(vec4f(pos, radius), vec4f(vel, sleepCounter));
}
`;
}

// ---------------------------------------------------------------------------
// After the flip animation the container snaps from pi back to 0. Rotating
// by pi about X maps (x, y, z) to (x, -y, -z). Everyone wakes up.
// ---------------------------------------------------------------------------
export const SNAP_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn snapAfterFlip(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= u.particleCount) { return; }
  let p = particles[tid];
  particles[tid] = Particle(
    vec4f(p.posRadius.x, -p.posRadius.y, -p.posRadius.z, p.posRadius.w),
    vec4f(p.velSleep.x, -p.velSleep.y, -p.velSleep.z, 0.0),
  );
}
`;
