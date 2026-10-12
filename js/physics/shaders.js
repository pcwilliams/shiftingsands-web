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
// The physics kernel is generated in three flavours that differ only in how a
// particle finds its neighbours: 'brute' tests every other particle (the
// app's O(N²) loop), 'grid' walks linked lists in a spatial hash, and
// 'sorted' reads ranges of a copy of the particles sorted by hash bucket.

import { BUCKET_WALK_LIMIT, SCAN_BLOCK } from '../models/grid.js';
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

// Where to start in a bucket's range (sorted grid): different for every
// grain, bucket and frame.
fn sampleOffset(tid: u32, bucket: u32) -> u32 {
  var h = (tid * 2654435761u) ^ (bucket * 2246822519u) ^ (u32(u.gridOrigin.w) * 3266489917u);
  h ^= h >> 15u;
  h *= 2246822519u;
  h ^= h >> 13u;
  return h;
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
// Sorted grid: a counting sort by bucket, in four dispatches per substep.
// models/grid.js explains the idea and has the CPU version.
//
// bucketStarts holds two things back to back: tableSize per-bucket starts,
// each counted from the beginning of its block of 1,024 buckets, and then
// one total per block, which the second scan pass turns into the block's
// own start. A bucket's real start is the sum of the two.
// ---------------------------------------------------------------------------

const SCAN_THREADS = SCAN_BLOCK / 4;

// 1. Count. atomicAdd returns the counter's value before my +1, which is my
//    rank among the grains in this bucket.
export const SORT_COUNT_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read_write> bucketCounts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> rankInBucket: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn countBuckets(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= u.particleCount) { return; }
  let bucket = hashCell(cellOf(particlesIn[tid].posRadius.xyz));
  rankInBucket[tid] = atomicAdd(&bucketCounts[bucket], 1u);
}
`;

// An inclusive prefix sum across one workgroup, in workgroup memory: eight
// rounds (log2 of 256), in each of which a thread adds the value 1, 2, 4 ...
// places to its left. Every thread must reach every barrier, so nothing
// returns early.
const WORKGROUP_SCAN = /* wgsl */ `
var<workgroup> tile: array<u32, ${SCAN_THREADS}>;

fn inclusiveScan(lane: u32, value: u32) -> u32 {
  tile[lane] = value;
  workgroupBarrier();
  for (var offset = 1u; offset < ${SCAN_THREADS}u; offset *= 2u) {
    var add = 0u;
    if (lane >= offset) { add = tile[lane - offset]; }
    workgroupBarrier();
    tile[lane] += add;
    workgroupBarrier();
  }
  return tile[lane];
}
`;

// 2a. Scan each block of 1,024 buckets on its own, and note the block's
//     total. Each thread loads four counters at once (the table size is a
//     power of two, so it divides by four), totals them, joins the
//     workgroup scan with that total, and then writes its four starts.
export const SCAN_BLOCKS_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read> bucketCounts: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> bucketStarts: array<u32>;
${WORKGROUP_SCAN}

@compute @workgroup_size(${SCAN_THREADS})
fn scanBlocks(@builtin(global_invocation_id) gid: vec3u,
              @builtin(local_invocation_id) lid: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
  let tableSize = u.tableMask + 1u;
  var four = vec4u(0u);
  if (gid.x < tableSize / 4u) { four = bucketCounts[gid.x]; }
  let total = four.x + four.y + four.z + four.w;
  let before = inclusiveScan(lid.x, total) - total;
  if (gid.x < tableSize / 4u) {
    let b = gid.x * 4u;
    bucketStarts[b] = before;
    bucketStarts[b + 1u] = before + four.x;
    bucketStarts[b + 2u] = before + four.x + four.y;
    bucketStarts[b + 3u] = before + four.x + four.y + four.z;
  }
  if (lid.x == ${SCAN_THREADS - 1}u) { bucketStarts[tableSize + wid.x] = before + total; }
}
`;

// 2b. One workgroup turns the block totals into block starts. There are up
//     to 512 blocks (200,000 grains), so each thread first adds up a short
//     run of them on its own.
export const SCAN_SUMS_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read_write> bucketStarts: array<u32>;
${WORKGROUP_SCAN}

@compute @workgroup_size(${SCAN_THREADS})
fn scanSums(@builtin(local_invocation_id) lid: vec3u) {
  let tableSize = u.tableMask + 1u;
  let blocks = (tableSize + ${SCAN_BLOCK - 1}u) / ${SCAN_BLOCK}u;
  let perThread = (blocks + ${SCAN_THREADS - 1}u) / ${SCAN_THREADS}u;
  let first = lid.x * perThread;
  var total = 0u;
  for (var k = 0u; k < perThread; k++) {
    if (first + k < blocks) { total += bucketStarts[tableSize + first + k]; }
  }
  var running = inclusiveScan(lid.x, total) - total;
  for (var k = 0u; k < perThread; k++) {
    if (first + k < blocks) {
      let blockTotal = bucketStarts[tableSize + first + k];
      bucketStarts[tableSize + first + k] = running;
      running += blockTotal;
    }
  }
}
`;

// 3. Scatter: copy each grain to its slot and remember where it came from.
//    No atomics: the rank from step 1 makes every slot unique.
export const SORT_SCATTER_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read> bucketStarts: array<u32>;
@group(0) @binding(3) var<storage, read> rankInBucket: array<u32>;
@group(0) @binding(4) var<storage, read_write> sortedParticles: array<Particle>;
@group(0) @binding(5) var<storage, read_write> sortedIds: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= u.particleCount) { return; }
  let p = particlesIn[tid];
  let bucket = hashCell(cellOf(p.posRadius.xyz));
  let slot = bucketStarts[bucket] + bucketStarts[u.tableMask + 1u + bucket / ${SCAN_BLOCK}u] + rankInBucket[tid];
  sortedParticles[slot] = p;
  sortedIds[slot] = tid;
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
${broadphase === 'sorted' ? sortedBucket(body) : listBucket(body)}
          }
        }
      }
    }`;
}

// Grid: walk the bucket's list, but only so far: see BUCKET_WALK_LIMIT in
// models/grid.js for why. A bucket may also hold particles from some far-off
// cube with the same hash; the distance tests in the body reject those.
function listBucket(body) {
  return /* wgsl */ `
            var link = bucketHeads[bucket];
            var walked = 0u;
            while (link != 0u && !stop && walked < ${BUCKET_WALK_LIMIT}u) {
              let j = link - 1u;
              link = nextInBucket[j];
              walked++;
              if (j == tid) { continue; }
              let other = particlesIn[j];
              ${body}
            }`;
}

// Sorted: the bucket is a range of the sorted copy. Visit up to
// BUCKET_WALK_LIMIT slots of it, starting at a random place and, when the
// range is longer than that, spread evenly along it (sampleSlot in
// models/grid.js is the same sum).
function sortedBucket(body) {
  return /* wgsl */ `
            let n = bucketCounts[bucket];
            if (n == 0u) { continue; }
            let first = bucketStarts[bucket] + bucketStarts[u.tableMask + 1u + bucket / ${SCAN_BLOCK}u];
            let visits = min(n, ${BUCKET_WALK_LIMIT}u);
            // A start in 0..n-1 without an integer division, which GPUs do
            // in software: scale the top 24 bits of the hash by n.
            let offset = min(u32(f32(sampleOffset(tid, bucket) >> 8u) * (1.0 / 16777216.0) * f32(n)), n - 1u);
            var at = offset;
            for (var v = 0u; v < visits && !stop; v++) {
              let slot = first + at;
              // Next slot. Usually the bucket fits under the limit and this
              // is just the next one round; only a crowded bucket pays for
              // the divisions that spread the visits along it.
              if (n > ${BUCKET_WALK_LIMIT}u) {
                at = (offset + ((v + 1u) * n) / ${BUCKET_WALK_LIMIT}u) % n;
              } else {
                at++;
                if (at == n) { at = 0u; }
              }
              let j = sortedIds[slot];
              if (j == tid) { continue; }
              let other = sortedParticles[slot];
              ${body}
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
  const sorted = broadphase === 'sorted';
  const neighbours = (centre, body) => forEachNeighbour(broadphase, centre, body);

  return /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(3) var<storage, read> profileTable: array<f32, ${PROFILE_TABLE_SIZE}>;
${grid ? /* wgsl */ `
@group(0) @binding(4) var<storage, read> bucketHeads: array<u32>;
@group(0) @binding(5) var<storage, read> nextInBucket: array<u32>;` : ''}
${sorted ? /* wgsl */ `
@group(0) @binding(4) var<storage, read> bucketCounts: array<u32>;
@group(0) @binding(5) var<storage, read> bucketStarts: array<u32>;
@group(0) @binding(6) var<storage, read> sortedParticles: array<Particle>;
@group(0) @binding(7) var<storage, read> sortedIds: array<u32>;` : ''}

// Inner glass radius at height y: O(1) lookup in a 256-entry table.
fn glassRadius(y: f32) -> f32 {
  let yNorm = clamp((y + 0.5) * ${PROFILE_TABLE_SIZE - 1}.0, 0.0, ${PROFILE_TABLE_SIZE - 1}.0);
  let i0 = u32(yNorm);
  let i1 = min(i0 + 1u, ${PROFILE_TABLE_SIZE - 1}u);
  return mix(profileTable[i0], profileTable[i1], yNorm - f32(i0));
}

// What counts as something to rest on. Touching another grain is not enough:
// a clump of grains can touch each other in mid-air. Support has to come from
// underneath: the floor, glass that slopes inwards below the grain, or a
// grain whose centre is clearly lower.
fn onFloor(pos: vec3f, radius: f32) -> bool {
  return (pos.y - radius) < (u.chamberBottom + radius * 0.1);
}

fn onGlassSlope(pos: vec3f, radius: f32) -> bool {
  let here = glassRadius(pos.y);
  let touching = length(pos.xz) > here - radius * 1.1;
  // Only a real ledge counts: the glass must come in by more than half a
  // radius over one radius of height, which is true of the rounded rim just
  // above the floor and nowhere else. The side of the bulb leans in a little
  // below its widest point, but a grain cannot rest on that.
  return touching && glassRadius(pos.y - radius) < here - radius * 0.5;
}

// delta = my centre minus the other grain's; dist = its length.
fn holdsMeUp(delta: vec3f, dist: f32) -> bool {
  return delta.y > dist * 0.25;
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
  if (gid.x >= u.particleCount) { return; }
${sorted ? /* wgsl */ `
  // Threads take grains in sorted order, so neighbouring threads handle
  // neighbouring grains and read the same few ranges. Everything else,
  // including where the result goes, uses the grain's own index.
  let tid = sortedIds[gid.x];` : /* wgsl */ `
  let tid = gid.x;`}

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
      supported = onFloor(pos, radius) || onGlassSlope(pos, radius);
      if (!supported) ${neighbours('pos', /* wgsl */ `
        let delta = pos - other.posRadius.xyz;
        let dist = length(delta);
        if (dist < (radius + other.posRadius.w) * 1.05 && holdsMeUp(delta, dist)) {
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
  var supportCount = 0u;
  ${neighbours('pos', /* wgsl */ `
        let delta = pos - other.posRadius.xyz;
        let dist = length(delta);
        let minDist = radius + other.posRadius.w;
        if (dist < minDist && dist > 0.0001) {
          contactCount++;
          if (holdsMeUp(delta, dist)) { supportCount++; }
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
  let hasSupport = supportCount > 0u || onFloor(pos, radius) || onGlassSlope(pos, radius);
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
