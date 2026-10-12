// Spatial hash grid sizing for the GPU broadphase. Pure module.
//
// All-pairs collision asks "is particle i touching particle j?" for every j.
// But a sphere can only touch things within one diameter of it. So chop space
// into cubes at least one diameter wide, note which particles sit in which
// cube, and each particle only has to look in its own cube and the 26 around
// it. The cost per particle stops depending on N.
//
// A plain 3D array of cubes would be mostly empty air (the sand fills a small
// part of the glass), and with 200,000 particles at the smallest size it
// would not fit in one GPU buffer. So the cube coordinates are hashed into a
// table whose size depends on the particle count, not on the volume.
//
// Each bucket is a linked list of particle indices, so a bucket can hold any
// number of particles. That matters here: this simulation lets grains overlap
// heavily where the pile is squeezed (the neck above all), and a bucket with
// a fixed number of slots silently loses the grains that don't fit.
//
// Two different cubes can land in the same bucket. That only adds candidates
// which the distance test then rejects.
//
// The functions here are the reference implementation. The WGSL in
// physics/shaders.js does the same arithmetic on the GPU, and the tests check
// this version finds every true contact.

/** Cells are a little wider than a diameter so the 1.05x support test fits. */
export const CELL_SCALE = 1.05;
/** Buckets per particle, before rounding up to a power of two. */
export const TABLE_LOAD = 2;

/**
 * How far the GPU walks along one bucket's list before giving up on it.
 *
 * A resting pile puts at most about ten grains in a bucket. But where the
 * sand is squeezed through the neck this simulation piles hundreds or
 * thousands of grains into a few cells, and then every grain nearby walks
 * all of them: on an iPhone one physics step went from 2 ms to 300 ms for
 * about a second of every run. Measured with the walk limited, the sand
 * drains at the same rate and settles into the same pile, because in a cell
 * that crowded the forces are already meaningless. So the walk stops here.
 * Nothing is lost from the list (every grain still does its own walk); a
 * grain past the limit just isn't seen by its neighbours that step.
 *
 * It must stay well above a resting pile's bucket size, or ordinary piles
 * start losing neighbours, which is the bug the linked lists replaced.
 */
export const BUCKET_WALK_LIMIT = 32;

const ORIGIN = Object.freeze([-0.25, -0.52, -0.25]); // just outside the glass

export function nextPowerOfTwo(n) {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

export function computeGrid(particleRadius, count) {
  const tableSize = nextPowerOfTwo(Math.max(count * TABLE_LOAD, 64));
  return {
    cellSize: particleRadius * 2 * CELL_SCALE,
    origin: ORIGIN,
    tableSize,
    tableMask: tableSize - 1,
  };
}

/** Integer cube coordinates for a position (never negative inside the glass). */
export function cellCoords(grid, x, y, z) {
  const c = (v, o) => Math.max(Math.floor((v - o) / grid.cellSize), 0);
  return [c(x, grid.origin[0]), c(y, grid.origin[1]), c(z, grid.origin[2])];
}

/** The classic three-primes spatial hash, in 32-bit unsigned arithmetic. */
export function hashCell(grid, cx, cy, cz) {
  const h = Math.imul(cx, 73856093) ^ Math.imul(cy, 19349663) ^ Math.imul(cz, 83492791);
  return (h >>> 0) & grid.tableMask;
}

/**
 * CPU build of the structure the GPU builds with atomics.
 * heads[b] is the first particle in bucket b, next[i] the one after particle
 * i. Both store index + 1, so 0 means "end of list".
 */
export function buildGridCPU(grid, particles, count, stride = 8) {
  const heads = new Uint32Array(grid.tableSize);
  const next = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const o = i * stride;
    const bucket = hashCell(grid, ...cellCoords(grid, particles[o], particles[o + 1], particles[o + 2]));
    next[i] = heads[bucket];
    heads[bucket] = i + 1;
  }
  return { heads, next };
}

/** The distinct buckets of the 27 cubes around (x, y, z), in sweep order. */
function neighbourBuckets(grid, x, y, z) {
  const [cx, cy, cz] = cellCoords(grid, x, y, z);
  const buckets = [];
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0) continue;
        const bucket = hashCell(grid, nx, ny, nz);
        if (!buckets.includes(bucket)) buckets.push(bucket);
      }
    }
  }
  return buckets;
}

/**
 * Indices of every particle in the buckets of the 27 cubes around (x, y, z).
 * No duplicates. May include particles from unrelated cubes that share a
 * bucket; callers reject those by distance.
 */
export function neighbourCandidates(grid, built, x, y, z) {
  const out = [];
  for (const bucket of neighbourBuckets(grid, x, y, z)) {
    for (let link = built.heads[bucket]; link !== 0; link = built.next[link - 1]) {
      out.push(link - 1);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The sorted grid: the same cells and the same hash, stored a different way.
//
// Instead of a linked list per bucket, every particle is copied into one long
// array in bucket order, so each bucket's particles sit side by side and a
// bucket is just a range: start and count. This is a counting sort:
//
//   1. count:   each particle bumps its bucket's counter, and the value it
//               saw is its rank within the bucket (0, 1, 2 ...)
//   2. scan:    a running total over the counters gives each bucket's start
//   3. scatter: particle i goes to slot start[bucket] + rank[i]
//
// Why bother, when the lists work?
//   - Walking a list is a chain of dependent loads: the GPU can't fetch the
//     next grain until the last one arrives. A range can be fetched in any
//     order, all at once.
//   - Neighbours are next to each other in memory, so they share cache lines.
//   - The physics threads run in sorted order too, so the 64 threads of a
//     workgroup handle 64 grains in the same few cells and read the same
//     ranges.
//   - A crowded bucket can be sampled evenly (see sampleSlot) rather than by
//     taking whichever 32 grains happen to be first in the list.
//
// The particle buffers themselves are never reordered. The sorted array is a
// copy made fresh every substep, with each slot remembering which particle it
// came from, and results are written back to the particle's own slot. So a
// grain keeps its index for good: its colour, its place in the deep-sleep
// stagger, and the tests that follow it by index all carry on working.
// ---------------------------------------------------------------------------

/**
 * Buckets per block in the scan. A workgroup of 256 threads scans one
 * block, each thread taking four neighbouring buckets as one 16-byte load.
 */
export const SCAN_BLOCK = 1024;

/** Blocks the scan splits the bucket table into. */
export function scanBlocks(tableSize) {
  return Math.ceil(tableSize / SCAN_BLOCK);
}

/**
 * CPU version of the three passes. The GPU computes ranks with atomics, so
 * its order within a bucket can differ from this one; the ranges can't.
 * Returns counts and starts per bucket, and ids[slot] = particle index.
 */
export function buildSortedCPU(grid, particles, count, stride = 8) {
  const counts = new Uint32Array(grid.tableSize);
  const rank = new Uint32Array(count);
  const bucketOf = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const o = i * stride;
    const bucket = hashCell(grid, ...cellCoords(grid, particles[o], particles[o + 1], particles[o + 2]));
    bucketOf[i] = bucket;
    rank[i] = counts[bucket]++;
  }
  const starts = new Uint32Array(grid.tableSize);
  let total = 0;
  for (let b = 0; b < grid.tableSize; b++) {
    starts[b] = total;
    total += counts[b];
  }
  const ids = new Uint32Array(count);
  for (let i = 0; i < count; i++) ids[starts[bucketOf[i]] + rank[i]] = i;
  return { counts, starts, ids };
}

/**
 * Which slot of an n-grain bucket to look at on visit i, starting from a
 * per-grain random offset. Visits min(n, BUCKET_WALK_LIMIT) slots, all
 * different. When the bucket fits under the limit that is every slot, in a
 * rotated order; when it doesn't, the visits are spread evenly across the
 * whole range instead of bunching at one end. The WGSL does the same sum.
 */
export function sampleSlot(i, n, offset) {
  const visits = Math.min(n, BUCKET_WALK_LIMIT);
  return (offset % n + Math.floor((i * n) / visits)) % n;
}

/** Like neighbourCandidates, from the sorted layout, with no limit. */
export function sortedCandidates(grid, sorted, x, y, z) {
  const out = [];
  for (const bucket of neighbourBuckets(grid, x, y, z)) {
    const start = sorted.starts[bucket];
    for (let k = 0; k < sorted.counts[bucket]; k++) out.push(sorted.ids[start + k]);
  }
  return out;
}
