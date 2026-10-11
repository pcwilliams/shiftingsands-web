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

/**
 * Indices of every particle in the buckets of the 27 cubes around (x, y, z).
 * No duplicates. May include particles from unrelated cubes that share a
 * bucket; callers reject those by distance.
 */
export function neighbourCandidates(grid, built, x, y, z) {
  const [cx, cy, cz] = cellCoords(grid, x, y, z);
  const out = [];
  const seen = [];
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0) continue;
        const bucket = hashCell(grid, nx, ny, nz);
        if (seen.includes(bucket)) continue;
        seen.push(bucket);
        for (let link = built.heads[bucket]; link !== 0; link = built.next[link - 1]) {
          out.push(link - 1);
        }
      }
    }
  }
  return out;
}
