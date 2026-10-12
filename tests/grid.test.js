import test from 'node:test';
import assert from 'node:assert/strict';
import { createProfile, neckRadiusForParticle } from '../js/models/hourglassProfile.js';
import { STRIDE, radiusForCount, packedParticles, mulberry32 } from '../js/models/packing.js';
import {
  computeGrid, cellCoords, hashCell, buildGridCPU, neighbourCandidates,
  nextPowerOfTwo, CELL_SCALE, buildSortedCPU, sortedCandidates, sampleSlot,
  BUCKET_WALK_LIMIT, SCAN_BLOCK, scanBlocks,
} from '../js/models/grid.js';

function pile(count, seed = 3) {
  const radius = radiusForCount(count);
  const profile = createProfile(neckRadiusForParticle(radius));
  const data = packedParticles(count, radius, profile, mulberry32(seed));
  // Jitter the lattice so it looks like a settled pile, not a crystal.
  const random = mulberry32(seed + 1);
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < 3; k++) data[i * STRIDE + k] += (random() - 0.5) * radius * 0.3;
  }
  return { radius, data };
}

test('nextPowerOfTwo', () => {
  assert.deepEqual([1, 2, 3, 64, 65, 400000].map(nextPowerOfTwo), [1, 2, 4, 64, 128, 524288]);
});

test('table size depends on the particle count, not the volume', () => {
  const small = computeGrid(0.02, 100);
  const large = computeGrid(0.0013, 200000);
  assert.equal(small.tableSize, 256);
  assert.equal(large.tableSize, 524288);
  assert.equal(large.tableMask, large.tableSize - 1);
  // 200k grains at half size would need about ten million cells as a dense array.
  // Heads plus one link per particle is under 3 MB.
  assert.ok(large.tableSize * 4 + 200000 * 4 < 3 * 1024 * 1024);
});

test('cells are wide enough for contact and for the 1.05x support test', () => {
  const grid = computeGrid(0.01, 1000);
  assert.ok(Math.abs(grid.cellSize - 0.02 * CELL_SCALE) < 1e-12);
  assert.ok(grid.cellSize >= 0.02 * 1.05 - 1e-12);
});

test('cell coordinates are non-negative everywhere inside the glass', () => {
  const grid = computeGrid(0.004, 50000);
  for (const [x, y, z] of [[-0.23, -0.5, -0.23], [0, 0, 0], [0.23, 0.5, 0.23]]) {
    assert.ok(cellCoords(grid, x, y, z).every((c) => c >= 0 && Number.isInteger(c)));
  }
});

test('every particle is in exactly one bucket list', () => {
  const count = 5000;
  const { radius, data } = pile(count);
  const grid = computeGrid(radius, count);
  const built = buildGridCPU(grid, data, count);
  const seen = new Uint8Array(count);
  let total = 0;
  for (let b = 0; b < grid.tableSize; b++) {
    for (let link = built.heads[b]; link !== 0; link = built.next[link - 1]) {
      seen[link - 1]++;
      total++;
    }
  }
  assert.equal(total, count);
  assert.ok(seen.every((n) => n === 1));
  for (let cx = 0; cx < 40; cx += 7) {
    const h = hashCell(grid, cx, cx * 3, cx + 11);
    assert.ok(h >= 0 && h <= grid.tableMask);
  }
});

test('a crowded cell loses nobody', () => {
  // 500 grains inside one grain's width: what the neck looks like mid-pour.
  const count = 500;
  const radius = 0.01;
  const random = mulberry32(5);
  const data = new Float32Array(count * STRIDE);
  for (let i = 0; i < count; i++) {
    data.set([(random() - 0.5) * radius, 0.2 + (random() - 0.5) * radius, (random() - 0.5) * radius, radius], i * STRIDE);
  }
  const grid = computeGrid(radius, count);
  const built = buildGridCPU(grid, data, count);
  const candidates = neighbourCandidates(grid, built, 0, 0.2, 0);
  assert.equal(new Set(candidates).size, count);
  assert.equal(candidates.length, count);
});

test('grid candidates include every true contact, with no duplicates', () => {
  const count = 3000;
  const { radius, data } = pile(count);
  const grid = computeGrid(radius, count);
  const built = buildGridCPU(grid, data, count);
  let contacts = 0;
  for (let i = 0; i < count; i += 7) {
    const a = i * STRIDE;
    const candidates = neighbourCandidates(grid, built, data[a], data[a + 1], data[a + 2]);
    assert.equal(new Set(candidates).size, candidates.length, 'duplicate candidate');
    const found = new Set(candidates);
    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      const b = j * STRIDE;
      const dist = Math.hypot(data[a] - data[b], data[a + 1] - data[b + 1], data[a + 2] - data[b + 2]);
      if (dist < radius * 2 * 1.05) {
        contacts++;
        assert.ok(found.has(j), `particle ${i} missed neighbour ${j} at distance ${dist}`);
      }
    }
  }
  assert.ok(contacts > 1000, `expected a packed pile, found ${contacts} contacts`);
});

test('candidates per particle stay small as the count grows', () => {
  const average = (count) => {
    const { radius, data } = pile(count);
    const grid = computeGrid(radius, count);
    const built = buildGridCPU(grid, data, count);
    let total = 0, samples = 0;
    for (let i = 0; i < count; i += 53) {
      const a = i * STRIDE;
      total += neighbourCandidates(grid, built, data[a], data[a + 1], data[a + 2]).length;
      samples++;
    }
    return total / samples;
  };
  const few = average(2000);
  const many = average(40000);
  // About 45: the true neighbours plus a few strangers that share a bucket.
  assert.ok(few < 60 && many < 60, `few=${few} many=${many}`);
  assert.ok(many < few * 1.6, 'cost per particle should not scale with N');
});

test('sorted grid: every grain gets one slot, and each bucket is one run', () => {
  const count = 3000;
  const { radius, data } = pile(count);
  const grid = computeGrid(radius, count);
  const sorted = buildSortedCPU(grid, data, count);
  assert.deepEqual([...sorted.ids].sort((a, b) => a - b), Array.from({ length: count }, (_, i) => i));
  let total = 0;
  for (let b = 0; b < grid.tableSize; b++) {
    assert.equal(sorted.starts[b], total, `bucket ${b} does not start where the last one ended`);
    for (let k = 0; k < sorted.counts[b]; k++) {
      const o = sorted.ids[total + k] * STRIDE;
      assert.equal(hashCell(grid, ...cellCoords(grid, data[o], data[o + 1], data[o + 2])), b);
    }
    total += sorted.counts[b];
  }
  assert.equal(total, count);
  // The scan works in blocks of 1,024 buckets; 200,000 grains means 512 blocks.
  assert.equal(scanBlocks(computeGrid(0.0013, 200000).tableSize), 512);
  assert.equal(scanBlocks(64), 1);
  assert.equal(SCAN_BLOCK % 4, 0);
});

test('sorted grid finds exactly the candidates the linked lists find', () => {
  const count = 3000;
  const { radius, data } = pile(count);
  const grid = computeGrid(radius, count);
  const lists = buildGridCPU(grid, data, count);
  const sorted = buildSortedCPU(grid, data, count);
  for (let i = 0; i < count; i += 11) {
    const a = i * STRIDE;
    const fromLists = neighbourCandidates(grid, lists, data[a], data[a + 1], data[a + 2]).sort((x, y) => x - y);
    const fromSorted = sortedCandidates(grid, sorted, data[a], data[a + 1], data[a + 2]).sort((x, y) => x - y);
    assert.deepEqual(fromSorted, fromLists);
  }
});

test('a crowded range is sampled evenly; a short one is visited whole', () => {
  for (const n of [1, 5, BUCKET_WALK_LIMIT, BUCKET_WALK_LIMIT + 1, 100, 938, 5000]) {
    for (const offset of [0, 7, 123456789]) {
      const visits = Math.min(n, BUCKET_WALK_LIMIT);
      const slots = Array.from({ length: visits }, (_, i) => sampleSlot(i, n, offset));
      assert.equal(new Set(slots).size, visits, `n=${n}: a slot visited twice`);
      assert.ok(slots.every((s) => s >= 0 && s < n));
      // Gaps between visited slots (round the range) are all n/visits, give or take one.
      const sortedSlots = [...slots].sort((a, b) => a - b);
      const gaps = sortedSlots.map((s, i) => ((sortedSlots[(i + 1) % visits] - s) + n) % n || n);
      assert.ok(Math.max(...gaps) - Math.min(...gaps) <= 1, `n=${n}: gaps ${Math.min(...gaps)}..${Math.max(...gaps)}`);
    }
  }
});
