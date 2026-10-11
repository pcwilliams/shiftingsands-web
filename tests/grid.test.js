import test from 'node:test';
import assert from 'node:assert/strict';
import { createProfile, neckRadiusForParticle } from '../js/models/hourglassProfile.js';
import { STRIDE, radiusForCount, packedParticles, mulberry32 } from '../js/models/packing.js';
import {
  computeGrid, cellCoords, hashCell, buildGridCPU, neighbourCandidates,
  nextPowerOfTwo, BUCKET_CAPACITY, CELL_SCALE,
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
  assert.ok(large.tableSize * BUCKET_CAPACITY * 4 < 32 * 1024 * 1024);
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

test('hash stays inside the table and spreads a pile across buckets', () => {
  const count = 5000;
  const { radius, data } = pile(count);
  const grid = computeGrid(radius, count);
  const built = buildGridCPU(grid, data, count);
  assert.equal(built.dropped, 0, 'no bucket should overflow');
  assert.ok(Math.max(...built.counts) <= BUCKET_CAPACITY);
  for (let cx = 0; cx < 40; cx += 7) {
    const h = hashCell(grid, cx, cx * 3, cx + 11);
    assert.ok(h >= 0 && h <= grid.tableMask);
  }
});

test('grid candidates include every true contact, with no duplicates', () => {
  const count = 3000;
  const { radius, data } = pile(count);
  const grid = computeGrid(radius, count);
  const built = buildGridCPU(grid, data, count);
  let contacts = 0;
  for (let i = 0; i < count; i += 7) {
    const a = i * STRIDE;
    const candidates = neighbourCandidates(grid, built, data, data[a], data[a + 1], data[a + 2]);
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
      total += neighbourCandidates(grid, built, data, data[a], data[a + 1], data[a + 2]).length;
      samples++;
    }
    return total / samples;
  };
  const few = average(2000);
  const many = average(40000);
  assert.ok(few < 40 && many < 40, `few=${few} many=${many}`);
  assert.ok(many < few * 1.6, 'cost per particle should not scale with N');
});
