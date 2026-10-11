import test from 'node:test';
import assert from 'node:assert/strict';
import { createProfile, neckRadiusForParticle } from '../js/models/hourglassProfile.js';
import {
  STRIDE, radiusForCount, packedParticles, mulberry32, hsbToRgb, particleColors,
} from '../js/models/packing.js';

function setup(count, size = 1, seed = 1) {
  const radius = radiusForCount(count, size);
  const profile = createProfile(neckRadiusForParticle(radius));
  const data = packedParticles(count, radius, profile, mulberry32(seed));
  const ys = [];
  for (let i = 0; i < count; i++) ys.push(data[i * STRIDE + 1]);
  return { radius, profile, data, ys };
}

test('radiusForCount matches the app\'s reference values', () => {
  // 250 is the calibration point, so the correction factor is exactly 1 there.
  assert.ok(Math.abs(radiusForCount(250) - 0.030 * Math.cbrt(100 / 250)) < 1e-12);
  assert.ok(radiusForCount(100) < 0.030, 'correction shrinks radius below the reference count');
  assert.ok(radiusForCount(10000) > 0.030 * Math.cbrt(100 / 10000), 'and grows it above');
  assert.ok(Math.abs(radiusForCount(5000, 1.3) - radiusForCount(5000) * 1.3) < 1e-12);
});

test('radius falls monotonically as count rises', () => {
  let prev = Infinity;
  for (const n of [50, 250, 1000, 5000, 20000, 50000, 200000]) {
    const r = radiusForCount(n);
    assert.ok(r < prev, `n=${n}`);
    prev = r;
  }
});

test('spawn packing stays in the lower chamber (10k at 1.3x)', () => {
  const count = 10000;
  const { radius, data, ys } = setup(count, 1.3);
  assert.equal(data.length, count * STRIDE);
  assert.equal(ys.filter((y) => y > 0).length, 0);
  assert.ok(Math.max(...ys) < -0.04, `maxY=${Math.max(...ys)}`);

  let badOverlaps = 0;
  for (let i = 0; i < 500; i++) {
    for (let j = i + 1; j < 500; j++) {
      const a = i * STRIDE, b = j * STRIDE;
      const dist = Math.hypot(data[a] - data[b], data[a + 1] - data[b + 1], data[a + 2] - data[b + 2]);
      if (dist < radius * 0.5) badOverlaps++;
    }
  }
  assert.ok(badOverlaps < 50, `severe overlaps: ${badOverlaps}`);
});

test('spawn packing at various sizes', () => {
  for (const size of [0.5, 1.0, 1.3, 1.5]) {
    const { data, ys } = setup(5000, size);
    assert.equal(data.length, 5000 * STRIDE);
    assert.equal(ys.filter((y) => y > 0).length, 0, `size ${size}`);
  }
});

test('spawn top layer is flat', () => {
  for (const [count, size] of [[5000, 1.0], [5000, 1.3], [10000, 1.0], [10000, 1.3]]) {
    const { radius, ys } = setup(count, size);
    ys.sort((a, b) => a - b);
    const spread = ys.at(-1) - ys[Math.floor(ys.length * 0.95)];
    assert.ok(spread < radius * 6, `count ${count} size ${size}: spread ${spread}`);
  }
});

test('spawned particles are inside the glass with radius set and zero velocity', () => {
  const { radius, profile, data } = setup(2000);
  for (let i = 0; i < 2000; i++) {
    const o = i * STRIDE;
    assert.ok(Math.hypot(data[o], data[o + 2]) < profile.innerRadiusAt(data[o + 1]), `particle ${i}`);
    assert.ok(Math.abs(data[o + 3] - radius) < 1e-7);
    assert.equal(data[o + 4] + data[o + 5] + data[o + 6] + data[o + 7], 0);
  }
});

test('packing scales to the largest count', () => {
  const { data, ys } = setup(200000);
  assert.equal(data.length, 200000 * STRIDE);
  assert.ok(ys.every((y) => y < 0 && Number.isFinite(y)));
});

test('same seed, same spawn', () => {
  assert.deepEqual(setup(3000, 1.5, 42).data, setup(3000, 1.5, 42).data);
});

test('hsbToRgb primaries', () => {
  assert.deepEqual(hsbToRgb(0, 1, 1), [1, 0, 0]);
  assert.deepEqual(hsbToRgb(1 / 3, 1, 1).map((v) => Math.round(v * 1000) / 1000), [0, 1, 0]);
  assert.deepEqual(hsbToRgb(2 / 3, 1, 1).map((v) => Math.round(v * 1000) / 1000), [0, 0, 1]);
  assert.deepEqual(hsbToRgb(0.5, 0, 0.4), [0.4, 0.4, 0.4]);
});

test('particleColors: golden sand by default, varied when random', () => {
  const sand = particleColors(3, false);
  assert.deepEqual([...sand.slice(0, 4)], [194, 153, 71, 255]);
  assert.deepEqual([...sand.slice(8, 12)], [194, 153, 71, 255]);
  const random = particleColors(200, true, mulberry32(7));
  const distinct = new Set();
  for (let i = 0; i < 200; i++) distinct.add(`${random[i * 4]},${random[i * 4 + 1]},${random[i * 4 + 2]}`);
  assert.ok(distinct.size > 100);
  // Saturation 0.7, brightness 0.85: the brightest channel is always 0.85.
  for (let i = 0; i < 200; i++) {
    assert.equal(Math.max(random[i * 4], random[i * 4 + 1], random[i * 4 + 2]), Math.round(0.85 * 255));
  }
});
