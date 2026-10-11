import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MODES, MODE_INFO, BRUTE_FORCE_MAX, gravityScale, neckDamping, cpuSubsteps, gpuSubsteps,
  maxCount, clampCount, clampSize, clampDuration, searchWorkPerFrame,
  sliderToCount, countToSlider, formatCount, sphereDetail, easeInOut,
} from '../js/models/physicsConfig.js';

test('gravity scales inversely with duration, with a floor', () => {
  assert.equal(gravityScale(5), 1);
  assert.equal(gravityScale(10), 0.5);
  assert.ok(Math.abs(gravityScale(30) - 1 / 6) < 1e-12);
  assert.equal(gravityScale(60), 0.15);
});

test('neck damping grows with duration', () => {
  assert.ok(Math.abs(neckDamping(5) - 0.1) < 1e-12);
  assert.ok(Math.abs(neckDamping(30) - 0.6) < 1e-12);
});

test('substep tables match the app', () => {
  assert.deepEqual([100, 500, 501, 1000, 1001].map(cpuSubsteps), [4, 4, 3, 3, 2]);
  assert.deepEqual([1500, 1501, 3000, 3001, 10000, 10001, 200000].map(gpuSubsteps), [4, 3, 3, 2, 2, 1, 1]);
});

test('brute force has a lower ceiling than the grid', () => {
  assert.equal(maxCount('direct', 'grid'), 200000);
  assert.equal(maxCount('direct', 'brute'), BRUTE_FORCE_MAX);
  assert.equal(maxCount('gpu', 'brute'), MODE_INFO.gpu.max);
  assert.equal(maxCount('cpu', 'brute'), MODE_INFO.cpu.max);
});

test('clamps', () => {
  assert.equal(clampCount('cpu', 99999), 2000);
  assert.equal(clampCount('cpu', 1), 50);
  assert.equal(clampCount('direct', 1234), 1250);
  assert.equal(clampCount('direct', 120000, 'brute'), 50000);
  assert.equal(clampCount('gpu', 'nope'), MODE_INFO.gpu.defaultCount);
  assert.equal(clampSize(9), 1.5);
  assert.ok(Math.abs(clampSize(1.31) - 1.3) < 1e-9);
  assert.equal(clampSize('x'), 1);
  assert.equal(clampDuration(12), 10);
  assert.equal(clampDuration(100), 30);
  assert.equal(clampDuration(0), 5);
});

test('search work: all-pairs grows with N squared, the grid with N', () => {
  const brute = (n) => searchWorkPerFrame('direct', n, 'brute').value;
  const grid = (n) => searchWorkPerFrame('direct', n, 'grid').value;
  assert.equal(brute(40000) / brute(20000), 4);
  assert.equal(grid(40000) / grid(20000), 2);
  assert.equal(searchWorkPerFrame('cpu', 100, 'grid').value, 4950 * 4);
  assert.equal(searchWorkPerFrame('direct', 50000, 'brute').value, 2.5e9);
});

test('count slider round-trips and covers the range in every mode', () => {
  for (const mode of MODES) {
    assert.equal(sliderToCount(0, mode), MODE_INFO[mode].min);
    assert.equal(sliderToCount(1000, mode), MODE_INFO[mode].max);
    for (const count of [MODE_INFO[mode].min, MODE_INFO[mode].defaultCount, MODE_INFO[mode].max]) {
      assert.equal(sliderToCount(countToSlider(count, mode), mode), count, `${mode} ${count}`);
    }
    let prev = 0;
    for (let p = 0; p <= 1000; p += 10) {
      const c = sliderToCount(p, mode);
      assert.ok(c >= prev && c % 50 === 0);
      prev = c;
    }
  }
});

test('formatCount', () => {
  assert.deepEqual([50, 950, 1000, 1250, 12500, 200000].map(formatCount), ['50', '950', '1k', '1.25k', '12.5k', '200k']);
});

test('sphere detail drops as the count rises', () => {
  assert.deepEqual([500, 2000, 2001, 60000, 60001].map(sphereDetail), [2, 2, 1, 1, 0]);
});

test('easeInOut starts and ends at rest', () => {
  assert.equal(easeInOut(0), 0);
  assert.equal(easeInOut(1), 1);
  assert.equal(easeInOut(0.5), 0.5);
  assert.equal(easeInOut(2), 1);
  assert.ok(easeInOut(0.01) < 0.001);
});
