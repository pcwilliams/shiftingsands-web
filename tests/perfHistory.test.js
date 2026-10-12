import test from 'node:test';
import assert from 'node:assert/strict';
import { PerfHistory, niceCeiling } from '../js/models/perfHistory.js';

test('starts empty', () => {
  const h = new PerfHistory(4);
  assert.deepEqual(h.series('physics'), []);
  assert.deepEqual(h.latest, { physics: 0, drawing: 0 });
  assert.equal(h.windowMax, 0);
});

test('keeps the newest samples, oldest first', () => {
  const h = new PerfHistory(3);
  for (const v of [1, 2, 3, 4, 5]) h.push(v, v * 10);
  assert.deepEqual(h.series('physics'), [3, 4, 5]);
  assert.deepEqual(h.series('drawing'), [30, 40, 50]);
  assert.deepEqual(h.latest, { physics: 5, drawing: 50 });
});

test('run maximum outlives the window; window maximum does not', () => {
  const h = new PerfHistory(3);
  h.push(290, 4);
  for (let i = 0; i < 5; i++) h.push(3, 6);
  assert.equal(h.maxPhysics, 290);
  assert.equal(h.maxDrawing, 6);
  assert.equal(h.windowMax, 6);
});

test('reset clears samples and maxima', () => {
  const h = new PerfHistory(3);
  h.push(50, 60);
  h.reset();
  assert.equal(h.length, 0);
  assert.equal(h.maxPhysics, 0);
  assert.equal(h.maxDrawing, 0);
});

test('bad samples count as zero', () => {
  const h = new PerfHistory(3);
  h.push(NaN, -5);
  h.push(undefined, Infinity);
  assert.deepEqual(h.series('physics'), [0, 0]);
  assert.deepEqual(h.series('drawing'), [0, 0]);
});

test('niceCeiling picks a round axis top and respects the floor', () => {
  assert.equal(niceCeiling(0), 20);
  assert.equal(niceCeiling(7), 20);
  assert.equal(niceCeiling(20), 20);
  assert.equal(niceCeiling(21), 25);
  assert.equal(niceCeiling(31), 50);
  assert.equal(niceCeiling(290), 500);
  assert.equal(niceCeiling(1200), 2000);
});
