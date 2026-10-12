import test from 'node:test';
import assert from 'node:assert/strict';
import { RunTimeline, SETTLE_TAIL, niceCeiling, sqrtPosition } from '../js/models/perfHistory.js';

test('starts empty and ignores samples until a run begins', () => {
  const t = new RunTimeline();
  t.push(1, 5, 5);
  assert.equal(t.length, 0);
  assert.equal(t.span, 0);
  assert.equal(t.recording, false);
  assert.deepEqual(t.latest, { physics: 0, drawing: 0 });
  assert.deepEqual(t.columns('physics', 4), [null, null, null, null]);
});

test('the axis spans flip, countdown and settling, with a mark at each boundary', () => {
  const t = new RunTimeline();
  t.begin(100, 1.2, 5);
  assert.ok(Math.abs(t.span - (1.2 + 5 + SETTLE_TAIL)) < 1e-12);
  assert.deepEqual(t.marks.map((m) => m.t), [1.2, 6.2]);
  assert.equal(t.recording, true);
});

test('records against time since the start, and stops at the end of the span', () => {
  const t = new RunTimeline();
  t.begin(100, 1, 2);          // span 6
  t.push(100.5, 3, 7);
  t.push(105.9, 4, 8);
  t.push(106.5, 99, 99);       // past the end: dropped, and recording ends
  t.push(103, 50, 50);         // so nothing more is taken
  assert.deepEqual(t.times.map((v) => +v.toFixed(3)), [0.5, 5.9]);
  assert.equal(t.recording, false);
  assert.deepEqual(t.latest, { physics: 4, drawing: 8 });
  assert.equal(t.maxPhysics, 4);
});

test('columns keep the worst frame in each column and leave the future empty', () => {
  const t = new RunTimeline();
  t.begin(0, 1, 6);            // span 10; with 10 columns, one column per second
  t.push(0.1, 3, 1);
  t.push(0.5, 290, 2);         // the spike must survive
  t.push(0.9, 4, 1);
  t.push(2.5, 6, 9);
  const physics = t.columns('physics', 10);
  assert.equal(physics[0], 290);
  assert.equal(physics[1], null);
  assert.equal(physics[2], 6);
  assert.deepEqual(physics.slice(3), [null, null, null, null, null, null, null]);
  assert.equal(t.columns('drawing', 10)[2], 9);
});

test('a new run clears the last one; clear() empties everything', () => {
  const t = new RunTimeline();
  t.begin(0, 1, 5);
  t.push(1, 50, 60);
  t.begin(20, 1, 10);
  assert.equal(t.length, 0);
  assert.equal(t.maxPhysics, 0);
  t.push(21, 5, 5);
  t.clear();
  assert.equal(t.span, 0);
  assert.equal(t.length, 0);
  assert.equal(t.recording, false);
});

test('bad samples count as zero', () => {
  const t = new RunTimeline();
  t.begin(0, 1, 5);
  t.push(1, NaN, -5);
  t.push(2, undefined, Infinity);
  assert.deepEqual(t.physics, [0, 0]);
  assert.deepEqual(t.drawing, [0, 0]);
});

test('niceCeiling picks a round axis top and respects the floor', () => {
  assert.deepEqual([0, 7, 20, 21, 31, 290, 1200].map((v) => niceCeiling(v)), [20, 20, 20, 25, 50, 500, 2000]);
});

test('sqrtPosition spreads small values and clamps to the axis', () => {
  assert.equal(sqrtPosition(0, 500), 0);
  assert.equal(sqrtPosition(500, 500), 1);
  assert.equal(sqrtPosition(9999, 500), 1);
  assert.equal(sqrtPosition(125, 500), 0.5);
  // 5 ms on a 500 ms axis: 1% of the way up if linear, 10% here.
  assert.ok(Math.abs(sqrtPosition(5, 500) - 0.1) < 1e-12);
  assert.equal(sqrtPosition(5, 0), 0);
});
