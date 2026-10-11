import test from 'node:test';
import assert from 'node:assert/strict';
import { TimerModel } from '../js/services/timerModel.js';

test('starts idle with the full duration showing', () => {
  const t = new TimerModel(5);
  assert.equal(t.state, 'idle');
  assert.equal(t.remainingText, '0:05');
  assert.equal(t.buttonLabel, 'Start');
  assert.equal(t.progress, 0);
});

test('two-phase start: the clock waits for the flip', () => {
  const t = new TimerModel(5);
  assert.equal(t.startFlip(), true);
  assert.equal(t.state, 'flipping');
  assert.equal(t.buttonLabel, 'Reset');
  t.tick(100);
  assert.equal(t.elapsed, 0, 'ticks during the flip do nothing');
  assert.equal(t.startFlip(), false, 'a second start during the flip is ignored');
  assert.equal(t.completeFlip(101.2), true);
  assert.equal(t.state, 'running');
  t.tick(103.2);
  assert.equal(t.elapsed, 2);
  assert.equal(t.remainingText, '0:03');
});

test('completeFlip only works from flipping', () => {
  const t = new TimerModel(5);
  assert.equal(t.completeFlip(1), false);
  assert.equal(t.state, 'idle');
});

test('runs to completion and stops', () => {
  const t = new TimerModel(10);
  t.startFlip();
  t.completeFlip(0);
  t.tick(9.99);
  assert.equal(t.state, 'running');
  assert.equal(t.remainingText, '0:00');
  t.tick(10);
  assert.equal(t.state, 'complete');
  assert.equal(t.elapsed, 10);
  assert.equal(t.progress, 1);
  assert.equal(t.buttonLabel, 'Start');
  assert.equal(t.isIdle, true);
  t.tick(50);
  assert.equal(t.elapsed, 10);
});

test('reset returns to idle from any state', () => {
  const t = new TimerModel(5);
  t.startFlip();
  t.reset();
  assert.equal(t.state, 'idle');
  t.startFlip();
  t.completeFlip(0);
  t.tick(2);
  t.reset();
  assert.equal(t.state, 'idle');
  assert.equal(t.elapsed, 0);
});

test('duration can only change at rest', () => {
  const t = new TimerModel(5);
  assert.equal(t.setDuration(30), true);
  assert.equal(t.remainingText, '0:30');
  t.startFlip();
  assert.equal(t.setDuration(10), false);
  assert.equal(t.duration, 30);
});

test('subscribers hear state changes and each new second, not every frame', () => {
  const t = new TimerModel(5);
  let calls = 0;
  const off = t.subscribe(() => calls++);
  t.startFlip();
  t.completeFlip(0);
  assert.equal(calls, 2);
  for (let f = 1; f <= 60; f++) t.tick(f / 60);
  assert.equal(calls, 3, 'one notification when 0:05 becomes 0:04');
  off();
  t.reset();
  assert.equal(calls, 3);
});

test('long durations format as minutes', () => {
  assert.equal(new TimerModel(75).remainingText, '1:15');
});
