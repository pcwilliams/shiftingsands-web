import test from 'node:test';
import assert from 'node:assert/strict';
import { SettingsStore, memoryStorage } from '../js/services/settingsStore.js';
import { parseLaunchArgs, applyLaunchArgs } from '../js/services/launchArgs.js';

test('defaults', () => {
  const s = new SettingsStore(memoryStorage());
  assert.equal(s.mode, 'direct');
  assert.equal(s.broadphase, 'grid');
  assert.equal(s.count, 10000);
  assert.equal(s.size, 1);
  assert.equal(s.duration, 5);
  assert.equal(s.randomColors, false);
});

test('each mode remembers its own count and size', () => {
  const storage = memoryStorage();
  const s = new SettingsStore(storage);
  s.setCount(40000);
  s.setSize(1.3);
  s.setMode('cpu');
  assert.equal(s.count, 250);
  assert.equal(s.size, 1);
  s.setCount(600);
  s.setMode('direct');
  assert.equal(s.count, 40000);
  assert.equal(s.size, 1.3);

  const again = new SettingsStore(storage);
  assert.equal(again.mode, 'direct');
  assert.equal(again.count, 40000);
  again.setMode('cpu');
  assert.equal(again.count, 600);
});

test('everything persists across a reload', () => {
  const storage = memoryStorage();
  const s = new SettingsStore(storage);
  s.setMode('gpu');
  s.setBroadphase('brute');
  s.setDuration(30);
  s.setRandomColors(true);
  const again = new SettingsStore(storage);
  assert.deepEqual(
    [again.mode, again.broadphase, again.duration, again.randomColors],
    ['gpu', 'brute', 30, true],
  );
});

test('switching to brute force pulls the count under its ceiling', () => {
  const s = new SettingsStore(memoryStorage());
  s.setCount(200000);
  const changes = [];
  s.subscribe((_, change) => changes.push(change));
  s.setBroadphase('brute');
  assert.equal(s.count, 50000);
  assert.deepEqual(changes, ['broadphase']);
});

test('corrupt storage falls back to defaults', () => {
  const storage = memoryStorage();
  storage.setItem('shiftingsands.mode', 'quantum');
  storage.setItem('shiftingsands.count.direct', 'lots');
  storage.setItem('shiftingsands.size.direct', '99');
  storage.setItem('shiftingsands.duration', '-4');
  storage.setItem('shiftingsands.broadphase', '???');
  const s = new SettingsStore(storage);
  assert.deepEqual([s.mode, s.count, s.size, s.duration, s.broadphase], ['direct', 10000, 1, 5, 'grid']);
});

test('storage that throws does not break the store', () => {
  const broken = {
    getItem() { throw new Error('denied'); },
    setItem() { throw new Error('denied'); },
  };
  const s = new SettingsStore(broken);
  s.setCount(500);
  assert.equal(s.count, 500);
});

test('no notification when nothing changed', () => {
  const s = new SettingsStore(memoryStorage());
  let calls = 0;
  s.subscribe(() => calls++);
  s.setMode('direct');
  s.setCount(10000);
  s.setSize(1);
  s.setDuration(5);
  s.setRandomColors(false);
  s.setBroadphase('grid');
  s.setMode('nonsense');
  assert.equal(calls, 0);
});

test('parseLaunchArgs: nothing given means no overrides', () => {
  assert.deepEqual(parseLaunchArgs(''), {
    mode: null, broadphase: null, count: null, size: null, duration: null,
    colors: null, autostart: false, test: false, stats: false, seed: null,
  });
});

test('parseLaunchArgs: values, aliases and flags', () => {
  const a = parseLaunchArgs('?mode=METAL&count=50000&size=1.3&duration=10&broadphase=n2&colors&autostart&seed=7&stats');
  assert.deepEqual(a, {
    mode: 'direct', broadphase: 'brute', count: 50000, size: 1.3, duration: 10,
    colors: true, autostart: true, test: false, stats: true, seed: 7,
  });
  assert.equal(parseLaunchArgs('?mode=cpu').mode, 'cpu');
  assert.equal(parseLaunchArgs('?mode=gpu').mode, 'gpu');
  assert.equal(parseLaunchArgs('?multicolor').colors, true);
  assert.equal(parseLaunchArgs('?broadphase=sorted').broadphase, 'sorted');
  assert.equal(parseLaunchArgs('?broadphase=sort').broadphase, 'sorted');
  assert.equal(parseLaunchArgs('?colors=0').colors, false);
});

test('parseLaunchArgs: junk is ignored', () => {
  const a = parseLaunchArgs('?mode=warp&count=many&size=&broadphase=octree&seed=x');
  assert.deepEqual([a.mode, a.count, a.size, a.broadphase, a.seed], [null, null, null, null, null]);
});

test('applyLaunchArgs: URL beats saved settings, and count beats the mode\'s saved count', () => {
  const storage = memoryStorage();
  const saved = new SettingsStore(storage);
  saved.setMode('cpu');
  saved.setCount(300);
  const s = new SettingsStore(storage);
  applyLaunchArgs(parseLaunchArgs('?mode=direct&count=20000&size=0.5&colors'), s);
  assert.deepEqual([s.mode, s.count, s.size, s.randomColors], ['direct', 20000, 0.5, true]);
  // The mode from a link is not saved: a plain visit afterwards opens as before.
  assert.equal(new SettingsStore(storage).mode, 'cpu');
});

test('applyLaunchArgs: ?test forces a 10 second run', () => {
  const s = new SettingsStore(memoryStorage());
  applyLaunchArgs(parseLaunchArgs('?test&duration=30'), s);
  assert.equal(s.duration, 10);
});

test('applyLaunchArgs: out-of-range values are clamped to the mode', () => {
  const s = new SettingsStore(memoryStorage());
  applyLaunchArgs(parseLaunchArgs('?mode=cpu&count=999999&size=7'), s);
  assert.deepEqual([s.count, s.size], [2000, 1.5]);
});
