// Ports of CPUPhysicsTests.swift, plus a few web-only checks.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createProfile, neckRadiusForParticle } from '../js/models/hourglassProfile.js';
import { STRIDE, radiusForCount, mulberry32 } from '../js/models/packing.js';
import { CpuSimulation, summarise } from '../js/physics/cpuSimulation.js';

const DT = 1 / 60;

function makeSim(count, radius = radiusForCount(count), profile = null) {
  return new CpuSimulation({
    count,
    particleRadius: radius,
    profile: profile ?? createProfile(neckRadiusForParticle(radius)),
    random: mulberry32(1),
  });
}

test('gravity moves a particle down', () => {
  const sim = makeSim(1, 0.02, createProfile());
  sim.setParticle(0, { position: [0, -0.2, 0] });
  sim.step(DT);
  assert.ok(sim.getParticle(0).position[1] < -0.2);
});

test('particle stays above the floor', () => {
  const sim = makeSim(1, 0.02, createProfile());
  sim.setParticle(0, { position: [0, -0.48 + 0.02, 0], velocity: [0, -1, 0] });
  for (let i = 0; i < 100; i++) sim.step(DT);
  const p = sim.getParticle(0);
  assert.ok(p.position[1] - p.radius >= -0.49);
});

test('overlapping particles separate', () => {
  const sim = makeSim(2, 0.02, createProfile());
  sim.setParticle(0, { position: [0, -0.3, 0] });
  sim.setParticle(1, { position: [0.01, -0.3, 0] });
  for (let i = 0; i < 60; i++) sim.step(DT);
  const a = sim.getParticle(0).position, b = sim.getParticle(1).position;
  assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) >= 0.04 * 0.95);
});

test('flip transform mirrors y and z', () => {
  const sim = makeSim(1, 0.02, createProfile());
  sim.setParticle(0, { position: [0.05, -0.3, 0.02], velocity: [0.1, -0.5, 0.3] });
  sim.containerEulerX = Math.PI;
  sim.snapAfterFlip();
  const p = sim.getParticle(0);
  const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-6, `${a} vs ${b}`);
  close(p.position[0], 0.05); close(p.position[1], 0.3); close(p.position[2], -0.02);
  close(p.velocity[0], 0.1); close(p.velocity[1], 0.5); close(p.velocity[2], -0.3);
  assert.equal(sim.containerEulerX, 0);
});

test('particles drain to the lower chamber', () => {
  const count = 50;
  const sim = makeSim(count);
  assert.equal(summarise(sim.data, count).upper, 0, 'all start in the lower chamber');
  sim.snapAfterFlip();
  assert.equal(summarise(sim.data, count).upper, count, 'all in the upper chamber after the flip');
  for (let i = 0; i < 600; i++) sim.step(DT);
  const { drainedPercent } = summarise(sim.data, count);
  assert.ok(drainedPercent > 80, `drained ${drainedPercent}%`);
});

test('particles settle to rest', () => {
  const count = 50;
  const sim = makeSim(count);
  for (let i = 0; i < 300; i++) sim.step(DT);
  const { avgSpeed } = summarise(sim.data, count);
  assert.ok(avgSpeed < 0.05, `avgSpeed=${avgSpeed}`);
});

test('particles stay inside the glass', () => {
  const count = 50;
  const sim = makeSim(count);
  for (let i = 0; i < 120; i++) sim.step(DT);
  for (let i = 0; i < count; i++) {
    const p = sim.getParticle(i);
    const radial = Math.hypot(p.position[0], p.position[2]);
    assert.ok(radial < sim.profile.innerRadiusAt(p.position[1]) + 0.01, `particle ${i}`);
    assert.ok(p.position[1] - p.radius >= -0.50);
    assert.ok(p.position[1] + p.radius <= 0.50);
  }
});

test('gravity follows the container during the flip', () => {
  // Half way round, "down" in the container's frame is +z.
  const sim = makeSim(1, 0.02, createProfile());
  sim.setParticle(0, { position: [0, -0.3, 0] });
  sim.containerEulerX = Math.PI / 2;
  sim.step(DT);
  const p = sim.getParticle(0);
  assert.ok(p.velocity[2] > 0);
  assert.ok(Math.abs(p.velocity[1]) < 1e-6);
});

test('longer durations flow more slowly', () => {
  const upperAfter = (gravity, neckDamping) => {
    const sim = makeSim(100);
    sim.gravity = gravity;
    sim.neckDamping = neckDamping;
    sim.snapAfterFlip();
    for (let i = 0; i < 150; i++) sim.step(DT);
    return summarise(sim.data, 100).upper;
  };
  const fast = upperAfter(1.0, 0.1);    // 5 s setting
  const slow = upperAfter(5 / 30, 0.6); // 30 s setting
  assert.ok(slow > fast, `slow=${slow} fast=${fast}`);
});

test('simulation can write straight into a caller-owned buffer', () => {
  const target = new Float32Array(20 * STRIDE);
  const radius = radiusForCount(20);
  const sim = new CpuSimulation({
    count: 20, particleRadius: radius,
    profile: createProfile(neckRadiusForParticle(radius)), target,
  });
  assert.equal(sim.data, target);
  assert.ok(Math.abs(target[3] - radius) < 1e-7);
  const before = target[1];
  sim.snapAfterFlip();
  assert.ok(Math.abs(target[1] + before) < 1e-7);
});

test('no NaNs after a long mixed run', () => {
  const sim = makeSim(200);
  for (let i = 0; i < 72; i++) { sim.containerEulerX = (i / 72) * Math.PI; sim.step(DT); }
  sim.snapAfterFlip();
  for (let i = 0; i < 200; i++) sim.step(1 / 30);
  assert.equal(summarise(sim.data, 200).nonFinite, 0);
});
