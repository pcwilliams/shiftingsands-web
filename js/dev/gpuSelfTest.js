// Browser-side tests for the WGSL physics: ports of GPUPhysicsTests.swift.
//
// Node can't run WebGPU, so these run in a page (gpu-tests.html). Each test
// runs twice, once per broadphase, and one extra test checks that the two
// broadphases compute the same step.

import { createProfile, neckRadiusForParticle } from '../models/hourglassProfile.js';
import { STRIDE, radiusForCount, packedParticles, mulberry32 } from '../models/packing.js';
import { summarise } from '../physics/cpuSimulation.js';
import { computeGrid, cellCoords, hashCell } from '../models/grid.js';
import { GpuPhysics } from '../physics/gpuPhysics.js';

const DT = 1 / 60;

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

async function read(engine) {
  const out = new Float32Array(engine.count * STRIDE);
  for (let attempt = 0; attempt < 5; attempt++) {
    if (await engine.readInto(out)) return out;
  }
  throw new Error('readback failed');
}

function configure(engine, broadphase, count, radius, particles = null) {
  const profile = createProfile(neckRadiusForParticle(radius));
  engine.gravity = 1;
  engine.neckDamping = 0;
  engine.configure({
    count, particleRadius: radius, profile, broadphase,
    particles: particles ?? packedParticles(count, radius, profile, mulberry32(1)),
  });
}

function particle(x, y, z, r, vx = 0, vy = 0, vz = 0) {
  return [x, y, z, r, vx, vy, vz, 0];
}

const TESTS = {
  async 'gravity moves a particle down'(engine, bp) {
    configure(engine, bp, 1, 0.02, new Float32Array(particle(0, -0.2, 0, 0.02)));
    engine.step(DT);
    const p = await read(engine);
    expect(p[1] < -0.2, `y=${p[1]}`);
  },

  async 'particle stays above the floor'(engine, bp) {
    configure(engine, bp, 1, 0.02, new Float32Array(particle(0, -0.46, 0, 0.02, 0, -1, 0)));
    for (let i = 0; i < 100; i++) engine.step(DT);
    const p = await read(engine);
    expect(p[1] - p[3] >= -0.49, `y-r=${p[1] - p[3]}`);
  },

  async 'flip transform mirrors y and z'(engine, bp) {
    configure(engine, bp, 1, 0.02, new Float32Array(particle(0.05, -0.3, 0.02, 0.02, 0.1, -0.5, 0.3)));
    engine.snapAfterFlip();
    const p = await read(engine);
    const want = [0.05, 0.3, -0.02, 0.02, 0.1, 0.5, -0.3, 0];
    want.forEach((w, i) => expect(Math.abs(p[i] - w) < 1e-6, `field ${i}: ${p[i]} vs ${w}`));
  },

  async 'particles drain to the lower chamber'(engine, bp) {
    const count = 100;
    configure(engine, bp, count, radiusForCount(count));
    expect(summarise(await read(engine), count).upper === 0, 'all should start in the lower chamber');
    engine.snapAfterFlip();
    expect(summarise(await read(engine), count).upper === count, 'all should be in the upper chamber after the flip');
    for (let i = 0; i < 600; i++) engine.step(DT);
    const s = summarise(await read(engine), count);
    expect(s.nonFinite === 0, 'NaN in particle data');
    expect(s.drainedPercent > 80, `drained ${s.drainedPercent}%`);
    return `${s.drainedPercent.toFixed(0)}% drained`;
  },

  async 'particles settle and go to sleep'(engine, bp) {
    const count = 100;
    configure(engine, bp, count, radiusForCount(count));
    for (let i = 0; i < 300; i++) engine.step(DT);
    const s = summarise(await read(engine), count);
    expect(s.avgSpeed < 0.05, `avgSpeed=${s.avgSpeed}`);
    expect(s.sleeping / count > 0.5, `only ${s.sleeping} of ${count} asleep`);
    return `${s.sleeping} asleep`;
  },

  async 'sleeping particle wakes when its support is removed'(engine, bp) {
    const r = 0.02;
    const floorY = -0.48 + r;
    // Index 7 on purpose: its turn in the 30-step stagger is not step 0.
    const count = 8;
    const data = new Float32Array(count * STRIDE);
    for (let i = 0; i < count; i++) data.set(particle(-0.15 + i * 0.001, floorY, 0.1, r), i * STRIDE);
    data.set(particle(0, floorY, 0, r), 0);            // B: on the floor
    data.set(particle(0, floorY + r * 2, 0, r), 7 * STRIDE); // A: resting on B
    for (let i = 1; i < 7; i++) data.set(particle(0.1, floorY, -0.15 + i * 0.045, r), i * STRIDE);
    configure(engine, bp, count, r, data);
    for (let i = 0; i < 300; i++) engine.step(DT);
    let p = await read(engine);
    expect(p[7 * STRIDE + 7] > 30, `A should be in deep sleep (counter=${p[7 * STRIDE + 7]})`);
    const before = p[7 * STRIDE + 1];
    p.set(particle(-0.1, floorY, -0.1, r), 0);         // take B away
    engine.upload(p);
    for (let i = 0; i < 60; i++) engine.step(DT);
    p = await read(engine);
    expect(p[7 * STRIDE + 1] < before - r, `A should fall (before=${before}, after=${p[7 * STRIDE + 1]})`);
  },

  async 'a clump asleep in mid-air wakes and falls'(engine, bp) {
    // Grains can fall asleep while stacked in the pouring column and then
    // lose whatever was under them. If "support" just means "touching
    // something", the leftover clump holds itself up for ever.
    const r = 0.012;
    const y = -0.25;   // lower chamber, well above the floor
    const clump = [[0, y, 0], [r * 1.9, y, 0], [r * 0.95, y + r * 1.6, 0], [r * 0.95, y + r * 0.5, r * 1.6], [r * 0.95, y - r * 1.5, r * 0.5]];
    const data = new Float32Array(clump.length * STRIDE);
    clump.forEach((p, i) => data.set([...p, r, 0, 0, 0, 40], i * STRIDE));   // 40 = deep sleep
    configure(engine, bp, clump.length, r, data);
    for (let i = 0; i < 360; i++) engine.step(1 / 120);
    const p = await read(engine);
    let highest = -1;
    for (let i = 0; i < clump.length; i++) highest = Math.max(highest, p[i * STRIDE + 1]);
    expect(highest < -0.48 + r * 5, `a grain is still hanging at y=${highest.toFixed(3)}`);
    return `all on the floor (highest y ${highest.toFixed(3)})`;
  },

  async 'a grain against the side of the glass does not stick there'(engine, bp) {
    // The wall of the lower bulb leans in only slightly below its widest
    // point. That is not a ledge: a grain touching it must still fall.
    const r = 0.0056;
    const count = 2;
    const profile = createProfile(neckRadiusForParticle(r));
    const table = profile.buildLookupTable();
    const y = -0.36;
    const wall = table[Math.round((y + 0.5) * 255)] - r;      // touching the glass
    const data = new Float32Array(count * STRIDE);
    data.set([wall, y, 0, r, 0, 0, 0, 40], 0);                // asleep on the wall
    data.set([0, y, -wall, r, 0, 0, 0, 0], STRIDE);           // awake and at rest on the wall
    configure(engine, bp, count, r, data);
    for (let i = 0; i < 240; i++) engine.step(1 / 120);
    const p = await read(engine);
    const highest = Math.max(p[1], p[STRIDE + 1]);
    expect(highest < -0.44, `a grain is still on the wall at y=${highest.toFixed(3)}`);
    return `both at the bottom (highest y ${highest.toFixed(3)})`;
  },

  async 'the pile comes to rest level and centred'(engine, bp) {
    // The glass is symmetric, so the settled pile must be too. Two earlier
    // grids failed this: one swept neighbours in a fixed direction and left
    // the pile 14 radii off-centre; the next alternated direction by odd and
    // even grain, which centred it but left a heap on one side.
    const count = 2000;
    const radius = radiusForCount(count);
    configure(engine, bp, count, radius);
    engine.neckDamping = 0.1;
    engine.snapAfterFlip();
    for (let i = 0; i < 600; i++) {
      engine.step(1 / 120);
      if (i % 16 === 0) await engine.device.queue.onSubmittedWorkDone();
    }
    const p = await read(engine);
    expect(summarise(p, count).upper === 0, 'the sand should have drained');

    let x = 0, z = 0;
    const sectors = Array.from({ length: 8 }, () => []);
    for (let i = 0; i < count; i++) {
      const px = p[i * STRIDE], pz = p[i * STRIDE + 2];
      x += px; z += pz;
      if (Math.hypot(px, pz) > 0.05) {
        sectors[Math.min(7, Math.floor((Math.atan2(pz, px) + Math.PI) / (Math.PI / 4)))].push(p[i * STRIDE + 1]);
      }
    }
    const offX = x / count / radius, offZ = z / count / radius;
    expect(Math.abs(offX) < 1.5 && Math.abs(offZ) < 1.5, `pile centre is off by ${offX.toFixed(2)}, ${offZ.toFixed(2)} radii`);

    // Height of the surface in each of eight slices round the axis.
    const tops = sectors.map((ys) => { ys.sort((a, b) => a - b); return ys[Math.floor(ys.length * 0.97)]; });
    const spread = (Math.max(...tops) - Math.min(...tops)) / radius;
    const sizes = sectors.map((ys) => ys.length);
    const ratio = Math.max(...sizes) / Math.min(...sizes);
    expect(spread < 2, `surface height varies by ${spread.toFixed(1)} radii round the pile`);
    expect(ratio < 1.4, `fullest slice holds ${ratio.toFixed(2)}x the emptiest`);
    return `surface level to ${spread.toFixed(1)} radii, slices within ${ratio.toFixed(2)}x`;
  },

  async 'no particle escapes or turns to NaN at 5,000'(engine, bp) {
    const count = 5000;
    const radius = radiusForCount(count);
    configure(engine, bp, count, radius);
    engine.snapAfterFlip();
    for (let i = 0; i < 90; i++) engine.step(DT);
    const p = await read(engine);
    const s = summarise(p, count);
    expect(s.nonFinite === 0, `${s.nonFinite} non-finite particles`);
    expect(s.minY >= -0.5 && s.maxY <= 0.5, `y range ${s.minY}..${s.maxY}`);
    let worst = 0;
    for (let i = 0; i < count; i++) worst = Math.max(worst, Math.hypot(p[i * STRIDE], p[i * STRIDE + 2]));
    expect(worst < 0.24, `radial ${worst}`);
    return `${s.lower} through after 1.5 s`;
  },
};

/** Both broadphases must produce the same step from the same state. */
async function broadphasesAgree(engine) {
  const count = 3000;
  const radius = radiusForCount(count);
  const results = {};
  for (const bp of ['brute', 'grid']) {
    configure(engine, bp, count, radius);
    engine.snapAfterFlip();
    // One substep each, so the comparison isn't blurred by chaos.
    for (let i = 0; i < 6; i++) engine.step(DT / 8);
    results[bp] = await read(engine);
  }
  let worst = 0;
  for (let i = 0; i < count * STRIDE; i++) worst = Math.max(worst, Math.abs(results.brute[i] - results.grid[i]));
  expect(worst < 1e-3, `largest difference ${worst}`);
  return `max difference ${worst.toExponential(1)}`;
}

/**
 * One touching pair, plus thirty lone grains placed in far-apart cells chosen
 * because they hash to the pair's bucket. The bucket's list is long, and the
 * pair must still find each other. (A bucket with a fixed number of slots
 * failed exactly here: measured mid-pour, one cell at the neck held 938
 * grains.) With a single contact the result doesn't depend on visiting
 * order, so brute force and grid must match to rounding.
 */
async function crowdedBucketKeepsEveryone(engine) {
  const radius = 0.008;
  const fillers = 30;
  const count = fillers + 2;
  const grid = computeGrid(radius, count);
  const a = [0, -0.3, 0];
  const b = [0.012, -0.3, 0];
  const bucket = hashCell(grid, ...cellCoords(grid, ...a));

  const data = new Float32Array(count * STRIDE);
  // The pair goes last, where slot-per-bucket schemes run out of room.
  const pairAt = fillers * STRIDE;
  data.set(particle(...a, radius), pairAt);
  data.set(particle(...b, radius), pairAt + STRIDE);
  const chosen = [];
  const across = Math.ceil(0.5 / grid.cellSize);
  const up = Math.ceil(1.04 / grid.cellSize);
  search:
  for (let cy = 0; cy < up; cy++) {
    for (let cx = 0; cx < across; cx++) {
      for (let cz = 0; cz < across; cz++) {
        if (hashCell(grid, cx, cy, cz) !== bucket) continue;
        const centre = [0, 1, 2].map((k) => grid.origin[k] + ([cx, cy, cz][k] + 0.5) * grid.cellSize);
        const inGlass = Math.hypot(centre[0], centre[2]) < 0.14 && Math.abs(centre[1]) > 0.16 && Math.abs(centre[1]) < 0.4;
        const clear = [a, b, ...chosen].every((p) => Math.hypot(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]) > radius * 3);
        if (!inGlass || !clear) continue;
        chosen.push(centre);
        if (chosen.length === fillers) break search;
      }
    }
  }
  expect(chosen.length === fillers, `only found ${chosen.length} cells sharing the bucket`);
  chosen.forEach((p, i) => data.set(particle(...p, radius), i * STRIDE));

  const results = {};
  for (const bp of ['brute', 'grid']) {
    configure(engine, bp, count, radius, data);
    engine.gravity = 0;
    engine.step(DT);
    results[bp] = await read(engine);
  }
  const gap = (p) => Math.hypot(p[pairAt] - p[pairAt + STRIDE], p[pairAt + 1] - p[pairAt + STRIDE + 1], p[pairAt + 2] - p[pairAt + STRIDE + 2]);
  expect(gap(results.grid) > 0.0125, `the pair did not push apart (gap ${gap(results.grid)})`);
  let worst = 0;
  for (let i = 0; i < count * STRIDE; i++) worst = Math.max(worst, Math.abs(results.brute[i] - results.grid[i]));
  expect(worst < 1e-5, `largest difference ${worst}`);
  return `${fillers + 1} grains in one bucket, max difference ${worst.toExponential(1)}`;
}

export async function runGpuSelfTest(device, report = () => {}, only = '') {
  const engine = await GpuPhysics.create(device);
  const results = [];
  const run = async (name, fn) => {
    if (only && !name.includes(only)) return;
    const started = performance.now();
    let entry;
    try {
      const note = await fn();
      entry = { name, pass: true, note: note ?? '' };
    } catch (error) {
      entry = { name, pass: false, note: error.message };
    }
    entry.ms = Math.round(performance.now() - started);
    results.push(entry);
    report(entry);
  };
  for (const bp of ['brute', 'grid']) {
    for (const [name, fn] of Object.entries(TESTS)) {
      await run(`${name} [${bp}]`, () => fn(engine, bp));
    }
  }
  await run('brute force and grid agree', () => broadphasesAgree(engine));
  await run('a crowded bucket keeps everyone', () => crowdedBucketKeepsEveryone(engine));
  engine.destroyBuffers();
  return results;
}
