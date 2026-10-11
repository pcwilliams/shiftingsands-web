import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createProfile, createRevolutionSurface, catmullRom, sampleLookupTable,
  neckRadiusForParticle, WALL_THICKNESS, WALL_CLEARANCE, PROFILE_TABLE_SIZE,
} from '../js/models/hourglassProfile.js';

test('catmullRom passes through its middle control points', () => {
  assert.equal(catmullRom(0, 1, 2, 3, 0), 1);
  assert.equal(catmullRom(0, 1, 2, 3, 1), 2);
  assert.ok(Math.abs(catmullRom(0, 1, 2, 3, 0.5) - 1.5) < 1e-12);
});

test('default profile has the app\'s point count and end points', () => {
  const { outer, inner } = createProfile();
  // 10 spans: the two end spans contribute 1 point, the other 8 contribute 8, plus the last point.
  assert.equal(outer.length, 2 + 8 * 8 + 1);
  assert.equal(inner.length, outer.length);
  assert.deepEqual(outer[0], { radius: 0, height: -0.5 });
  assert.deepEqual(outer.at(-1), { radius: 0, height: 0.5 });
});

test('profile is symmetric top to bottom (the flip relies on this)', () => {
  for (const profile of [createProfile(), createProfile(0.03)]) {
    for (const y of [0.01, 0.04, 0.1, 0.2, 0.3, 0.45]) {
      assert.ok(Math.abs(profile.innerRadiusAt(y) - profile.innerRadiusAt(-y)) < 1e-9, `y=${y}`);
    }
  }
});

test('inner profile is the outer profile minus the wall', () => {
  const { outer, inner } = createProfile(0.03);
  outer.forEach((p, i) => {
    assert.ok(Math.abs(inner[i].radius - Math.max(p.radius - WALL_THICKNESS, 0)) < 1e-12);
    assert.equal(inner[i].height, p.height);
  });
});

test('innerRadiusAt: neck is the narrowest point, bulb is the widest', () => {
  const profile = createProfile(0.03);
  assert.ok(Math.abs(profile.innerRadiusAt(0) - (0.03 - WALL_THICKNESS)) < 1e-9);
  // The spline overshoots its 0.22 control points a little, so the bulb is slightly wider.
  const bulb = profile.innerRadiusAt(-0.3);
  assert.ok(bulb > 0.22 - WALL_THICKNESS && bulb < 0.23, `bulb=${bulb}`);
  // Outside the shoulders the glass is always wider than the neck.
  for (let y = 0.05; y <= 0.45; y += 0.01) {
    assert.ok(profile.innerRadiusAt(y) > profile.innerRadiusAt(0), `y=${y}`);
    assert.ok(profile.innerRadiusAt(-y) > profile.innerRadiusAt(0), `y=${-y}`);
  }
});

test('the spline undershoots just beside a sharp neck (same as the app)', () => {
  // Catmull-Rom overshoots where the curve turns sharply, so the narrowest
  // point is not at y = 0 and is narrower than the nominal neck. The physics
  // depends on this: in the pinch a particle is steered onto the axis.
  const profile = createProfile(neckRadiusForParticle(0.0087));
  let min = Infinity;
  for (let y = -0.04; y <= 0.04; y += 0.0005) min = Math.min(min, profile.innerRadiusAt(y));
  assert.ok(min < profile.innerRadiusAt(0));
  assert.ok(min >= 0);
});

test('dynamic neck lets exactly one particle through', () => {
  const r = 0.012;
  const profile = createProfile(neckRadiusForParticle(r));
  const gap = profile.innerRadiusAt(0);
  assert.ok(gap > r, 'one fits');
  assert.ok(gap < 2 * r, 'two side by side do not');
});

test('lookup table matches the linear search it replaces', () => {
  const profile = createProfile(0.025);
  const table = profile.buildLookupTable();
  assert.equal(table.length, PROFILE_TABLE_SIZE);
  // Away from the sharp neck the 256-sample table is within a fraction of a millimetre-unit.
  for (let y = -0.47; y <= 0.47; y += 0.013) {
    if (Math.abs(y) < 0.05) continue;
    const exact = profile.innerRadiusAt(y) - WALL_CLEARANCE;
    assert.ok(Math.abs(sampleLookupTable(table, y) - exact) < 0.002, `y=${y}`);
  }
  // The neck itself is at y = 0, which falls between two samples.
  assert.ok(Math.abs(sampleLookupTable(table, 0) - (profile.innerRadiusAt(0) - WALL_CLEARANCE)) < 0.002);
});

test('revolution surface: counts, unit normals, indices in range', () => {
  const { outer } = createProfile();
  const segments = 16;
  const { positions, normals, uvs, indices } = createRevolutionSurface(outer, segments);
  const vertexCount = outer.length * (segments + 1);
  assert.equal(positions.length, vertexCount * 3);
  assert.equal(uvs.length, vertexCount * 2);
  assert.equal(indices.length, (outer.length - 1) * segments * 6);
  assert.ok(indices.every((i) => i < vertexCount));
  for (let v = 0; v < vertexCount; v++) {
    const len = Math.hypot(normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]);
    assert.ok(Math.abs(len - 1) < 1e-4, `normal ${v} length ${len}`);
  }
  // Every vertex sits at its profile radius from the Y axis.
  for (let p = 0; p < outer.length; p++) {
    const v = p * (segments + 1) + 3;
    const radial = Math.hypot(positions[v * 3], positions[v * 3 + 2]);
    assert.ok(Math.abs(radial - outer[p].radius) < 1e-6);
  }
});

test('revolution surface normals point outwards on the bulb', () => {
  const { outer } = createProfile();
  const { positions, normals } = createRevolutionSurface(outer, 8);
  // Find the vertex nearest y = -0.3 on the first meridian (angle 0, so +X is outwards).
  let best = 0;
  outer.forEach((p, i) => { if (Math.abs(p.height + 0.3) < Math.abs(outer[best].height + 0.3)) best = i; });
  const v = best * 9;
  assert.ok(positions[v * 3] > 0.2);
  assert.ok(normals[v * 3] > 0.9, 'radial normal component should be positive');
});
