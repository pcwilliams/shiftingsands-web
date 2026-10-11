// Hourglass glass profile: a port of SandGeometry.swift.
//
// The glass is a "surface of revolution": a 2D curve (radius against height)
// spun around the Y axis. Eleven control points are smoothed with Catmull-Rom
// splines into about 75 profile points. Everything else (the visible mesh, the
// wall collision, the GPU lookup table) is derived from that one curve.
//
// Pure module: no DOM, no three.js, so Node can test it.

export const WALL_THICKNESS = 0.008;
export const CHAMBER_BOTTOM = -0.48;
export const CHAMBER_TOP = 0.48;
/** Extra clearance between a particle and the inner glass (matches the app). */
export const WALL_CLEARANCE = 0.003;
export const PROFILE_TABLE_SIZE = 256;

const DEFAULT_NECK = { neck: 0.04, shoulder: 0.15, shoulderHeight: 0.08 };

function controlPoints({ neck, shoulder, shoulderHeight }) {
  return [
    [0.00, -0.50],
    [0.16, -0.50],
    [0.22, -0.42],
    [0.22, -0.20],
    [shoulder, -shoulderHeight],
    [neck, 0.00],
    [shoulder, shoulderHeight],
    [0.22, 0.20],
    [0.22, 0.42],
    [0.16, 0.50],
    [0.00, 0.50],
  ].map(([radius, height]) => ({ radius, height }));
}

export function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (
    2 * p1 +
    (-p0 + p2) * t +
    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
  );
}

/** Catmull-Rom through the control points. The two end spans stay straight. */
export function interpolateProfile(points, subdivisions = 8) {
  if (points.length < 4) return points.slice();
  const out = [];
  const last = points.length - 1;
  for (let i = 0; i < last; i++) {
    const p0 = points[Math.max(i - 1, 0)];
    const p1 = points[i];
    const p2 = points[Math.min(i + 1, last)];
    const p3 = points[Math.min(i + 2, last)];
    const steps = (i === 0 || i === last - 1) ? 1 : subdivisions;
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      out.push({
        radius: Math.max(catmullRom(p0.radius, p1.radius, p2.radius, p3.radius, t), 0),
        height: catmullRom(p0.height, p1.height, p2.height, p3.height, t),
      });
    }
  }
  out.push(points[last]);
  return out;
}

/**
 * Outer neck radius that lets exactly one particle through at a time.
 * Inner radius = particle radius + 0.002 clearance; outer adds the wall.
 */
export function neckRadiusForParticle(particleRadius) {
  return particleRadius + WALL_THICKNESS + 0.002;
}

/**
 * Build a profile. With no argument you get the default hourglass; with an
 * outer neck radius you get the steep-shouldered funnel the app uses so the
 * sand backs up and flows single-file.
 */
export function createProfile(neckRadius = null) {
  const shape = neckRadius == null
    ? DEFAULT_NECK
    : {
        neck: neckRadius,
        shoulder: Math.max(neckRadius * 1.8, neckRadius + 0.01),
        shoulderHeight: 0.04,
      };
  const outer = interpolateProfile(controlPoints(shape), 8);
  const inner = outer.map((p) => ({
    radius: Math.max(p.radius - WALL_THICKNESS, 0),
    height: p.height,
  }));

  /** Inner glass radius at height y (linear search, like the Swift original). */
  function innerRadiusAt(y) {
    for (let i = 0; i < inner.length - 1; i++) {
      const a = inner[i];
      const b = inner[i + 1];
      if ((a.height <= y && b.height >= y) || (b.height <= y && a.height >= y)) {
        if (Math.abs(b.height - a.height) < 0.0001) return a.radius;
        const t = (y - a.height) / (b.height - a.height);
        return a.radius + t * (b.radius - a.radius);
      }
    }
    return 0.21;
  }

  /**
   * 256 samples of (inner radius - clearance) over y in [-0.5, 0.5].
   * The GPU kernel can't run a linear search per particle cheaply, so it
   * reads this table and interpolates: O(1) instead of O(75).
   */
  function buildLookupTable() {
    const table = new Float32Array(PROFILE_TABLE_SIZE);
    for (let i = 0; i < PROFILE_TABLE_SIZE; i++) {
      const y = -0.5 + i / (PROFILE_TABLE_SIZE - 1);
      table[i] = innerRadiusAt(y) - WALL_CLEARANCE;
    }
    return table;
  }

  return { outer, inner, neckRadius: shape.neck, innerRadiusAt, buildLookupTable };
}

/** The same lookup the WGSL kernel does, so tests can compare the two paths. */
export function sampleLookupTable(table, y) {
  const n = table.length - 1;
  const yNorm = Math.min(Math.max((y + 0.5) * n, 0), n);
  const i0 = Math.floor(yNorm);
  const i1 = Math.min(i0 + 1, n);
  const f = yNorm - i0;
  return table[i0] + (table[i1] - table[i0]) * f;
}

/**
 * Spin a profile around Y into a triangle mesh.
 * Returns plain typed arrays; the scene wraps them in a BufferGeometry.
 */
export function createRevolutionSurface(profile, segments = 64) {
  const ring = segments + 1;
  const count = profile.length * ring;
  const positions = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);
  const indices = new Uint32Array((profile.length - 1) * segments * 6);
  const last = profile.length - 1;

  for (let p = 0; p < profile.length; p++) {
    // Tangent along the curve, by central difference. The outward normal in
    // the (radius, height) plane is the tangent turned 90 degrees.
    const a = profile[Math.max(p - 1, 0)];
    const b = profile[Math.min(p + 1, last)];
    const dr = b.radius - a.radius;
    const dh = b.height - a.height;
    const len = Math.hypot(dr, dh);
    const nRadial = len > 0.0001 ? dh / len : 1;
    const nY = len > 0.0001 ? -dr / len : 0;

    for (let s = 0; s <= segments; s++) {
      const angle = (s / segments) * Math.PI * 2;
      const c = Math.cos(angle);
      const sn = Math.sin(angle);
      const v = p * ring + s;
      positions[v * 3] = profile[p].radius * c;
      positions[v * 3 + 1] = profile[p].height;
      positions[v * 3 + 2] = profile[p].radius * sn;
      normals[v * 3] = nRadial * c;
      normals[v * 3 + 1] = nY;
      normals[v * 3 + 2] = nRadial * sn;
      uvs[v * 2] = s / segments;
      uvs[v * 2 + 1] = p / last;
    }
  }

  let k = 0;
  for (let p = 0; p < last; p++) {
    for (let s = 0; s < segments; s++) {
      const cur = p * ring + s;
      const next = cur + 1;
      const above = cur + ring;
      const aboveNext = above + 1;
      indices[k++] = cur; indices[k++] = above; indices[k++] = next;
      indices[k++] = next; indices[k++] = above; indices[k++] = aboveNext;
    }
  }
  return { positions, normals, uvs, indices };
}
