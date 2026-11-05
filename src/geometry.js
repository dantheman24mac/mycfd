// Geometry: preset meshes, STL parsing, transforms and voxelisation.
// Lattice axes: x = flow direction, y = up, z = lateral.
// A "shape" is an array of closed parts; each part is a Float32Array of triangles (9 floats each).

function pushTri(arr, a, b, c) { arr.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]); }

// Grid-of-rings surface helper: rings[i][j] = point. Closes around j, caps ends to their centroids.
function loft(rings, capStart = true, capEnd = true) {
  const t = [];
  const R = rings.length, S = rings[0].length;
  for (let i = 0; i < R - 1; i++) {
    for (let j = 0; j < S; j++) {
      const j2 = (j + 1) % S;
      pushTri(t, rings[i][j], rings[i + 1][j], rings[i + 1][j2]);
      pushTri(t, rings[i][j], rings[i + 1][j2], rings[i][j2]);
    }
  }
  const centroid = (ring) => ring.reduce((s, p) => [s[0] + p[0] / S, s[1] + p[1] / S, s[2] + p[2] / S], [0, 0, 0]);
  if (capStart) { const c = centroid(rings[0]); for (let j = 0; j < S; j++) pushTri(t, c, rings[0][(j + 1) % S], rings[0][j]); }
  if (capEnd) { const c = centroid(rings[R - 1]); for (let j = 0; j < S; j++) pushTri(t, c, rings[R - 1][j], rings[R - 1][(j + 1) % S]); }
  return new Float32Array(t);
}

function sphere(r = 0.5, seg = 48) {
  const rings = [];
  for (let i = 1; i < seg; i++) {
    const th = (i / seg) * Math.PI, ring = [];
    for (let j = 0; j < seg * 2; j++) {
      const ph = (j / (seg * 2)) * Math.PI * 2;
      ring.push([r * Math.cos(th), r * Math.sin(th) * Math.cos(ph), r * Math.sin(th) * Math.sin(ph)]);
    }
    rings.push(ring);
  }
  // caps converge to poles
  return loft(rings);
}

// Cylinder along `axis` ('y' or 'z'), centred at c.
function cylinder(r, h, axis = 'y', c = [0, 0, 0], seg = 48) {
  const rings = [];
  for (const s of [-h / 2, h / 2]) {
    const ring = [];
    for (let j = 0; j < seg; j++) {
      const a = (j / seg) * Math.PI * 2, u = r * Math.cos(a), v = r * Math.sin(a);
      ring.push(axis === 'y' ? [c[0] + u, c[1] + s, c[2] + v] : [c[0] + u, c[1] + v, c[2] + s]);
    }
    rings.push(ring);
  }
  return loft(rings);
}

function box(sx, sy, sz) {
  const rings = [];
  for (const x of [-sx / 2, sx / 2]) rings.push([[x, -sy / 2, -sz / 2], [x, sy / 2, -sz / 2], [x, sy / 2, sz / 2], [x, -sy / 2, sz / 2]]);
  return loft(rings);
}

// NACA 0012 wing, chord along x, span along z.
function wing(chord = 1, span = 1.6, seg = 80) {
  const t = 0.12, rings = [];
  const profile = [];
  for (let i = 0; i < seg; i++) {
    const a = (i / seg) * Math.PI * 2;
    const xc = 0.5 * (1 + Math.cos(a)); // 1 -> 0 -> 1
    const yt = 5 * t * (0.2969 * Math.sqrt(xc) - 0.126 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1036 * xc ** 4);
    profile.push([(xc - 0.35) * chord, (a < Math.PI ? 1 : -1) * yt * chord]);
  }
  for (const z of [-span / 2, span / 2]) rings.push(profile.map(([x, y]) => [x, y, z]));
  return loft(rings);
}

// Stylised car: lofted superellipse body + four wheels. Length 1 along x.
function car() {
  const N = 72, S = 40;
  const lerp = (a, b, t) => a + (b - a) * t;
  // side profile (x in [0,1]) -> [bottom, top], half width
  const top = (x) => {
    if (x < 0.08) return lerp(0.13, 0.20, Math.sqrt(x / 0.08));
    if (x < 0.30) return lerp(0.20, 0.235, (x - 0.08) / 0.22);
    if (x < 0.48) return lerp(0.235, 0.36, Math.sin(((x - 0.30) / 0.18) * Math.PI / 2));
    if (x < 0.72) return lerp(0.36, 0.345, (x - 0.48) / 0.24);
    if (x < 0.94) return lerp(0.345, 0.25, Math.pow((x - 0.72) / 0.22, 1.4));
    return lerp(0.25, 0.20, (x - 0.94) / 0.06);
  };
  const bottom = (x) => 0.055 + 0.02 * Math.pow(Math.abs(x - 0.5) * 2, 3);
  const halfW = (x) => 0.21 * Math.pow(Math.sin(Math.PI * Math.min(1, Math.max(0, x * 0.94 + 0.03))), 0.25);
  const rings = [];
  for (let i = 0; i <= N; i++) {
    const x = 0.002 + 0.996 * (0.5 - 0.5 * Math.cos((i / N) * Math.PI));
    const b = bottom(x), tp = top(x), w = halfW(x);
    const cy = (b + tp) / 2, hy = (tp - b) / 2;
    const ring = [];
    for (let j = 0; j < S; j++) {
      const a = (j / S) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      const e = 0.35; // superellipse exponent → boxy but rounded
      const py = cy + hy * Math.sign(sa) * Math.pow(Math.abs(sa), e);
      // greenhouse narrows the upper half
      const taper = py > 0.24 ? lerp(1, 0.78, Math.min(1, (py - 0.24) / 0.12)) : 1;
      const pz = w * taper * Math.sign(ca) * Math.pow(Math.abs(ca), e);
      ring.push([x - 0.5, py, pz]);
    }
    rings.push(ring);
  }
  const parts = [loft(rings)];
  const wr = 0.075, ww = 0.06;
  for (const wx of [0.19, 0.79]) for (const wz of [-0.175, 0.175]) {
    parts.push(cylinder(wr, ww, 'z', [wx - 0.5, wr, wz], 32));
  }
  return parts;
}

export function presetShape(name) {
  switch (name) {
    case 'sphere': return { parts: [sphere(0.5)], ground: false };
    case 'cylinder': return { parts: [cylinder(0.25, 1.4, 'z')], ground: false };
    case 'cube': return { parts: [box(0.7, 0.7, 0.7)], ground: false };
    case 'wing': return { parts: [wing(1, 1.6)], ground: false };
    case 'car': return { parts: car(), ground: true };
    default: return { parts: [], ground: false };
  }
}

// Binary + ASCII STL. Returns Float32Array of triangles. STL files are usually Z-up; convert to Y-up.
export function parseSTL(buffer) {
  const dv = new DataView(buffer);
  let tris;
  const isBinary = buffer.byteLength >= 84 && 84 + dv.getUint32(80, true) * 50 === buffer.byteLength;
  if (isBinary) {
    const count = dv.getUint32(80, true);
    tris = new Float32Array(count * 9);
    for (let i = 0; i < count; i++) {
      const o = 84 + i * 50 + 12;
      for (let k = 0; k < 9; k++) tris[i * 9 + k] = dv.getFloat32(o + k * 4, true);
    }
  } else {
    const text = new TextDecoder().decode(buffer);
    const nums = [];
    const re = /vertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)/g;
    let m;
    while ((m = re.exec(text))) nums.push(+m[1], +m[2], +m[3]);
    tris = new Float32Array(nums);
  }
  if (!tris.length) throw new Error('No triangles found in STL file.');
  // Z-up -> Y-up: (x, y, z) -> (x, z, -y)
  for (let i = 0; i < tris.length; i += 3) {
    const y = tris[i + 1], z = tris[i + 2];
    tris[i + 1] = z; tris[i + 2] = -y;
  }
  // centre and normalise so that the largest extent is 1
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < tris.length; i++) { const a = i % 3; mn[a] = Math.min(mn[a], tris[i]); mx[a] = Math.max(mx[a], tris[i]); }
  const ext = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) || 1;
  for (let i = 0; i < tris.length; i++) {
    const a = i % 3;
    tris[i] = a === 1 ? (tris[i] - mn[1]) / ext : (tris[i] - (mn[a] + mx[a]) / 2) / ext;
  }
  return tris;
}

// Transform a shape into lattice coordinates. Returns { parts, tris (merged), bounds }.
export function placeShape(shape, grid, { size, yawDeg, pitchDeg }) {
  const { nx, ny, nz } = grid;
  const yaw = (yawDeg * Math.PI) / 180, pitch = (pitchDeg * Math.PI) / 180;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  // object extent before rotation
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const p of shape.parts) for (let i = 0; i < p.length; i++) {
    const a = i % 3; mn[a] = Math.min(mn[a], p[i]); mx[a] = Math.max(mx[a], p[i]);
  }
  const ext = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) || 1;
  const scale = (size * Math.min(nx * 0.6, ny * 1.4)) / ext;
  const pivot = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
  const centre = [nx * 0.3, shape.ground ? 0 : ny / 2, nz / 2];

  const out = shape.parts.map((p) => {
    const q = new Float32Array(p.length);
    for (let i = 0; i < p.length; i += 3) {
      let x = p[i] - pivot[0], y = p[i + 1] - pivot[1], z = p[i + 2] - pivot[2];
      // pitch (about z, nose up for positive angle-of-attack)
      let x1 = x * cp + y * sp, y1 = -x * sp + y * cp;
      // yaw (about y)
      let x2 = x1 * cy + z * sy, z2 = -x1 * sy + z * cy;
      q[i] = x2 * scale + centre[0];
      q[i + 1] = y1 * scale + (shape.ground ? (pivot[1] - mn[1]) * scale + 1.5 : centre[1]);
      q[i + 2] = z2 * scale + centre[2];
    }
    return q;
  });
  let len = 0; for (const p of out) len += p.length;
  const tris = new Float32Array(len);
  let o = 0; for (const p of out) { tris.set(p, o); o += p.length; }
  return { parts: out, tris, scale };
}

// Parity-fill voxelisation along x for each (y,z) cell centre. Parts are OR-ed.
export function voxelize(parts, grid) {
  const { nx, ny, nz } = grid;
  const mask = new Uint32Array(nx * ny * nz);
  let solidCount = 0;
  for (const tris of parts) {
    const T = tris.length / 9;
    // bucket triangles by their y,z bounding box rows to keep this fast
    const buckets = Array.from({ length: ny }, () => []);
    for (let t = 0; t < T; t++) {
      const o = t * 9;
      const y0 = Math.min(tris[o + 1], tris[o + 4], tris[o + 7]), y1 = Math.max(tris[o + 1], tris[o + 4], tris[o + 7]);
      for (let y = Math.max(0, Math.floor(y0 - 0.5)); y <= Math.min(ny - 1, Math.ceil(y1 - 0.5)); y++) buckets[y].push(t);
    }
    const hits = [];
    for (let y = 0; y < ny; y++) {
      const py = y + 0.5 + 1e-4;
      const list = buckets[y];
      if (!list.length) continue;
      for (let z = 0; z < nz; z++) {
        const pz = z + 0.5 + 1.3e-4;
        hits.length = 0;
        for (const t of list) {
          const o = t * 9;
          const ay = tris[o + 1], az = tris[o + 2], by = tris[o + 4], bz = tris[o + 5], cy = tris[o + 7], cz = tris[o + 8];
          // 2D barycentric test in the y-z plane
          const d = (by - ay) * (cz - az) - (cy - ay) * (bz - az);
          if (Math.abs(d) < 1e-12) continue;
          const u = ((py - ay) * (cz - az) - (cy - ay) * (pz - az)) / d;
          const v = ((by - ay) * (pz - az) - (py - ay) * (bz - az)) / d;
          if (u < 0 || v < 0 || u + v > 1) continue;
          hits.push(tris[o] + u * (tris[o + 3] - tris[o]) + v * (tris[o + 6] - tris[o]));
        }
        if (hits.length < 2) continue;
        hits.sort((a, b) => a - b);
        for (let h = 0; h + 1 < hits.length; h += 2) {
          const x0 = Math.max(0, Math.ceil(hits[h] - 0.5)), x1 = Math.min(nx - 1, Math.floor(hits[h + 1] - 0.5));
          const row = (z * ny + y) * nx;
          for (let x = x0; x <= x1; x++) { if (!mask[row + x]) solidCount++; mask[row + x] = 1; }
        }
      }
    }
  }
  return { mask, solidCount };
}
