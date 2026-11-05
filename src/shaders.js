// WGSL shaders for the renderer.

export const COMMON = /* wgsl */ `
struct U {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  eye: vec3<f32>, time: f32,
  dims: vec3<f32>, dyeDensity: f32,
  screen: vec2<f32>, streak: f32, psize: f32,
};
@group(0) @binding(0) var<uniform> U0: U;

fn colormap(t0: f32) -> vec3<f32> {
  let t = clamp(t0, 0.0, 1.0);
  let c0 = vec3<f32>(0.05, 0.10, 0.45);
  let c1 = vec3<f32>(0.10, 0.45, 1.00);
  let c2 = vec3<f32>(0.20, 0.95, 0.95);
  let c3 = vec3<f32>(1.00, 0.95, 0.55);
  let c4 = vec3<f32>(1.00, 0.40, 0.15);
  if (t < 0.25) { return mix(c0, c1, t / 0.25); }
  if (t < 0.5) { return mix(c1, c2, (t - 0.25) / 0.25); }
  if (t < 0.75) { return mix(c2, c3, (t - 0.5) / 0.25); }
  return mix(c3, c4, (t - 0.75) / 0.25);
}

fn rayDir(ndc: vec2<f32>) -> vec3<f32> {
  let a = U0.invViewProj * vec4<f32>(ndc, 0.0, 1.0);
  let b = U0.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  return normalize(b.xyz / b.w - a.xyz / a.w);
}
`;

export const BACKGROUND = COMMON + /* wgsl */ `
struct VO { @builtin(position) pos: vec4<f32>, @location(0) ndc: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2<f32>(f32((i << 1u) & 2u) * 2.0 - 1.0, f32(i & 2u) * 2.0 - 1.0);
  var o: VO; o.pos = vec4<f32>(p, 1.0, 1.0); o.ndc = p; return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
  let d = rayDir(i.ndc);
  let up = d.y * 0.5 + 0.5;
  var c = mix(vec3<f32>(0.010, 0.012, 0.020), vec3<f32>(0.035, 0.050, 0.090), smoothstep(0.2, 0.9, up));
  // soft horizon glow
  c += vec3<f32>(0.04, 0.07, 0.13) * exp(-abs(d.y) * 9.0);
  let v = length(i.ndc * vec2<f32>(0.8, 1.0));
  c *= 1.0 - 0.35 * smoothstep(0.6, 1.5, v);
  return vec4<f32>(c, 1.0);
}
`;

export const LINES = COMMON + /* wgsl */ `
struct VO { @builtin(position) pos: vec4<f32>, @location(0) a: f32, @location(1) w: vec3<f32> };
@vertex fn vs(@location(0) p: vec4<f32>) -> VO {
  var o: VO; o.pos = U0.viewProj * vec4<f32>(p.xyz, 1.0); o.a = p.w; o.w = p.xyz; return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
  let fade = 1.0 - smoothstep(0.0, 1.0, distance(i.w, U0.eye) / (length(U0.dims) * 3.0));
  return vec4<f32>(vec3<f32>(0.35, 0.6, 1.0) * i.a * fade, 1.0);
}
`;

export const MESH = COMMON + /* wgsl */ `
struct VO { @builtin(position) pos: vec4<f32>, @location(0) n: vec3<f32>, @location(1) w: vec3<f32> };
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>) -> VO {
  var o: VO; o.pos = U0.viewProj * vec4<f32>(p, 1.0); o.n = n; o.w = p; return o;
}
@fragment fn fs(i: VO, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
  var n = normalize(i.n);
  if (!ff) { n = -n; }
  let v = normalize(U0.eye - i.w);
  let key = normalize(vec3<f32>(-0.4, 0.8, 0.45));
  let fill = normalize(vec3<f32>(0.6, 0.2, -0.7));
  let base = vec3<f32>(0.62, 0.66, 0.74);
  let diff = max(dot(n, key), 0.0) * 0.85 + max(dot(n, fill), 0.0) * 0.25 + 0.12;
  let h = normalize(key + v);
  let spec = pow(max(dot(n, h), 0.0), 64.0) * 0.9;
  let fres = pow(1.0 - max(dot(n, v), 0.0), 4.0);
  // fake environment reflection: sky above, dark below
  let r = reflect(-v, n);
  let env = mix(vec3<f32>(0.02, 0.03, 0.05), vec3<f32>(0.25, 0.4, 0.7), smoothstep(-0.2, 0.8, r.y));
  var c = base * diff * 0.55 + env * (0.25 + 0.6 * fres) + vec3<f32>(spec);
  c += vec3<f32>(0.3, 0.6, 1.0) * fres * 0.35;
  return vec4<f32>(c, 1.0);
}
`;

export const DYE = COMMON + /* wgsl */ `
@group(0) @binding(1) var dyeTex: texture_3d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var depthTex: texture_depth_2d;

struct VO { @builtin(position) pos: vec4<f32>, @location(0) ndc: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2<f32>(f32((i << 1u) & 2u) * 2.0 - 1.0, f32(i & 2u) * 2.0 - 1.0);
  var o: VO; o.pos = vec4<f32>(p, 0.0, 1.0); o.ndc = p; return o;
}

fn hash2(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453); }

@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
  let ro = U0.eye;
  let rd = rayDir(i.ndc);
  let inv = 1.0 / rd;
  let t0 = (vec3<f32>(0.0) - ro) * inv;
  let t1 = (U0.dims - ro) * inv;
  let tmin = min(t0, t1); let tmax = max(t0, t1);
  let tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  var tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tf <= tn) { discard; }

  // stop at opaque geometry
  let d = textureLoad(depthTex, vec2<i32>(i.pos.xy), 0);
  if (d < 1.0) {
    let wp = U0.invViewProj * vec4<f32>(i.ndc, d, 1.0);
    tf = min(tf, distance(wp.xyz / wp.w, ro));
  }

  let stepLen = 0.75;
  let n = min(i32((tf - tn) / stepLen), 512);
  var t = tn + stepLen * hash2(i.pos.xy + fract(U0.time));
  var acc = vec4<f32>(0.0);
  for (var k = 0; k < n; k++) {
    let p = ro + rd * t;
    let s = textureSampleLevel(dyeTex, samp, p / U0.dims, 0.0);
    if (s.a > 0.003) {
      let dens = s.a * U0.dyeDensity;
      let a = 1.0 - exp(-dens * stepLen * 0.35);
      let col = s.rgb / max(s.a, 1e-3) * (0.6 + 0.8 * s.a);
      acc += vec4<f32>(col * a, a) * (1.0 - acc.a);
      if (acc.a > 0.97) { break; }
    }
    t += stepLen;
  }
  return acc;
}
`;

export const PARTICLES = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> parts: array<vec4<f32>>;

struct VO {
  @builtin(position) pos: vec4<f32>,
  @location(0) col: vec3<f32>,
  @location(1) across: f32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let a = parts[2u * ii];
  let b = parts[2u * ii + 1u];
  let head = a.xyz;
  let tail = a.xyz - b.xyz * U0.streak;
  let ch = U0.viewProj * vec4<f32>(head, 1.0);
  let ct = U0.viewProj * vec4<f32>(tail, 1.0);
  var o: VO;
  if (ch.w <= 0.01 || ct.w <= 0.01) { o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0); return o; }
  let sh = ch.xy / ch.w * U0.screen * 0.5;
  let st = ct.xy / ct.w * U0.screen * 0.5;
  var dir = sh - st;
  let len = length(dir);
  dir = select(vec2<f32>(1.0, 0.0), dir / len, len > 1e-3);
  let perp = vec2<f32>(-dir.y, dir.x);
  let w = U0.psize * clamp(40.0 / ch.w, 0.5, 1.6);

  // quad: 0,1,2 / 2,1,3 with corners (end, side)
  var corner = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0));
  let c = corner[vi];
  let isHead = c.x > 0.5;
  let base = select(ct, ch, isHead);
  let ext = select(-dir, dir, isHead) * w * 0.5;
  let off = (perp * c.y * w + ext) / (U0.screen * 0.5) * base.w;
  o.pos = vec4<f32>(base.xy + off, base.zw);

  let fadeIn = smoothstep(0.0, 30.0, a.w);
  let tailFade = select(0.15, 1.0, isHead);
  o.col = colormap(b.w) * fadeIn * tailFade * 0.14;
  o.across = c.y;
  return o;
}

@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
  let f = 1.0 - i.across * i.across;
  return vec4<f32>(i.col * f * f, 1.0);
}
`;
