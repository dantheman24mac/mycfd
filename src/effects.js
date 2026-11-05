// Particle tracers and volumetric dye advection, both driven by the LBM velocity texture.

const PARTICLE_WGSL = /* wgsl */ `
struct PP {
  dims: vec3<f32>, dt: f32,
  frame: u32, mode: u32, count: u32, maxAge: f32,
  colorMode: u32, uin: f32, pad0: f32, pad1: f32,
};
@group(0) @binding(0) var<uniform> P: PP;
@group(0) @binding(1) var<storage, read_write> parts: array<vec4<f32>>;
@group(0) @binding(2) var velTex: texture_3d<f32>;
@group(0) @binding(3) var samp: sampler;

fn hash(x: u32) -> u32 {
  var v = x * 747796405u + 2891336453u;
  let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(s: ptr<function, u32>) -> f32 { *s = hash(*s); return f32(*s) / 4294967295.0; }

fn vel(p: vec3<f32>) -> vec4<f32> { return textureSampleLevel(velTex, samp, p / P.dims, 0.0); }

// Each particle has a fixed "lane" (seed height) so streaks stay coherent.
fn lane(i: u32) -> vec2<f32> {
  let G = 40u;
  let L = hash(i * 9781u + 17u) % (G * G);
  return vec2<f32>((f32(L % G) + 0.5) / f32(G), (f32(L / G) + 0.5) / f32(G));
}

fn spawn(i: u32, s: ptr<function, u32>, anywhere: bool) -> vec3<f32> {
  let l = lane(i);
  let ry = (0.03 + 0.80 * l.y) * P.dims.y;
  let rz = (0.22 + 0.56 * l.x) * P.dims.z;
  if (P.mode == 1u && !anywhere) {
    let jy = (rnd(s) - 0.5) * 0.3; let jz = (rnd(s) - 0.5) * 0.3;
    return vec3<f32>(rnd(s) * 1.5 + 0.5, ry + jy, rz + jz);
  }
  if (P.mode == 1u) {
    return vec3<f32>(rnd(s) * (P.dims.x - 1.0) + 0.5, ry, rz);
  }
  return vec3<f32>(rnd(s) * (P.dims.x - 1.0) + 0.5, rnd(s) * P.dims.y, rnd(s) * P.dims.z);
}

@compute @workgroup_size(256)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) { return; }
  var s = hash(i ^ (P.frame * 1664525u + 1013904223u));
  var a = parts[2u * i];
  var p = a.xyz;
  var age = a.w + P.dt;

  // RK2 (midpoint) integration in lattice units
  let v1 = vel(p);
  let v2 = vel(p + 0.5 * P.dt * v1.xyz);
  p = p + P.dt * v2.xyz;
  let v = vel(p);

  let out = p.x < 0.0 || p.x >= P.dims.x - 1.0 || p.y < 0.0 || p.y >= P.dims.y || p.z < 0.0 || p.z >= P.dims.z;
  if (out || v.w > 0.5 || age > P.maxAge * (0.6 + 0.4 * f32(hash(i) % 1000u) / 1000.0) || P.frame == 0u) {
    let first = P.frame == 0u;
    p = spawn(i, &s, first);
    // stagger ages on the initial fill so particles don't all expire together
    age = select(0.0, rnd(&s) * P.maxAge * 0.6, first);
  }

  var c = 0.0;
  if (P.colorMode == 0u) {
    c = length(v.xyz) / max(P.uin, 1e-3) * 0.6;
  } else if (P.colorMode == 1u) {
    let e = vec3<f32>(1.0, 0.0, 0.0);
    let dx = vel(p + e.xyy).xyz - vel(p - e.xyy).xyz;
    let dy = vel(p + e.yxy).xyz - vel(p - e.yxy).xyz;
    let dz = vel(p + e.yyx).xyz - vel(p - e.yyx).xyz;
    let w = vec3<f32>(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) * 0.5;
    c = clamp(length(w) / max(P.uin, 1e-3) * 4.0, 0.0, 1.2);
  } else {
    c = lane(i).y;
  }
  parts[2u * i] = vec4<f32>(p, age);
  parts[2u * i + 1u] = vec4<f32>(v.xyz, c);
}
`;

const DYE_WGSL = /* wgsl */ `
struct DP {
  dims: vec3<f32>, dt: f32,
  decay: f32, radius: f32, rows: f32, cols: f32,
};
@group(0) @binding(0) var<uniform> P: DP;
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var dyeIn: texture_3d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var dyeOut: texture_storage_3d<rgba16float, write>;

fn palette(t: f32) -> vec3<f32> {
  // cyan -> violet -> magenta -> amber
  let a = vec3<f32>(0.25, 0.85, 1.0);
  let b = vec3<f32>(0.55, 0.35, 1.0);
  let c = vec3<f32>(1.0, 0.3, 0.7);
  let d = vec3<f32>(1.0, 0.7, 0.25);
  if (t < 0.33) { return mix(a, b, t / 0.33); }
  if (t < 0.66) { return mix(b, c, (t - 0.33) / 0.33); }
  return mix(c, d, (t - 0.66) / 0.34);
}

@compute @workgroup_size(4, 4, 4)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dimsU = vec3<u32>(P.dims);
  if (any(gid >= dimsU)) { return; }
  let p = vec3<f32>(gid) + 0.5;
  let v = textureSampleLevel(velTex, samp, p / P.dims, 0.0);
  if (v.w > 0.5) { textureStore(dyeOut, gid, vec4<f32>(0.0)); return; }
  let mid = p - 0.5 * P.dt * v.xyz;
  let v2 = textureSampleLevel(velTex, samp, mid / P.dims, 0.0).xyz;
  let back = p - P.dt * v2;
  var d = textureSampleLevel(dyeIn, samp, back / P.dims, 0.0) * P.decay;

  // smoke nozzles on the inlet plane (rake concentrated around the centre plane)
  if (p.x < 3.0) {
    let y0 = 0.04; let y1 = 0.80; let z0 = 0.34; let z1 = 0.66;
    let fy = (p.y / P.dims.y - y0) / (y1 - y0) * P.rows;
    let fz = (p.z / P.dims.z - z0) / (z1 - z0) * P.cols;
    let cy = clamp(floor(fy), 0.0, P.rows - 1.0) + 0.5;
    let cz = clamp(floor(fz), 0.0, P.cols - 1.0) + 0.5;
    let ny = (cy / P.rows * (y1 - y0) + y0) * P.dims.y;
    let nz = (cz / P.cols * (z1 - z0) + z0) * P.dims.z;
    let r = length(vec2<f32>(p.y - ny, p.z - nz));
    if (r < P.radius) {
      let k = 1.0 - smoothstep(P.radius * 0.4, P.radius, r);
      d = max(d, vec4<f32>(palette(cy / P.rows) * k, k));
    }
  }
  textureStore(dyeOut, gid, d);
}
`;

export class Effects {
  constructor(device, solver, particleCount) {
    this.device = device;
    this.solver = solver;
    const { nx, ny, nz } = solver;
    this.dims = [nx, ny, nz];
    this.frame = 0;
    this.mode = 1; this.colorMode = 0;

    this.sampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge',
    });

    // particles
    this.pParams = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pModule = device.createShaderModule({ code: PARTICLE_WGSL, label: 'particles' });
    this.pPipe = device.createComputePipeline({ layout: 'auto', compute: { module: pModule, entryPoint: 'advect' } });
    this.setParticleCount(particleCount);

    // dye (ping-pong 3D textures)
    const mkTex = () => device.createTexture({
      size: [nx, ny, nz], dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.dye = [mkTex(), mkTex()];
    this.dyeViews = this.dye.map((t) => t.createView({ dimension: '3d' }));
    this.dParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const dModule = device.createShaderModule({ code: DYE_WGSL, label: 'dye' });
    this.dPipe = device.createComputePipeline({ layout: 'auto', compute: { module: dModule, entryPoint: 'advect' } });
    this.dBG = [0, 1].map((i) => device.createBindGroup({
      layout: this.dPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.dParams } },
        { binding: 1, resource: solver.velView },
        { binding: 2, resource: this.dyeViews[i] },
        { binding: 3, resource: this.sampler },
        { binding: 4, resource: this.dyeViews[1 - i] },
      ],
    }));
    this.dyeIndex = 0; // texture holding the latest dye
  }

  setParticleCount(count) {
    if (this.particles) this.particles.destroy();
    this.count = count;
    this.particles = this.device.createBuffer({
      size: count * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    });
    this.pBG = this.device.createBindGroup({
      layout: this.pPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.pParams } },
        { binding: 1, resource: { buffer: this.particles } },
        { binding: 2, resource: this.solver.velView },
        { binding: 3, resource: this.sampler },
      ],
    });
    this.frame = 0;
  }

  respawn() { this.frame = 0; }

  get dyeView() { return this.dyeViews[this.dyeIndex]; }

  encode(encoder, dt, { particles, dye }) {
    const [nx, ny, nz] = this.dims;
    const uin = this.solver.uin;
    if (particles) {
      const b = new ArrayBuffer(48), f = new Float32Array(b), u = new Uint32Array(b);
      f[0] = nx; f[1] = ny; f[2] = nz; f[3] = dt;
      u[4] = this.frame; u[5] = this.mode; u[6] = this.count; f[7] = (nx * 1.6) / Math.max(uin, 0.02);
      u[8] = this.colorMode; f[9] = uin;
      this.device.queue.writeBuffer(this.pParams, 0, b);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pPipe);
      pass.setBindGroup(0, this.pBG);
      pass.dispatchWorkgroups(Math.ceil(this.count / 256));
      pass.end();
      this.frame++;
    }
    if (dye) {
      const radius = Math.max(1.1, ny * 0.016);
      this.device.queue.writeBuffer(this.dParams, 0, new Float32Array([nx, ny, nz, dt, 0.9985, radius, 11, 3]));
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.dPipe);
      pass.setBindGroup(0, this.dBG[this.dyeIndex]);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
      pass.end();
      this.dyeIndex = 1 - this.dyeIndex;
    }
  }

  destroy() {
    this.particles.destroy(); this.dye.forEach((t) => t.destroy());
    this.pParams.destroy(); this.dParams.destroy();
  }
}
