// D3Q19 Lattice-Boltzmann solver (BGK + Smagorinsky LES) running in WebGPU compute.
// Fused pull-stream + collide, ping-pong buffers, SoA layout f[q * N + cell].

const LBM_WGSL = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, n: u32,
  uin: f32, tau: f32, cs2: f32, pad: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> solid: array<u32>;
@group(0) @binding(4) var velTex: texture_storage_3d<rgba16float, write>;

var<private> CX: array<i32, 19> = array<i32, 19>(0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0);
var<private> CY: array<i32, 19> = array<i32, 19>(0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1);
var<private> CZ: array<i32, 19> = array<i32, 19>(0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1);
var<private> OPP: array<u32, 19> = array<u32, 19>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u, 16u, 15u, 18u, 17u);
var<private> W: array<f32, 19> = array<f32, 19>(
  1.0/3.0,
  1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);

fn cvec(q: u32) -> vec3<f32> { return vec3<f32>(f32(CX[q]), f32(CY[q]), f32(CZ[q])); }

fn feq(q: u32, rho: f32, u: vec3<f32>) -> f32 {
  let cu = 3.0 * dot(cvec(q), u);
  return W[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * dot(u, u));
}

@compute @workgroup_size(128)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= P.n) { return; }
  let nx = P.nx; let ny = P.ny; let nz = P.nz; let n = P.n;
  let x = idx % nx;
  let y = (idx / nx) % ny;
  let z = idx / (nx * ny);
  let coord = vec3<i32>(i32(x), i32(y), i32(z));

  if (solid[idx] != 0u) {
    for (var q = 0u; q < 19u; q++) { dst[q * n + idx] = W[q]; }
    textureStore(velTex, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let uin = vec3<f32>(P.uin, 0.0, 0.0);
  var f: array<f32, 19>;
  for (var q = 0u; q < 19u; q++) {
    let sx = i32(x) - CX[q];
    let sy = i32(y) - CY[q];
    let sz = i32(z) - CZ[q];
    if (sx < 0 || sy < 0 || sy >= i32(ny) || sz < 0 || sz >= i32(nz)) {
      // inlet / far-field: free-stream equilibrium
      f[q] = feq(q, 1.0, uin);
    } else {
      let sxc = u32(min(sx, i32(nx) - 1)); // outlet: zero-gradient
      let sidx = sxc + u32(sy) * nx + u32(sz) * nx * ny;
      if (solid[sidx] != 0u) {
        f[q] = src[OPP[q] * n + idx]; // half-way bounce-back
      } else {
        f[q] = src[q * n + sidx];
      }
    }
  }

  var rho = 0.0;
  var m = vec3<f32>(0.0);
  for (var q = 0u; q < 19u; q++) { rho += f[q]; m += f[q] * cvec(q); }
  rho = clamp(rho, 0.5, 2.0);
  var u = m / rho;
  let sp = length(u);
  if (sp > 0.3) { u = u * (0.3 / sp); }

  // Non-equilibrium momentum flux for the Smagorinsky eddy viscosity
  var fe: array<f32, 19>;
  var pxx = 0.0; var pyy = 0.0; var pzz = 0.0; var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
  for (var q = 0u; q < 19u; q++) {
    fe[q] = feq(q, rho, u);
    let d = f[q] - fe[q];
    let c = cvec(q);
    pxx += c.x * c.x * d; pyy += c.y * c.y * d; pzz += c.z * c.z * d;
    pxy += c.x * c.y * d; pxz += c.x * c.z * d; pyz += c.y * c.z * d;
  }
  let Q = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
  let tau = 0.5 * (P.tau + sqrt(P.tau * P.tau + 18.0 * 1.41421356 * P.cs2 * Q / rho));
  let omega = 1.0 / tau;

  for (var q = 0u; q < 19u; q++) {
    dst[q * n + idx] = f[q] + omega * (fe[q] - f[q]);
  }
  textureStore(velTex, coord, vec4<f32>(u, 0.0));
}

@compute @workgroup_size(128)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= P.n) { return; }
  let u = select(vec3<f32>(P.uin, 0.0, 0.0), vec3<f32>(0.0), solid[idx] != 0u);
  for (var q = 0u; q < 19u; q++) {
    let v = feq(q, 1.0, u);
    dst[q * P.n + idx] = v;
  }
}
`;

export class LBMSolver {
  constructor(device, nx, ny, nz) {
    this.device = device;
    this.nx = nx; this.ny = ny; this.nz = nz;
    this.n = nx * ny * nz;
    this.uin = 0.08;
    this.nu = 0.008;
    this.cs = 0.16;
    this.steps = 0;

    const fBytes = this.n * 19 * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.fA = device.createBuffer({ size: fBytes, usage });
    this.fB = device.createBuffer({ size: fBytes, usage });
    this.solid = device.createBuffer({ size: this.n * 4, usage });
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.velTex = device.createTexture({
      size: [nx, ny, nz], dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.velView = this.velTex.createView({ dimension: '3d' });

    const module = device.createShaderModule({ code: LBM_WGSL, label: 'lbm' });
    this.stepPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'step' } });
    this.initPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'init' } });

    const mk = (pipe, a, b) => device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: a } },
        { binding: 2, resource: { buffer: b } },
        { binding: 3, resource: { buffer: this.solid } },
        { binding: 4, resource: this.velView },
      ],
    });
    this.bgAB = mk(this.stepPipe, this.fA, this.fB);
    this.bgBA = mk(this.stepPipe, this.fB, this.fA);
    this.initBG = device.createBindGroup({
      layout: this.initPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 2, resource: { buffer: this.fA } },
        { binding: 3, resource: { buffer: this.solid } },
      ],
    });
    this.flip = false;
    this.groups = Math.ceil(this.n / 128);
    this.writeParams();
  }

  writeParams() {
    const buf = new ArrayBuffer(32);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0] = this.nx; u[1] = this.ny; u[2] = this.nz; u[3] = this.n;
    f[4] = this.uin; f[5] = 3 * this.nu + 0.5; f[6] = this.cs * this.cs;
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  setSolid(mask /* Uint32Array(n) */) {
    this.device.queue.writeBuffer(this.solid, 0, mask);
  }

  reset() {
    this.writeParams();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.initPipe);
    pass.setBindGroup(0, this.initBG);
    pass.dispatchWorkgroups(this.groups);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.flip = false;
    this.steps = 0;
  }

  encodeSteps(encoder, count) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.stepPipe);
    for (let i = 0; i < count; i++) {
      pass.setBindGroup(0, this.flip ? this.bgBA : this.bgAB);
      pass.dispatchWorkgroups(this.groups);
      this.flip = !this.flip;
    }
    pass.end();
    this.steps += count;
  }

  destroy() {
    this.fA.destroy(); this.fB.destroy(); this.solid.destroy();
    this.params.destroy(); this.velTex.destroy();
  }
}
