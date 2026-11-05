import { BACKGROUND, LINES, MESH, DYE, PARTICLES } from './shaders.js';

export class Renderer {
  constructor(device, context, format) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.uniform = device.createBuffer({ size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.meshCount = 0;
    this.lineCount = 0;

    const mod = (code, label) => device.createShaderModule({ code, label });
    const depthFull = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' };
    const additive = { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } };
    const over = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    };

    let m = mod(BACKGROUND, 'bg');
    this.bgPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: m, entryPoint: 'vs' },
      fragment: { module: m, entryPoint: 'fs', targets: [{ format }] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'always' },
    });

    m = mod(LINES, 'lines');
    this.linePipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: m, entryPoint: 'vs', buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }] },
      fragment: { module: m, entryPoint: 'fs', targets: [{ format, blend: additive }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less' },
    });

    m = mod(MESH, 'mesh');
    this.meshPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: m, entryPoint: 'vs',
        buffers: [{ arrayStride: 24, attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
        ] }],
      },
      fragment: { module: m, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: depthFull,
    });

    m = mod(DYE, 'dye');
    this.dyePipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: m, entryPoint: 'vs' },
      fragment: { module: m, entryPoint: 'fs', targets: [{ format, blend: over }] },
    });

    m = mod(PARTICLES, 'particles');
    this.partPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: m, entryPoint: 'vs' },
      fragment: { module: m, entryPoint: 'fs', targets: [{ format, blend: additive }] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less' },
    });

    const ub = (pipe) => device.createBindGroup({
      layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }],
    });
    this.bgBG = ub(this.bgPipe);
    this.lineBG = ub(this.linePipe);
    this.meshBG = ub(this.meshPipe);
    this.dyeBGs = new Map();
  }

  resize(w, h) {
    if (this.depth && this.w === w && this.h === h) return;
    this.w = w; this.h = h;
    if (this.depth) this.depth.destroy();
    this.depth = this.device.createTexture({
      size: [w, h], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthView = this.depth.createView();
    this.dyeBGs.clear();
  }

  setDomain(nx, ny, nz) {
    this.dims = [nx, ny, nz];
    const v = [];
    const L = (a, b, alpha) => v.push(...a, alpha, ...b, alpha);
    // tunnel box edges
    const c = [[0, 0, 0], [nx, 0, 0], [nx, ny, 0], [0, ny, 0], [0, 0, nz], [nx, 0, nz], [nx, ny, nz], [0, ny, nz]];
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    for (const [a, b] of edges) L(c[a], c[b], 0.35);
    // floor grid
    const step = Math.max(4, Math.round(ny / 12));
    for (let x = 0; x <= nx; x += step) L([x, 0, 0], [x, 0, nz], 0.09);
    for (let z = 0; z <= nz; z += step) L([0, 0, z], [nx, 0, z], 0.09);
    // inlet frame highlight
    L([0, 0, 0], [0, ny, 0], 0.6); L([0, 0, nz], [0, ny, nz], 0.6);
    const data = new Float32Array(v);
    if (this.lineBuf) this.lineBuf.destroy();
    this.lineBuf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.lineBuf, 0, data);
    this.lineCount = data.length / 4;
  }

  setMesh(tris) {
    const T = tris.length / 9;
    if (!T) { this.meshCount = 0; return; }
    const out = new Float32Array(T * 18);
    for (let t = 0; t < T; t++) {
      const o = t * 9;
      const ax = tris[o], ay = tris[o + 1], az = tris[o + 2];
      const ux = tris[o + 3] - ax, uy = tris[o + 4] - ay, uz = tris[o + 5] - az;
      const vx = tris[o + 6] - ax, vy = tris[o + 7] - ay, vz = tris[o + 8] - az;
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
      for (let k = 0; k < 3; k++) {
        const d = t * 18 + k * 6;
        out[d] = tris[o + k * 3]; out[d + 1] = tris[o + k * 3 + 1]; out[d + 2] = tris[o + k * 3 + 2];
        out[d + 3] = nx; out[d + 4] = ny; out[d + 5] = nz;
      }
    }
    if (this.meshBuf) this.meshBuf.destroy();
    this.meshBuf = this.device.createBuffer({ size: out.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.meshBuf, 0, out);
    this.meshCount = T * 3;
  }

  dyeBindGroup(view) {
    let bg = this.dyeBGs.get(view);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.dyePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: view },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: this.depthView },
        ],
      });
      this.dyeBGs.set(view, bg);
    }
    return bg;
  }

  particleBindGroup(buffer) {
    if (this._pBuf !== buffer) {
      this._pBuf = buffer;
      this._pBG = this.device.createBindGroup({
        layout: this.partPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer } }],
      });
    }
    return this._pBG;
  }

  encode(encoder, cam, opts) {
    const { w, h } = this;
    const b = new Float32Array(44);
    b.set(cam.viewProj, 0); b.set(cam.invViewProj, 16);
    b.set(cam.eye, 32); b[35] = opts.time;
    b.set(this.dims, 36); b[39] = opts.dyeDensity;
    b[40] = w; b[41] = h; b[42] = opts.streak; b[43] = Math.max(1.2, h / 900 * 1.6);
    this.device.queue.writeBuffer(this.uniform, 0, b);

    const colorView = this.context.getCurrentTexture().createView();
    let pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      depthStencilAttachment: { view: this.depthView, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    pass.setPipeline(this.bgPipe); pass.setBindGroup(0, this.bgBG); pass.draw(3);
    if (opts.showObject && this.meshCount) {
      pass.setPipeline(this.meshPipe); pass.setBindGroup(0, this.meshBG);
      pass.setVertexBuffer(0, this.meshBuf); pass.draw(this.meshCount);
    }
    pass.setPipeline(this.linePipe); pass.setBindGroup(0, this.lineBG);
    pass.setVertexBuffer(0, this.lineBuf); pass.draw(this.lineCount);
    pass.end();

    if (opts.dyeView) {
      pass = encoder.beginRenderPass({ colorAttachments: [{ view: colorView, loadOp: 'load', storeOp: 'store' }] });
      pass.setPipeline(this.dyePipe); pass.setBindGroup(0, this.dyeBindGroup(opts.dyeView)); pass.draw(3);
      pass.end();
    }

    if (opts.particles) {
      pass = encoder.beginRenderPass({
        colorAttachments: [{ view: colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: this.depthView, depthReadOnly: true },
      });
      pass.setPipeline(this.partPipe); pass.setBindGroup(0, this.particleBindGroup(opts.particles.buffer));
      pass.draw(6, opts.particles.count);
      pass.end();
    }
  }
}
