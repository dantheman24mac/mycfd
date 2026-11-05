// Minimal column-major mat4 helpers and an orbit camera.

export function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0]);
}

export function lookAt(eye, target, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = (v) => { const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const zA = norm(sub(eye, target)), xA = norm(cross(up, zA)), yA = cross(zA, xA);
  return new Float32Array([
    xA[0], yA[0], zA[0], 0, xA[1], yA[1], zA[1], 0, xA[2], yA[2], zA[2], 0,
    -dot(xA, eye), -dot(yA, eye), -dot(zA, eye), 1,
  ]);
}

export function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

export function invert(m) {
  const inv = new Float32Array(16);
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = 1 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det; inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det; inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det; inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det; inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det; inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det; inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det; inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det; inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return inv;
}

export class OrbitCamera {
  constructor(canvas) {
    this.yaw = -0.65; this.pitch = 0.32; this.dist = 2.2;
    this.target = [0, 0, 0];
    this.scale = 100; // world size (largest grid dimension)
    let mode = null, lx = 0, ly = 0;
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', (e) => {
      mode = e.button === 2 || e.button === 1 ? 'pan' : 'orbit';
      lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointerup', () => { mode = null; });
    canvas.addEventListener('pointermove', (e) => {
      if (!mode) return;
      const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
      if (mode === 'orbit') {
        this.yaw -= dx * 0.006;
        this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + dy * 0.006));
      } else {
        const k = this.dist * this.scale * 0.0012;
        const r = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];
        this.target[0] -= r[0] * dx * k; this.target[2] -= r[2] * dx * k;
        this.target[1] += dy * k;
      }
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.dist = Math.max(0.4, Math.min(6, this.dist * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
  }

  eye() {
    const d = this.dist * this.scale;
    return [
      this.target[0] + d * Math.cos(this.pitch) * Math.sin(this.yaw),
      this.target[1] + d * Math.sin(this.pitch),
      this.target[2] + d * Math.cos(this.pitch) * Math.cos(this.yaw),
    ];
  }

  matrices(aspect) {
    const eye = this.eye();
    const view = lookAt(eye, this.target, [0, 1, 0]);
    const proj = perspective(0.75, aspect, this.scale * 0.02, this.scale * 20);
    const viewProj = mul(proj, view);
    return { eye, view, proj, viewProj, invViewProj: invert(viewProj) };
  }
}
