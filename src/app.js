import { LBMSolver } from './lbm.js';
import { Effects } from './effects.js';
import { Renderer } from './render.js';
import { OrbitCamera } from './camera.js';
import { presetShape, parseSTL, placeShape, voxelize } from './geometry.js';

const $ = (id) => document.getElementById(id);

function fail(msg) {
  const e = $('error');
  e.textContent = msg;
  e.hidden = false;
  console.error(msg);
}

async function main() {
  if (!navigator.gpu) return fail('WebGPU is not available on this system.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU adapter found.');
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
    },
  });
  device.lost.then((info) => fail('GPU device lost: ' + info.message));
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', e.error.message));

  const canvas = $('view');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const renderer = new Renderer(device, context, format);
  const camera = new OrbitCamera(canvas);

  const state = {
    paused: false,
    spf: 4,
    shape: presetShape('car'),
    shapeName: 'Car',
    placed: null,
    solidCount: 0,
  };
  let solver = null, fx = null;

  const maxCells = Math.floor(L.maxStorageBufferBindingSize / (19 * 4));

  function buildSim() {
    const [nx, ny, nz] = $('resolution').value.split('x').map(Number);
    if (nx * ny * nz > maxCells) {
      fail(`This GPU cannot allocate a ${nx}×${ny}×${nz} grid. Pick a lower resolution.`);
      $('resolution').value = '96x40x40';
      return buildSim();
    }
    if (fx) fx.destroy();
    if (solver) solver.destroy();
    solver = new LBMSolver(device, nx, ny, nz);
    solver.uin = +$('wind').value;
    solver.nu = +$('visc').value;
    fx = new Effects(device, solver, +$('particleCount').value);
    fx.mode = $('emitMode').value === 'rake' ? 1 : 0;
    fx.colorMode = { speed: 0, vorticity: 1, seed: 2 }[$('colorMode').value];
    renderer.setDomain(nx, ny, nz);
    camera.scale = nx;
    camera.dist = 1.25;
    camera.target = [nx * 0.42, ny * 0.3, nz / 2];
    rebuildObject(true);
  }

  function rebuildObject(resetFlow = false) {
    const grid = solver;
    const placed = placeShape(state.shape, grid, {
      size: +$('size').value, yawDeg: +$('yaw').value, pitchDeg: +$('pitch').value,
    });
    const { mask, solidCount } = voxelize(placed.parts, grid);
    state.placed = placed;
    state.solidCount = solidCount;
    solver.setSolid(mask);
    renderer.setMesh(placed.tris);
    if (resetFlow) solver.reset();
    updateLabels();
  }

  // debounce heavy object rebuilds while dragging sliders
  let rebuildTimer = 0;
  const scheduleRebuild = () => { clearTimeout(rebuildTimer); rebuildTimer = setTimeout(() => rebuildObject(false), 60); };

  function updateLabels() {
    $('windOut').textContent = (+$('wind').value).toFixed(3);
    $('viscOut').textContent = (+$('visc').value).toFixed(3);
    $('spfOut').textContent = $('spf').value;
    $('sizeOut').textContent = Math.round(+$('size').value * 100) + '%';
    $('yawOut').textContent = $('yaw').value + '°';
    $('pitchOut').textContent = $('pitch').value + '°';
    $('dyeOut').textContent = (+$('dyeDensity').value).toFixed(1);
    if (solver) {
      const Lc = +$('size').value * Math.min(solver.nx * 0.6, solver.ny * 1.4);
      const re = (solver.uin * Lc) / solver.nu;
      $('reynolds').textContent = `Reynolds number ≈ ${Math.round(re).toLocaleString()} (L = ${Lc.toFixed(0)} cells) · Mach ${(solver.uin * Math.sqrt(3)).toFixed(2)}`;
    }
  }

  // ---- UI wiring
  $('btnPause').onclick = () => { state.paused = !state.paused; $('btnPause').textContent = state.paused ? 'Resume' : 'Pause'; };
  $('btnReset').onclick = () => { solver.reset(); fx.respawn(); };
  $('resolution').onchange = buildSim;
  $('wind').oninput = () => { solver.uin = +$('wind').value; solver.writeParams(); updateLabels(); };
  $('visc').oninput = () => { solver.nu = +$('visc').value; solver.writeParams(); updateLabels(); };
  $('spf').oninput = () => { state.spf = +$('spf').value; updateLabels(); };
  for (const id of ['size', 'yaw', 'pitch']) $(id).oninput = () => { updateLabels(); scheduleRebuild(); };
  $('dyeDensity').oninput = updateLabels;
  $('preset').onchange = () => {
    state.shape = presetShape($('preset').value);
    state.shapeName = $('preset').selectedOptions[0].textContent;
    if ($('preset').value === 'wing') { $('size').value = 0.75; }
    rebuildObject(false);
  };
  $('btnImport').onclick = () => $('fileStl').click();
  $('fileStl').onchange = async () => {
    const file = $('fileStl').files[0];
    if (!file) return;
    try {
      const tris = parseSTL(await file.arrayBuffer());
      const ground = confirm('Place the model on the tunnel floor (e.g. vehicles)?\nOK = on floor, Cancel = centred in the stream');
      state.shape = { parts: [tris], ground };
      state.shapeName = `${file.name} (${(tris.length / 9).toLocaleString()} tris)`;
      rebuildObject(false);
    } catch (err) {
      alert('Could not load STL: ' + err.message);
    }
    $('fileStl').value = '';
  };
  $('particleCount').onchange = () => fx.setParticleCount(+$('particleCount').value);
  $('colorMode').onchange = () => { fx.colorMode = { speed: 0, vorticity: 1, seed: 2 }[$('colorMode').value]; };
  $('emitMode').onchange = () => { fx.mode = $('emitMode').value === 'rake' ? 1 : 0; fx.respawn(); };

  // ---- frame loop
  let last = performance.now(), fpsAcc = 0, fpsFrames = 0, fps = 0, stepsAcc = 0, mlups = 0;
  const t0 = performance.now();

  function frame(now) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    renderer.resize(w, h);

    const dtMs = now - last; last = now;
    fpsAcc += dtMs; fpsFrames++;
    if (fpsAcc > 500) {
      fps = (fpsFrames * 1000) / fpsAcc;
      mlups = (stepsAcc * solver.n) / (fpsAcc * 1000);
      fpsAcc = 0; fpsFrames = 0; stepsAcc = 0;
      $('stats').textContent =
        `${solver.nx}×${solver.ny}×${solver.nz} grid · ${(solver.n / 1e6).toFixed(2)}M cells\n` +
        `${fps.toFixed(0)} fps · ${mlups.toFixed(0)} MLUPS · step ${solver.steps.toLocaleString()}\n` +
        `${state.shapeName} · ${state.solidCount.toLocaleString()} solid cells · ${(fx.count / 1000).toFixed(0)}k particles`;
    }

    const enc = device.createCommandEncoder();
    const showParticles = $('showParticles').checked, showDye = $('showDye').checked;
    if (!state.paused) {
      solver.encodeSteps(enc, state.spf);
      stepsAcc += state.spf;
      fx.encode(enc, state.spf, { particles: showParticles, dye: showDye });
    }
    renderer.encode(enc, camera.matrices(w / h), {
      time: (now - t0) / 1000,
      dyeDensity: +$('dyeDensity').value,
      streak: Math.max(4, state.spf * 2.5),
      showObject: $('showObject').checked,
      dyeView: showDye ? fx.dyeView : null,
      particles: showParticles ? { buffer: fx.particles, count: fx.count } : null,
    });
    device.queue.submit([enc.finish()]);
    if (!state.loggedFirst) { state.loggedFirst = true; console.log('first frame submitted', w, h); }
    requestAnimationFrame(frame);
  }

  buildSim();
  updateLabels();
  requestAnimationFrame(frame);
}

main().catch((e) => fail(String(e && e.stack || e)));
