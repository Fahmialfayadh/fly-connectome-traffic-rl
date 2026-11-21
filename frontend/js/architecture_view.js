/**
 * 3D architecture explorer for FlyConnectomePolicy.
 *
 * Renders the full forward path x_t -> input_projection -> ConnectomeRecurrentCell
 * (frozen FAFB v783 E/I synapses, trainable tau, runtime g_exc/g_inh/lesion_mask)
 * -> actor_head / critic_head -> policy & value, plus the environment loop.
 * Components are hoverable and selectable; details are shown in the inspector.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const ASSET_URL = '/assets/brain/fly_brain_neurons.json';

const COLORS = {
  trainable: 0xffa53d,
  frozenExc: 0x5fb4ff,
  frozenInh: 0x8a7dff,
  runtime: 0xd08cff,
  data: 0xe9edf3,
  neuronCX: 0xdfe9ff,
  neuronMB: 0xb8ead6,
  grad: 0xff6b3d,
  env: 0x8d97a8,
};

const STATUS_LABELS = {
  trainable: 'Trainable',
  frozen: 'Frozen',
  runtime: 'Runtime (tidak dilatih)',
  data: 'Data / State',
};

/** Factual registry of every inspectable component (sourced from src/fly_traffic). */
const COMPONENTS = {
  obs: {
    group: 'Input',
    title: 'Observasi x_t',
    code: 'TrafficIntersectionEnv._build_observation()',
    file: 'src/fly_traffic/environment/traffic_env.py',
    status: 'data',
    shape: '[B, 22]',
    params: '0',
    role: 'Vektor 22 fitur kondisi simpang pada step t. Menjadi satu-satunya masukan eksternal ke policy.',
    training: 'Tidak memiliki parameter. Nilainya dihitung ulang oleh environment pada setiap step.',
    rows: [
      ['obs[0:8]', 'lane.length / 30.0 (antrean 8 lajur)'],
      ['obs[8:16]', 'max_wait_s / 120.0 (waktu tunggu per lajur)'],
      ['obs[16:20]', 'one-hot fase hijau aktif'],
      ['obs[20]', 'is_transitioning (YELLOW / ALL_RED)'],
      ['obs[21]', 'phase_elapsed / 20.0'],
    ],
  },
  input_projection: {
    group: 'Input',
    title: 'input_projection',
    code: 'nn.Linear(22, 2500)',
    file: 'src/fly_traffic/agents/connectome_cell.py',
    status: 'trainable',
    shape: 'W [2500, 22] + b [2500]',
    params: '57.500',
    role: 'Adapter masukan. Memetakan 22 fitur x_t menjadi arus masukan untuk setiap 2.500 neuron (W_in · x_t + b_in).',
    training: 'Diupdate oleh optimizer Adam pada setiap PPO update. Inisialisasi orthogonal (gain 1.0), bias nol.',
    rows: [
      ['weight', '55.000 parameter'],
      ['bias', '2.500 parameter'],
      ['dibuat oleh', 'kode proyek (bukan dari dataset FAFB)'],
    ],
  },
  hidden_state: {
    group: 'Recurrent core',
    title: 'hidden_state h_t',
    code: 'ConnectomeRecurrentCell.forward()',
    file: 'src/fly_traffic/agents/connectome_cell.py',
    status: 'data',
    shape: '[B, 2500]',
    params: '0',
    role: 'Aktivasi 2.500 neuron FAFB v783. Indeks 0..1799 = Central Complex (PB, EB, FB, NO), 1800..2499 = Mushroom Body. Posisi titik mengikuti lokasi neuropil asli.',
    training: 'Bukan parameter. Nilainya berubah setiap step dan direset ke nol saat episode baru dimulai.',
    rows: [
      ['CX', '1.800 neuron'],
      ['MB', '700 neuron'],
      ['update', 'h_t = (1 - tau) h_{t-1} + tau tanh(...)'],
    ],
  },
  w_exc: {
    group: 'Recurrent core',
    title: 'w_exc_csr',
    code: "register_buffer('w_exc_csr', ...)",
    file: 'src/fly_traffic/agents/connectome_cell.py',
    status: 'frozen',
    shape: 'sparse CSR [2500, 2500]',
    params: '36.908 sinapsis (ACH)',
    role: 'Matriks sinapsis eksitatoris dari FAFB v783. Bobot = log1p(jumlah sinapsis), lalu diskalakan bersama w_inh_csr ke spectral radius 0,95.',
    training: 'Tidak diupdate. Disimpan sebagai buffer, bukan nn.Parameter, sehingga tidak masuk optimizer. Gradien tetap mengalir melewatinya (BPTT) menuju input_projection.',
    rows: [
      ['sumber', 'dataset FlyWire FAFB v783'],
      ['ditampilkan', '1.445 edge teratas berdasarkan |W|'],
    ],
  },
  w_inh: {
    group: 'Recurrent core',
    title: 'w_inh_csr',
    code: "register_buffer('w_inh_csr', ...)",
    file: 'src/fly_traffic/agents/connectome_cell.py',
    status: 'frozen',
    shape: 'sparse CSR [2500, 2500]',
    params: '50.263 sinapsis (GABA / GLUT)',
    role: 'Matriks sinapsis inhibitoris dari FAFB v783. Dikurangkan dari arus eksitatoris pada setiap step.',
    training: 'Tidak diupdate. Buffer frozen, sama seperti w_exc_csr. Gradien mengalir melewatinya tanpa mengubah nilainya.',
    rows: [
      ['sumber', 'dataset FlyWire FAFB v783'],
      ['ditampilkan', '1.555 edge teratas berdasarkan |W|'],
    ],
  },
  tau: {
    group: 'Recurrent core',
    title: 'leak_rate (tau)',
    code: 'nn.Parameter(tau_logit)',
    file: 'src/fly_traffic/agents/connectome_cell.py',
    status: 'trainable',
    shape: 'skalar',
    params: '1',
    role: 'Laju integrasi membran. Menentukan porsi state lama (1 - tau) dan masukan baru (tau) di setiap step. tau = 0,05 + 0,90 sigmoid(tau_logit).',
    training: 'Diupdate oleh Adam pada setiap PPO update. Nilai awal 0,35, selalu berada di rentang (0,05; 0,95).',
    rows: [
      ['dipakai bersama', 'oleh seluruh 2.500 neuron'],
    ],
  },
  g_exc: {
    group: 'Recurrent core',
    title: 'g_exc',
    code: 'ConnectomeRecurrentCell.g_exc (float)',
    file: 'src/fly_traffic/agents/connectome_cell.py',
    status: 'runtime',
    shape: 'skalar',
    params: '0',
    role: 'Pengali arus eksitatoris (ACH): g_exc · W_exc · h. Dipakai untuk uji neurofarmakologi in-silico.',
    training: 'Tidak dilatih dan tidak disimpan di checkpoint. Bernilai 1,0 selama training, diubah lewat slider di Live Simulation (0,0 - 2,5).',
    rows: [['default', '1,0']],
  },
  g_inh: {
    group: 'Recurrent core',
    title: 'g_inh',
    code: 'ConnectomeRecurrentCell.g_inh (float)',
    file: 'src/fly_traffic/agents/connectome_cell.py',
    status: 'runtime',
    shape: 'skalar',
    params: '0',
    role: 'Pengali arus inhibitoris (GABA / GLUT): g_inh · W_inh · h.',
    training: 'Tidak dilatih dan tidak disimpan di checkpoint. Bernilai 1,0 selama training, diubah lewat slider di Live Simulation (0,0 - 2,5).',
    rows: [['default', '1,0']],
  },
  lesion_mask: {
    group: 'Recurrent core',
    title: 'lesion_mask',
    code: "register_buffer('lesion_mask', ...)",
    file: 'src/fly_traffic/agents/connectome_cell.py',
    status: 'runtime',
    shape: '[1, 2500]',
    params: '0',
    role: 'Mask biner per neuron. Neuron bernilai 0 dibungkam: state dan keluarannya dikalikan nol.',
    training: 'Tidak dilatih. Bernilai 1 untuk semua neuron selama training; diubah oleh tombol Lesion CX / MB / Random.',
    rows: [['default', 'semua 1 (2.500 aktif)']],
  },
  actor_head: {
    group: 'Readout',
    title: 'actor_head',
    code: 'nn.Linear(2500, 4)',
    file: 'src/fly_traffic/agents/fly_policy.py',
    status: 'trainable',
    shape: 'W [4, 2500] + b [4]',
    params: '10.004',
    role: 'Adapter keluaran aksi. Membaca h_t dan menghasilkan 4 logit fase sinyal.',
    training: 'Diupdate oleh Adam melalui policy loss (clipped surrogate). Inisialisasi orthogonal gain 0,01 sehingga policy awal mendekati uniform.',
    rows: [
      ['weight', '10.000 parameter'],
      ['bias', '4 parameter'],
    ],
  },
  critic_head: {
    group: 'Readout',
    title: 'critic_head',
    code: 'nn.Linear(2500, 1)',
    file: 'src/fly_traffic/agents/fly_policy.py',
    status: 'trainable',
    shape: 'W [1, 2500] + b [1]',
    params: '2.501',
    role: 'Adapter keluaran nilai. Membaca h_t dan mengestimasi V(s_t).',
    training: 'Diupdate oleh Adam melalui value loss terhadap target return GAE. Inisialisasi orthogonal gain 1,0.',
    rows: [
      ['weight', '2.500 parameter'],
      ['bias', '1 parameter'],
    ],
  },
  policy_out: {
    group: 'Readout',
    title: 'pi(a | h_t)',
    code: 'Categorical(logits=actor_head(h_t))',
    file: 'src/fly_traffic/agents/base.py',
    status: 'data',
    shape: '[B, 4]',
    params: '0',
    role: 'Distribusi probabilitas 4 fase: NS Straight, NS Turn, EW Straight, EW Turn.',
    training: 'Saat training aksi di-sample dari distribusi. Di Live Simulation dipilih fase dengan probabilitas tertinggi. Step yang terkunci (YELLOW / ALL_RED / min green) dikeluarkan dari actor loss lewat decision_mask.',
    rows: [['action space', 'Discrete(4)']],
  },
  value_out: {
    group: 'Readout',
    title: 'V(s_t)',
    code: 'critic_head(h_t).squeeze(-1)',
    file: 'src/fly_traffic/agents/fly_policy.py',
    status: 'data',
    shape: '[B]',
    params: '0',
    role: 'Estimasi return dari state saat ini. Dipakai untuk menghitung advantage.',
    training: 'Dipakai di GAE (gamma 0,99, lambda 0,95) untuk menghasilkan advantage dan target return.',
    rows: [],
  },
  env: {
    group: 'Environment',
    title: 'TrafficIntersectionEnv',
    code: 'gymnasium.Env',
    file: 'src/fly_traffic/environment/traffic_env.py',
    status: 'data',
    shape: 'Discrete(4) -> obs [22]',
    params: '0',
    role: 'Simulator simpang empat. Menerima fase dari policy, menjalankan 1 step (dt 2,0 s), lalu mengembalikan x_{t+1} dan reward r_t.',
    training: 'Tidak memiliki parameter. Reward r_t = 0,005 (-Q - 0,1 W - 1,0 S - 15,0 C) menjadi sinyal untuk PPO.',
    rows: [
      ['episode', '300 step'],
      ['training', '8 environment paralel x 128 step per update'],
    ],
  },
};

const GROUP_ORDER = ['Input', 'Recurrent core', 'Readout', 'Environment'];

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeDotTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.35, 'rgba(255,255,255,0.85)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Weight-matrix heatmap drawn on a canvas; cells can be nudged to depict updates. */
class MatrixTexture {
  constructor(cols, rows, seed) {
    this.cols = cols;
    this.rows = rows;
    this.rng = mulberry32(seed);
    this.values = new Float32Array(cols * rows).map(() => this.rng() * 2 - 1);
    this.canvas = document.createElement('canvas');
    this.cellW = Math.max(4, Math.floor(256 / cols));
    this.cellH = Math.max(4, Math.floor(512 / rows));
    this.canvas.width = cols * this.cellW;
    this.canvas.height = rows * this.cellH;
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.magFilter = THREE.NearestFilter;
    this.draw();
  }

  draw() {
    const { ctx, cols, rows, cellW, cellH } = this;
    ctx.fillStyle = '#1a1206';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const v = this.values[r * cols + c];
        const mag = Math.min(1, Math.abs(v));
        const light = v >= 0 ? 38 + mag * 30 : 18 + mag * 10;
        const alpha = 0.25 + mag * 0.75;
        ctx.fillStyle = `hsla(${v >= 0 ? 34 : 22}, 100%, ${light}%, ${alpha})`;
        ctx.fillRect(c * cellW + 0.5, r * cellH + 0.5, cellW - 1, cellH - 1);
      }
    }
    this.texture.needsUpdate = true;
  }

  nudge(fraction) {
    const count = Math.max(1, Math.floor(this.values.length * fraction));
    for (let i = 0; i < count; i += 1) {
      const idx = Math.floor(this.rng() * this.values.length);
      const next = this.values[idx] + (this.rng() - 0.5) * 0.35;
      this.values[idx] = Math.max(-1, Math.min(1, next));
    }
    this.draw();
  }
}

class ReversedCurve {
  constructor(curve) {
    this.curve = curve;
  }

  getPointAt(t, target) {
    return this.curve.getPointAt(1 - t, target);
  }
}

export class ArchitectureView {
  constructor(root) {
    this.root = root;
    this.stage = root.querySelector('[data-arch="stage"]');
    this.canvas = root.querySelector('[data-arch="canvas"]');
    this.labelLayer = root.querySelector('[data-arch="labels"]');
    this.hoverTag = root.querySelector('[data-arch="hover"]');
    this.detailEl = root.querySelector('[data-arch="detail"]');
    this.indexEl = root.querySelector('[data-arch="index"]');
    this.modeCaption = root.querySelector('[data-arch="mode-caption"]');

    this.visible = false;
    this.initialized = false;
    this.initPromise = null;
    this.mode = 'forward';
    this.showLabels = true;
    this.filterStatus = null;
    this.hoveredId = null;
    this.selectedId = null;

    this.components = {};
    this.pickables = [];
    this.labels = [];
    this.particles = [];
    this.matrixTextures = [];
    this.clock = new THREE.Clock();
    this.pointer = new THREE.Vector2(-10, -10);
    this.pointerPx = { x: 0, y: 0 };
    this.pointerInside = false;
    this.cameraTween = null;
    this.lastNudgeMs = 0;

    this._renderIndex();
    this._renderDetail(null);
    this._bindControls();
  }

  setVisible(isVisible) {
    this.visible = Boolean(isVisible);
    if (this.visible && !this.initPromise) {
      this.initPromise = this._init().catch((err) => {
        console.error('Architecture view failed to initialize:', err);
        this.stage.classList.add('is-failed');
      });
    }
    if (this.visible && this.initialized) {
      this._resize();
      this.clock.getDelta();
      this._loop();
    }
  }

  async _init() {
    const response = await fetch(ASSET_URL);
    const layout = await response.json();

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0b0d11, 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x0b0d11, 0.018);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
    this.homeCamera = new THREE.Vector3(9.5, 7.5, 24);
    this.homeTarget = new THREE.Vector3(0, -0.4, 0);
    this.camera.position.copy(this.homeCamera);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 6;
    this.controls.maxDistance = 48;
    this.controls.target.copy(this.homeTarget);
    this.controls.addEventListener('start', () => {
      this.cameraTween = null;
    });

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.62, 0.55, 0.18);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Line.threshold = 0.06;
    this.raycaster.params.Points.threshold = 0.12;
    this.dotTexture = makeDotTexture();

    this._buildFloor();
    this._buildObservation();
    this._buildReservoir(layout);
    this._buildInputProjection();
    this._buildTau();
    this._buildRuntimeControls();
    this._buildReadout();
    this._buildEnvironmentLoop();
    this._buildParticles();
    this._buildLabels();
    this._applyEmphasis();

    this.canvas.addEventListener('pointermove', (e) => this._onPointerMove(e));
    this.canvas.addEventListener('pointerleave', () => {
      this.pointerInside = false;
      this._setHovered(null);
    });
    this.canvas.addEventListener('click', () => {
      if (this.hoveredId) {
        this.select(this.hoveredId, { focus: true });
      }
    });

    new ResizeObserver(() => this._resize()).observe(this.stage);
    this.initialized = true;
    this.stage.classList.add('is-ready');
    this._resize();
    if (this.visible) {
      this._loop();
    }
  }

  /* ------------------------------------------------------------------ */
  /* Scene construction                                                  */
  /* ------------------------------------------------------------------ */

  _register(id, objects, center, focusDist = 10) {
    const entry = this.components[id] || { id, objects: [], materials: [], center, focusDist };
    for (const obj of objects) {
      obj.userData.componentId = id;
      entry.objects.push(obj);
      this.pickables.push(obj);
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of mats) {
        if (!mat) continue;
        mat.transparent = true;
        if (mat.userData.baseOpacity === undefined) {
          mat.userData.baseOpacity = mat.opacity;
        }
        entry.materials.push(mat);
      }
    }
    entry.center = center;
    entry.focusDist = focusDist;
    this.components[id] = entry;
    return entry;
  }

  _buildFloor() {
    const grid = new THREE.GridHelper(60, 60, 0x1c2230, 0x141922);
    grid.position.y = -8.4;
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    this.scene.add(grid);
  }

  _buildObservation() {
    const segments = [8, 8, 4, 1, 1];
    const shades = [0xe9edf3, 0xcfd6e2, 0xb3bfd2, 0x9aa6ba, 0x9aa6ba];
    const cubeGeo = new THREE.BoxGeometry(0.34, 0.26, 0.34);
    const edgeGeo = new THREE.EdgesGeometry(cubeGeo);
    this.obsCubes = [];
    const objects = [];
    let y = 4.6;
    segments.forEach((count, segIdx) => {
      for (let i = 0; i < count; i += 1) {
        const mat = new THREE.MeshBasicMaterial({ color: shades[segIdx], opacity: 0.85 });
        const cube = new THREE.Mesh(cubeGeo, mat);
        cube.position.set(-11.5, y, 0);
        const edge = new THREE.LineSegments(
          edgeGeo,
          new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.55 })
        );
        cube.add(edge);
        this.scene.add(cube);
        this.obsCubes.push(cube);
        objects.push(cube, edge);
        y -= 0.36;
      }
      y -= 0.22;
    });
    this.obsCenter = new THREE.Vector3(-11.5, 0.4, 0);
    this._register('obs', objects, this.obsCenter, 9);
  }

  _buildReservoir(layout) {
    const raw = layout.layout_neuropil;
    const count = layout.num_neurons;
    const cxCount = layout.cx_count;
    const scale = 1.75;

    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (let i = 0; i < count; i += 1) {
      cx += raw[i * 3];
      cy += raw[i * 3 + 1];
      cz += raw[i * 3 + 2];
    }
    cx /= count;
    cy /= count;
    cz /= count;

    this.neuronPositions = [];
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const colCX = new THREE.Color(COLORS.neuronCX);
    const colMB = new THREE.Color(COLORS.neuronMB);
    for (let i = 0; i < count; i += 1) {
      const p = new THREE.Vector3(
        (raw[i * 3] - cx) * scale,
        (raw[i * 3 + 1] - cy) * scale,
        (raw[i * 3 + 2] - cz) * scale
      );
      this.neuronPositions.push(p);
      positions.set([p.x, p.y, p.z], i * 3);
      const c = i < cxCount ? colCX : colMB;
      colors.set([c.r, c.g, c.b], i * 3);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.13,
      map: this.dotTexture,
      vertexColors: true,
      opacity: 0.95,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    this.neuronPoints = new THREE.Points(geo, mat);
    this.scene.add(this.neuronPoints);
    this._register('hidden_state', [this.neuronPoints], new THREE.Vector3(0, 0, 0), 11);

    const { pre, post, polarity, weight_norm: weights } = layout.edges;
    const buildEdges = (sign, color) => {
      const pts = [];
      const alphas = [];
      for (let i = 0; i < pre.length; i += 1) {
        if (polarity[i] !== sign) continue;
        const a = this.neuronPositions[pre[i]];
        const b = this.neuronPositions[post[i]];
        if (!a || !b) continue;
        pts.push(a.x, a.y, a.z, b.x, b.y, b.z);
        const w = 0.35 + 0.65 * Number(weights[i] || 0);
        alphas.push(w, w);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      const base = new THREE.Color(color);
      const cols = new Float32Array(alphas.length * 3);
      alphas.forEach((w, idx) => cols.set([base.r * w, base.g * w, base.b * w], idx * 3));
      g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      const m = new THREE.LineBasicMaterial({
        vertexColors: true,
        opacity: 0.32,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const lines = new THREE.LineSegments(g, m);
      this.scene.add(lines);
      return lines;
    };
    this.excEdges = buildEdges(1, COLORS.frozenExc);
    this.inhEdges = buildEdges(-1, COLORS.frozenInh);
    this._register('w_exc', [this.excEdges], new THREE.Vector3(0, 0, 0), 10);
    this._register('w_inh', [this.inhEdges], new THREE.Vector3(0, 0, 0), 10);

    this.recurrentPaths = [];
    const rng = mulberry32(17);
    for (let i = 0; i < 90; i += 1) {
      const idx = Math.floor(rng() * pre.length);
      const a = this.neuronPositions[pre[idx]];
      const b = this.neuronPositions[post[idx]];
      if (a && b && a.distanceTo(b) > 0.4) {
        this.recurrentPaths.push(new THREE.LineCurve3(a.clone(), b.clone()));
      }
    }

    const sampleRng = mulberry32(5);
    this.inFanNeurons = Array.from({ length: 150 }, () => Math.floor(sampleRng() * count));
    this.outFanNeurons = Array.from({ length: 150 }, () => Math.floor(sampleRng() * count));
  }

  _makeSlab({ id, x, y, h, d, cols, rows, seed }) {
    const matrix = new MatrixTexture(cols, rows, seed);
    this.matrixTextures.push(matrix);
    const faceMat = new THREE.MeshBasicMaterial({ map: matrix.texture, opacity: 0.92, side: THREE.DoubleSide });
    const sideMat = new THREE.MeshBasicMaterial({ color: COLORS.trainable, opacity: 0.55 });
    const geo = new THREE.BoxGeometry(0.1, h, d);
    const slab = new THREE.Mesh(geo, [faceMat, faceMat, sideMat, sideMat, sideMat, sideMat]);
    slab.position.set(x, y, 0);
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: COLORS.trainable, opacity: 0.95 })
    );
    slab.add(outline);
    this.scene.add(slab);
    this._register(id, [slab, outline], new THREE.Vector3(x, y, 0), 8);
    return slab;
  }

  _fanLines(id, fromPoints, toPoints, color, opacity) {
    const pts = [];
    for (let i = 0; i < fromPoints.length; i += 1) {
      const a = fromPoints[i];
      const b = toPoints[i];
      pts.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const m = new THREE.LineBasicMaterial({
      color,
      opacity,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const lines = new THREE.LineSegments(g, m);
    this.scene.add(lines);
    if (id) {
      this._register(id, [lines], this.components[id]?.center || new THREE.Vector3(), 8);
    }
    return lines;
  }

  _slabPoint(slab, h, d, rng) {
    return new THREE.Vector3(
      slab.position.x,
      slab.position.y + (rng() - 0.5) * h * 0.9,
      (rng() - 0.5) * d * 0.9
    );
  }

  _buildInputProjection() {
    const h = 7.2;
    const d = 3.4;
    this.inSlab = this._makeSlab({ id: 'input_projection', x: -7.2, y: 0.2, h, d, cols: 22, rows: 64, seed: 11 });
    const rng = mulberry32(23);

    const obsFrom = [];
    const obsTo = [];
    this.obsPaths = [];
    this.obsCubes.forEach((cube) => {
      for (let k = 0; k < 3; k += 1) {
        const target = this._slabPoint(this.inSlab, h, d, rng);
        obsFrom.push(cube.position.clone());
        obsTo.push(target);
        this.obsPaths.push(new THREE.LineCurve3(cube.position.clone(), target));
      }
    });
    const obsLines = this._fanLines(null, obsFrom, obsTo, COLORS.data, 0.08);
    this._register('obs', [obsLines], this.obsCenter, 9);

    const from = [];
    const to = [];
    this.inPaths = [];
    for (const idx of this.inFanNeurons) {
      const start = this._slabPoint(this.inSlab, h, d, rng);
      const end = this.neuronPositions[idx].clone();
      from.push(start);
      to.push(end);
      const mid = start.clone().lerp(end, 0.5).add(new THREE.Vector3(0, 0.6, 0));
      this.inPaths.push(new THREE.QuadraticBezierCurve3(start, mid, end));
    }
    this._fanLines('input_projection', from, to, COLORS.trainable, 0.07);
  }

  _buildTau() {
    const geo = new THREE.TorusGeometry(4.9, 0.045, 16, 220);
    const mat = new THREE.MeshBasicMaterial({ color: COLORS.trainable, opacity: 0.9 });
    this.tauRing = new THREE.Mesh(geo, mat);
    this.tauRing.rotation.x = Math.PI / 2;
    this.scene.add(this.tauRing);

    const tickGeo = new THREE.SphereGeometry(0.12, 16, 16);
    this.tauBead = new THREE.Mesh(tickGeo, new THREE.MeshBasicMaterial({ color: 0xffd9a8, opacity: 1 }));
    this.scene.add(this.tauBead);
    this._register('tau', [this.tauRing, this.tauBead], new THREE.Vector3(4.9, 0, 0), 9);
  }

  _buildRuntimeControls() {
    const defs = [
      ['g_exc', -2.4, COLORS.frozenExc],
      ['g_inh', 0, COLORS.frozenInh],
      ['lesion_mask', 2.4, COLORS.runtime],
    ];
    this.runtimeAnchors = {};
    for (const [id, x, accent] of defs) {
      const pos = new THREE.Vector3(x, 6.1, 0);
      const disc = new THREE.Mesh(
        new THREE.CylinderGeometry(0.5, 0.5, 0.1, 48),
        new THREE.MeshBasicMaterial({ color: COLORS.runtime, opacity: 0.32 })
      );
      disc.position.copy(pos);
      const rim = new THREE.Mesh(
        new THREE.TorusGeometry(0.5, 0.03, 10, 64),
        new THREE.MeshBasicMaterial({ color: COLORS.runtime, opacity: 0.95 })
      );
      rim.rotation.x = Math.PI / 2;
      rim.position.copy(pos);
      const core = new THREE.Mesh(
        new THREE.SphereGeometry(0.13, 16, 16),
        new THREE.MeshBasicMaterial({ color: accent, opacity: 1 })
      );
      core.position.copy(pos).add(new THREE.Vector3(0, 0.12, 0));

      const dashGeo = new THREE.BufferGeometry().setFromPoints([
        pos.clone().add(new THREE.Vector3(0, -0.1, 0)),
        new THREE.Vector3(x * 0.55, 2.6, 0),
      ]);
      const dash = new THREE.Line(
        dashGeo,
        new THREE.LineDashedMaterial({ color: COLORS.runtime, dashSize: 0.18, gapSize: 0.14, opacity: 0.6 })
      );
      dash.computeLineDistances();
      this.scene.add(disc, rim, core, dash);
      this._register(id, [disc, rim, core, dash], pos, 7);
      this.runtimeAnchors[id] = pos;
    }
  }

  _buildReadout() {
    this.actorSlab = this._makeSlab({ id: 'actor_head', x: 7.2, y: 1.9, h: 3.4, d: 2.6, cols: 4, rows: 40, seed: 31 });
    this.criticSlab = this._makeSlab({ id: 'critic_head', x: 7.2, y: -2.6, h: 1.6, d: 1.6, cols: 1, rows: 20, seed: 47 });
    const rng = mulberry32(53);

    const buildFan = (slab, h, d, id, neuronIdxs) => {
      const from = [];
      const to = [];
      const paths = [];
      for (const idx of neuronIdxs) {
        const start = this.neuronPositions[idx].clone();
        const end = this._slabPoint(slab, h, d, rng);
        from.push(start);
        to.push(end);
        const mid = start.clone().lerp(end, 0.5).add(new THREE.Vector3(0, 0.5, 0));
        paths.push(new THREE.QuadraticBezierCurve3(start, mid, end));
      }
      this._fanLines(id, from, to, COLORS.trainable, 0.07);
      return paths;
    };
    this.actorPaths = buildFan(this.actorSlab, 3.4, 2.6, 'actor_head', this.outFanNeurons.slice(0, 100));
    this.criticPaths = buildFan(this.criticSlab, 1.6, 1.6, 'critic_head', this.outFanNeurons.slice(100));

    this.actionBars = [];
    this.actionProbs = [0.25, 0.25, 0.25, 0.25];
    this.actionTargets = [0.25, 0.25, 0.25, 0.25];
    const barObjects = [];
    this.actionPaths = [];
    for (let i = 0; i < 4; i += 1) {
      const y = 3.2 - i * 0.86;
      const bar = new THREE.Mesh(
        new THREE.BoxGeometry(1, 0.42, 0.42),
        new THREE.MeshBasicMaterial({ color: COLORS.data, opacity: 0.85 })
      );
      bar.geometry.translate(0.5, 0, 0);
      bar.position.set(10.3, y, 0);
      const track = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(2.2, 0.42, 0.42).translate(1.1, 0, 0)),
        new THREE.LineBasicMaterial({ color: COLORS.data, opacity: 0.25 })
      );
      track.position.copy(bar.position);
      this.scene.add(bar, track);
      this.actionBars.push(bar);
      barObjects.push(bar, track);
      const start = this._slabPoint(this.actorSlab, 3.4, 2.6, rng);
      this.actionPaths.push(new THREE.LineCurve3(start, new THREE.Vector3(10.3, y, 0)));
    }
    this._register('policy_out', barObjects, new THREE.Vector3(11.2, 1.9, 0), 7);

    this.valueOrb = new THREE.Mesh(
      new THREE.SphereGeometry(0.42, 32, 32),
      new THREE.MeshBasicMaterial({ color: COLORS.data, opacity: 0.9 })
    );
    this.valueOrb.position.set(10.8, -2.6, 0);
    const halo = new THREE.Mesh(
      new THREE.TorusGeometry(0.62, 0.02, 8, 64),
      new THREE.MeshBasicMaterial({ color: COLORS.data, opacity: 0.5 })
    );
    halo.position.copy(this.valueOrb.position);
    this.valueHalo = halo;
    this.scene.add(this.valueOrb, halo);
    this._register('value_out', [this.valueOrb, halo], this.valueOrb.position.clone(), 6);
    this.valuePaths = [
      new THREE.LineCurve3(this.criticSlab.position.clone(), this.valueOrb.position.clone()),
    ];
  }

  _buildEnvironmentLoop() {
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(12.9, 1.2, 0),
      new THREE.Vector3(14.6, -2.5, 0),
      new THREE.Vector3(12.4, -6.6, 0),
      new THREE.Vector3(0, -7.6, 0),
      new THREE.Vector3(-12.4, -6.6, 0),
      new THREE.Vector3(-14.4, -2.4, 0),
      new THREE.Vector3(-12.3, 0.4, 0),
    ]);
    this.envCurve = curve;
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 260, 0.05, 10, false),
      new THREE.MeshBasicMaterial({ color: COLORS.env, opacity: 0.6 })
    );
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(3.6, 0.9, 1.2),
      new THREE.MeshBasicMaterial({ color: 0x1a1f2a, opacity: 0.95 })
    );
    box.position.set(0, -7.6, 0);
    const boxEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(box.geometry),
      new THREE.LineBasicMaterial({ color: COLORS.env, opacity: 0.9 })
    );
    box.add(boxEdge);
    this.scene.add(tube, box);
    this._register('env', [tube, box, boxEdge], new THREE.Vector3(0, -7.2, 0), 12);
  }

  _buildParticles() {
    const max = 520;
    this.particleGeo = new THREE.BufferGeometry();
    this.particlePositions = new Float32Array(max * 3);
    this.particleColors = new Float32Array(max * 3);
    this.particleGeo.setAttribute('position', new THREE.BufferAttribute(this.particlePositions, 3));
    this.particleGeo.setAttribute('color', new THREE.BufferAttribute(this.particleColors, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.2,
      map: this.dotTexture,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.particlePoints = new THREE.Points(this.particleGeo, mat);
    this.scene.add(this.particlePoints);
    this.particleMax = max;
    this._rebuildParticleSet();
  }

  _rebuildParticleSet() {
    const rng = mulberry32(this.mode === 'forward' ? 3 : 9);
    const pools = [];
    const data = new THREE.Color(COLORS.data);
    const amber = new THREE.Color(COLORS.trainable);
    const ice = new THREE.Color(COLORS.frozenExc);
    const grad = new THREE.Color(COLORS.grad);
    const env = new THREE.Color(COLORS.env);

    if (this.mode === 'forward') {
      pools.push({ paths: this.obsPaths, count: 70, color: data, speed: 0.55 });
      pools.push({ paths: this.inPaths, count: 110, color: amber, speed: 0.4 });
      pools.push({ paths: this.recurrentPaths, count: 90, color: ice, speed: 0.6 });
      pools.push({ paths: this.actorPaths, count: 80, color: amber, speed: 0.42 });
      pools.push({ paths: this.criticPaths, count: 40, color: amber, speed: 0.42 });
      pools.push({ paths: this.actionPaths, count: 24, color: data, speed: 0.7 });
      pools.push({ paths: this.valuePaths, count: 8, color: data, speed: 0.7 });
      pools.push({ paths: [this.envCurve], count: 36, color: env, speed: 0.09 });
    } else {
      const rev = (paths) => paths.map((p) => new ReversedCurve(p));
      pools.push({ paths: rev(this.actionPaths), count: 24, color: grad, speed: 0.7 });
      pools.push({ paths: rev(this.valuePaths), count: 10, color: grad, speed: 0.7 });
      pools.push({ paths: rev(this.actorPaths), count: 90, color: grad, speed: 0.42 });
      pools.push({ paths: rev(this.criticPaths), count: 45, color: grad, speed: 0.42 });
      pools.push({ paths: rev(this.recurrentPaths), count: 90, color: grad.clone().lerp(ice, 0.45), speed: 0.5 });
      pools.push({ paths: rev(this.inPaths), count: 120, color: grad, speed: 0.4 });
    }

    this.particles = [];
    for (const pool of pools) {
      if (!pool.paths || pool.paths.length === 0) continue;
      for (let i = 0; i < pool.count && this.particles.length < this.particleMax; i += 1) {
        this.particles.push({
          path: pool.paths[Math.floor(rng() * pool.paths.length)],
          paths: pool.paths,
          t: rng(),
          speed: pool.speed * (0.75 + rng() * 0.5),
          color: pool.color,
        });
      }
    }
    for (let i = 0; i < this.particleMax; i += 1) {
      this.particlePositions.set([0, -999, 0], i * 3);
      this.particleColors.set([0, 0, 0], i * 3);
    }
    this.particleGeo.setDrawRange(0, this.particles.length);
  }

  _buildLabels() {
    const defs = [
      ['obs', 'x_t  [22]', new THREE.Vector3(-11.5, 5.4, 0)],
      ['input_projection', 'input_projection', new THREE.Vector3(-7.2, 4.4, 0)],
      ['hidden_state', 'h_t  2.500 neuron', new THREE.Vector3(0, -3.9, 0)],
      ['w_exc', 'w_exc_csr', new THREE.Vector3(-2.6, 3.0, 1.2)],
      ['w_inh', 'w_inh_csr', new THREE.Vector3(2.8, 2.6, -1.0)],
      ['tau', 'tau', new THREE.Vector3(-4.9, 0.3, 0)],
      ['g_exc', 'g_exc', new THREE.Vector3(-2.4, 6.85, 0)],
      ['g_inh', 'g_inh', new THREE.Vector3(0, 6.85, 0)],
      ['lesion_mask', 'lesion_mask', new THREE.Vector3(2.4, 6.85, 0)],
      ['actor_head', 'actor_head', new THREE.Vector3(7.2, 4.05, 0)],
      ['critic_head', 'critic_head', new THREE.Vector3(7.2, -3.85, 0)],
      ['policy_out', 'pi(a | h_t)', new THREE.Vector3(11.3, 3.95, 0)],
      ['value_out', 'V(s_t)', new THREE.Vector3(10.8, -3.5, 0)],
      ['env', 'TrafficIntersectionEnv', new THREE.Vector3(0, -8.55, 0)],
    ];
    this.labelLayer.innerHTML = '';
    this.labels = defs.map(([id, text, pos]) => {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = `arch-label status-${COMPONENTS[id].status}`;
      el.textContent = text;
      el.addEventListener('click', () => this.select(id, { focus: true }));
      el.addEventListener('mouseenter', () => this._setHovered(id));
      el.addEventListener('mouseleave', () => this._setHovered(null));
      this.labelLayer.appendChild(el);
      return { id, el, pos };
    });
  }

  /* ------------------------------------------------------------------ */
  /* Interaction                                                         */
  /* ------------------------------------------------------------------ */

  _bindControls() {
    this.root.querySelectorAll('[data-arch-mode]').forEach((btn) => {
      btn.addEventListener('click', () => this.setMode(btn.dataset.archMode));
    });
    this.root.querySelectorAll('[data-arch-filter]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const status = btn.dataset.archFilter;
        this.filterStatus = this.filterStatus === status ? null : status;
        this.root.querySelectorAll('[data-arch-filter]').forEach((b) => {
          const active = b.dataset.archFilter === this.filterStatus;
          b.classList.toggle('is-active', active);
          b.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        this._applyEmphasis();
      });
    });
    const labelsBtn = this.root.querySelector('[data-arch-action="labels"]');
    if (labelsBtn) {
      labelsBtn.addEventListener('click', () => {
        this.showLabels = !this.showLabels;
        labelsBtn.classList.toggle('is-active', this.showLabels);
        labelsBtn.setAttribute('aria-pressed', this.showLabels ? 'true' : 'false');
        this.labelLayer.classList.toggle('is-hidden', !this.showLabels);
      });
    }
    const resetBtn = this.root.querySelector('[data-arch-action="reset"]');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        this.select(null);
        this._tweenCamera(this.homeCamera.clone(), this.homeTarget.clone());
      });
    }
  }

  setMode(mode) {
    if (mode !== 'forward' && mode !== 'backward') return;
    this.mode = mode;
    this.root.querySelectorAll('[data-arch-mode]').forEach((btn) => {
      const active = btn.dataset.archMode === mode;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    this.stage.classList.toggle('is-backward', mode === 'backward');
    if (this.modeCaption) {
      this.modeCaption.textContent =
        mode === 'forward'
          ? 'Forward pass: x_t -> input_projection -> h_t -> actor_head / critic_head -> aksi -> environment'
          : 'PPO update: gradien loss mengalir mundur ke actor_head, critic_head, tau, dan input_projection. w_exc_csr / w_inh_csr dilewati gradien (BPTT) tetapi nilainya tidak diubah.';
    }
    if (this.initialized) {
      this._rebuildParticleSet();
      this._applyEmphasis();
    }
  }

  select(id, { focus = false } = {}) {
    this.selectedId = id && COMPONENTS[id] ? id : null;
    this._renderDetail(this.selectedId);
    this.indexEl.querySelectorAll('[data-arch-select]').forEach((btn) => {
      const active = btn.dataset.archSelect === this.selectedId;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-current', active ? 'true' : 'false');
    });
    if (this.initialized) {
      this._applyEmphasis();
      if (focus && this.selectedId) {
        const entry = this.components[this.selectedId];
        if (entry) {
          const dir = this.camera.position.clone().sub(this.controls.target).normalize();
          const target = entry.center.clone();
          const pos = target.clone().add(dir.multiplyScalar(entry.focusDist));
          this._tweenCamera(pos, target);
        }
      }
    }
  }

  _tweenCamera(toPos, toTarget) {
    this.cameraTween = {
      fromPos: this.camera.position.clone(),
      fromTarget: this.controls.target.clone(),
      toPos,
      toTarget,
      start: performance.now(),
      duration: 750,
    };
  }

  _onPointerMove(event) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointerPx = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    this.pointer.x = (this.pointerPx.x / rect.width) * 2 - 1;
    this.pointer.y = -(this.pointerPx.y / rect.height) * 2 + 1;
    this.pointerInside = true;
  }

  _pick() {
    if (!this.pointerInside) return;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables, false);
    let id = null;
    for (const hit of hits) {
      const cid = hit.object.userData.componentId;
      if (!cid) continue;
      if (this.filterStatus && COMPONENTS[cid].status !== this.filterStatus) continue;
      id = cid;
      if (hit.object.type === 'Mesh') break;
      if (hit.object.type === 'Points') break;
    }
    this._setHovered(id);
  }

  _setHovered(id) {
    if (this.hoveredId === id) {
      this._positionHoverTag();
      return;
    }
    this.hoveredId = id;
    this.canvas.style.cursor = id ? 'pointer' : 'grab';
    if (!id) {
      this.hoverTag.hidden = true;
    } else {
      const info = COMPONENTS[id];
      this.hoverTag.innerHTML = `
        <span class="arch-hover-title">${info.title}</span>
        <span class="arch-status-chip status-${info.status}">${STATUS_LABELS[info.status]}</span>
        <span class="arch-hover-meta">${info.params === '0' ? 'tanpa parameter' : info.params}</span>
      `;
      this.hoverTag.hidden = false;
      this._positionHoverTag();
    }
    if (this.initialized) this._applyEmphasis();
  }

  _positionHoverTag() {
    if (this.hoverTag.hidden) return;
    const w = this.stage.clientWidth;
    const tagW = this.hoverTag.offsetWidth || 200;
    const x = Math.min(this.pointerPx.x + 16, w - tagW - 12);
    const y = Math.max(this.pointerPx.y - 12, 12);
    this.hoverTag.style.transform = `translate(${x}px, ${y}px)`;
  }

  _applyEmphasis() {
    const focusId = this.hoveredId || this.selectedId;
    for (const [id, entry] of Object.entries(this.components)) {
      const status = COMPONENTS[id].status;
      let factor = 1;
      if (this.filterStatus && status !== this.filterStatus) factor = 0.08;
      if (this.mode === 'backward' && status === 'frozen') factor *= 0.55;
      if (this.mode === 'backward' && (status === 'data' || status === 'runtime')) factor *= 0.45;
      if (focusId && id !== focusId && !this.filterStatus) factor *= 0.4;
      if (focusId && id === focusId) factor = Math.max(factor, 1) * 1.35;
      for (const mat of entry.materials) {
        mat.opacity = Math.min(1, mat.userData.baseOpacity * factor);
      }
    }
    for (const label of this.labels) {
      const status = COMPONENTS[label.id].status;
      label.el.classList.toggle('is-dim', Boolean(this.filterStatus && status !== this.filterStatus));
      label.el.classList.toggle('is-focus', label.id === focusId);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Inspector DOM                                                       */
  /* ------------------------------------------------------------------ */

  _renderIndex() {
    if (!this.indexEl) return;
    const groups = GROUP_ORDER.map((group) => {
      const items = Object.entries(COMPONENTS)
        .filter(([, info]) => info.group === group)
        .map(
          ([id, info]) => `
          <button type="button" class="arch-index-item" data-arch-select="${id}" aria-current="false">
            <span class="arch-dot status-${info.status}" aria-hidden="true"></span>
            <span class="arch-index-name mono-value">${info.title}</span>
            <span class="arch-index-params mono-value">${info.params === '0' ? '-' : info.params.split(' ')[0]}</span>
          </button>`
        )
        .join('');
      return `<div class="arch-index-group"><div class="arch-index-heading">${group}</div>${items}</div>`;
    });
    this.indexEl.innerHTML = groups.join('');
    this.indexEl.querySelectorAll('[data-arch-select]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.dataset.archSelect;
        if (this.selectedId === id) {
          this.select(null);
        } else {
          this.select(id, { focus: true });
        }
      });
      btn.addEventListener('mouseenter', () => this.initialized && this._setHovered(btn.dataset.archSelect));
      btn.addEventListener('mouseleave', () => this.initialized && this._setHovered(null));
    });
  }

  _renderDetail(id) {
    if (!this.detailEl) return;
    if (!id) {
      this.detailEl.innerHTML = `
        <div class="arch-detail-kicker">FlyConnectomePolicy</div>
        <h3 class="arch-detail-title">Alur satu step</h3>
        <p class="arch-detail-text">
          Klik komponen di scene atau di daftar untuk melihat fungsi, bentuk tensor, jumlah parameter,
          dan apakah nilainya berubah selama training.
        </p>
        <div class="arch-equation mono-value">
          <div>h~ = h_{t-1} * lesion_mask</div>
          <div>s  = g_exc W_exc h~ - g_inh W_inh h~ + W_in x_t + b_in</div>
          <div>h_t = [(1 - tau) h~ + tau tanh(s)] * lesion_mask</div>
          <div>logits = actor_head(h_t),  V = critic_head(h_t)</div>
        </div>
        <dl class="arch-kv">
          <div><dt>Trainable</dt><dd class="mono-value">70.006 params</dd></div>
          <div><dt>Frozen</dt><dd class="mono-value">87.171 sinapsis</dd></div>
          <div><dt>Neuron</dt><dd class="mono-value">2.500 (1.800 CX + 700 MB)</dd></div>
        </dl>
      `;
      return;
    }
    const info = COMPONENTS[id];
    const changes = {
      trainable: 'Berubah selama training',
      frozen: 'Tidak berubah selama training',
      runtime: 'Tidak dilatih, dapat diubah saat inferensi',
      data: 'Bukan parameter',
    }[info.status];
    const rows = info.rows
      .map(([k, v]) => `<div><dt class="mono-value">${k}</dt><dd>${v}</dd></div>`)
      .join('');
    this.detailEl.innerHTML = `
      <div class="arch-detail-kicker">${info.group}</div>
      <h3 class="arch-detail-title mono-value">${info.title}</h3>
      <div class="arch-detail-badges">
        <span class="arch-status-chip status-${info.status}">${STATUS_LABELS[info.status]}</span>
        <span class="arch-change-note status-${info.status}">${changes}</span>
      </div>
      <dl class="arch-kv">
        <div><dt>Kode</dt><dd class="mono-value">${info.code}</dd></div>
        <div><dt>File</dt><dd class="mono-value">${info.file}</dd></div>
        <div><dt>Shape</dt><dd class="mono-value">${info.shape}</dd></div>
        <div><dt>Parameter</dt><dd class="mono-value">${info.params}</dd></div>
      </dl>
      <h4 class="arch-detail-sub">Fungsi</h4>
      <p class="arch-detail-text">${info.role}</p>
      <h4 class="arch-detail-sub">Selama training</h4>
      <p class="arch-detail-text">${info.training}</p>
      ${rows ? `<dl class="arch-kv arch-kv-compact">${rows}</dl>` : ''}
    `;
  }

  /* ------------------------------------------------------------------ */
  /* Render loop                                                         */
  /* ------------------------------------------------------------------ */

  _resize() {
    if (!this.renderer) return;
    const w = this.stage.clientWidth;
    const h = this.stage.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.bloom.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _loop() {
    if (this.loopActive) return;
    this.loopActive = true;
    const tick = () => {
      if (!this.visible) {
        this.loopActive = false;
        return;
      }
      this._update();
      this.composer.render();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  _update() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const now = performance.now();
    const time = now / 1000;

    if (this.cameraTween) {
      const tw = this.cameraTween;
      const k = Math.min(1, (now - tw.start) / tw.duration);
      const e = 1 - Math.pow(1 - k, 3);
      this.camera.position.lerpVectors(tw.fromPos, tw.toPos, e);
      this.controls.target.lerpVectors(tw.fromTarget, tw.toTarget, e);
      if (k >= 1) this.cameraTween = null;
    }
    this.controls.update();

    // Particles along paths
    const tmp = new THREE.Vector3();
    for (let i = 0; i < this.particles.length; i += 1) {
      const p = this.particles[i];
      p.t += p.speed * dt;
      if (p.t >= 1) {
        p.t -= 1;
        p.path = p.paths[Math.floor(Math.random() * p.paths.length)];
      }
      p.path.getPointAt(p.t, tmp);
      this.particlePositions[i * 3] = tmp.x;
      this.particlePositions[i * 3 + 1] = tmp.y;
      this.particlePositions[i * 3 + 2] = tmp.z;
      const fade = Math.sin(Math.PI * p.t);
      this.particleColors[i * 3] = p.color.r * fade;
      this.particleColors[i * 3 + 1] = p.color.g * fade;
      this.particleColors[i * 3 + 2] = p.color.b * fade;
    }
    this.particleGeo.attributes.position.needsUpdate = true;
    this.particleGeo.attributes.color.needsUpdate = true;

    // tau bead orbit and runtime pulse
    const beadAngle = time * 0.6;
    this.tauBead.position.set(Math.cos(beadAngle) * 4.9, 0, Math.sin(beadAngle) * 4.9);
    this.neuronPoints.rotation.y = Math.sin(time * 0.15) * 0.04;
    this.excEdges.rotation.y = this.neuronPoints.rotation.y;
    this.inhEdges.rotation.y = this.neuronPoints.rotation.y;

    // Policy bars: animate toward drifting target distribution
    if (Math.floor(time * 0.8) !== this.lastBarEpoch) {
      this.lastBarEpoch = Math.floor(time * 0.8);
      const raw = [0, 1, 2, 3].map(() => Math.random() ** 2 + 0.05);
      const sum = raw.reduce((a, b) => a + b, 0);
      this.actionTargets = raw.map((v) => v / sum);
    }
    for (let i = 0; i < 4; i += 1) {
      this.actionProbs[i] += (this.actionTargets[i] - this.actionProbs[i]) * Math.min(1, dt * 3);
      this.actionBars[i].scale.x = Math.max(0.04, this.actionProbs[i] * 2.2);
    }
    this.valueHalo.rotation.y = time * 0.8;
    this.valueHalo.rotation.x = Math.PI / 2 + Math.sin(time) * 0.2;

    // Trainable weights visibly shift only during the PPO update view.
    if (this.mode === 'backward') {
      if (now - this.lastNudgeMs > 140) {
        this.lastNudgeMs = now;
        this.matrixTextures.forEach((m) => m.nudge(0.02));
      }
      const pulse = 0.75 + 0.25 * Math.sin(time * 5);
      this.components.tau.materials.forEach((m) => {
        m.opacity = Math.min(1, m.userData.baseOpacity * pulse * (this.filterStatus && this.filterStatus !== 'trainable' ? 0.08 : 1));
      });
    }

    this._pick();
    this._updateLabels();
  }

  _updateLabels() {
    if (!this.showLabels) return;
    const w = this.stage.clientWidth;
    const h = this.stage.clientHeight;
    const v = new THREE.Vector3();
    for (const label of this.labels) {
      v.copy(label.pos).project(this.camera);
      const visible = v.z < 1 && v.x > -1.1 && v.x < 1.1 && v.y > -1.1 && v.y < 1.1;
      label.el.style.display = visible ? '' : 'none';
      if (!visible) continue;
      const x = (v.x * 0.5 + 0.5) * w;
      const y = (-v.y * 0.5 + 0.5) * h;
      label.el.style.transform = `translate(-50%, -50%) translate(${x}px, ${y}px)`;
    }
  }
}
