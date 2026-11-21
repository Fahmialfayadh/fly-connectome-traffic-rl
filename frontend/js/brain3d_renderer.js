/**
 * Live 3D Drosophila brain renderer (Three.js).
 *
 * Anatomy: FlyWire whole-brain neuropil surface + JFRC2 neuropil meshes in FAFB14.1
 * space (built by scripts/build_brain_assets.py). Each of the 2,500 recurrent neurons
 * of the Fly Connectome PPO core is drawn at either (a) a deterministic sample inside its
 * dominant neuropil or (b) its FlyWire soma coordinate. Colour, size, and travelling
 * synaptic pulses encode the live hidden state h_t streamed by the simulation WebSocket.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const H_SCALE = 0.4;
const ASSET_BASE = '/assets/brain';
const REGION_KEYS = ['PB', 'EB', 'FB', 'NO', 'MB_CA', 'MB_PED', 'MB_ML', 'MB_VL'];
const REGION_LABELS = {
  PB: 'PB', EB: 'EB', FB: 'FB', NO: 'NO',
  MB_CA: 'MB CA', MB_PED: 'MB PED', MB_ML: 'MB ML', MB_VL: 'MB VL',
};
const REGION_TOOLTIPS = {
  PB: 'Protocerebral Bridge (CX): jembatan simetris yang menyalurkan sinyal orientasi arah ke FB dan EB.',
  EB: 'Ellipsoid Body (CX): struktur cincin pusat yang menjaga representasi arah dan status fasa.',
  FB: 'Fan-shaped Body (CX): struktur kipas berlapis terbesar di CX untuk integrasi konteks dan pemilihan aksi.',
  NO: 'Noduli (CX): sepasang lobus ventral di bawah FB/EB untuk integrasi masukan ke Central Complex.',
  MB_CA: 'Mushroom Body Calyx (MB): wilayah masukan dendritik utama sel Kenyon di bagian posterior MB.',
  MB_PED: 'Mushroom Body Pedunculus (MB): berkas akson penghubung dari Calyx menuju lobus keluaran MB.',
  MB_ML: 'Mushroom Body Medial Lobe (MB): lobus keluaran horizontal yang terhubung ke neuron keluaran MBON.',
  MB_VL: 'Mushroom Body Vertical Lobe (MB): lobus keluaran vertikal untuk asosiasi memori dan konteks waktu.',
};
const MESH_LABELS = [
  ['PB', 'PB'], ['FB', 'FB'], ['EB', 'EB'], ['NO', 'NO'],
  ['MB_CA_L', 'MB CA L'], ['MB_CA_R', 'MB CA R'],
  ['MB_ML_L', 'MB ML L'], ['MB_ML_R', 'MB ML R'],
];
const CAMERA_PRESETS = {
  front: new THREE.Vector3(0, 0.6, 15.5),
  top: new THREE.Vector3(0, 15.5, 0.6),
  side: new THREE.Vector3(15.5, 0.8, 0.2),
};
const CAMERA_TARGET = new THREE.Vector3(0, 0.15, 0);

const regionOfNeuropil = (name) =>
  name.startsWith('MB_') ? name.slice(0, name.lastIndexOf('_')) : name;
const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);

const POINT_VERTEX = /* glsl */ `
  attribute float aAct;
  attribute float aSpike;
  attribute float aAlive;
  attribute float aGroup;
  attribute float aIndex;
  uniform float uPixelRatio;
  uniform float uSize;
  uniform float uFocus;
  uniform float uSelected;
  uniform float uHover;
  uniform float uFlyActive;
  uniform float uActThreshold;
  uniform float uQuietAlpha;
  uniform float uNormScale;
  uniform vec3 uNeg;
  uniform vec3 uPos;
  uniform vec3 uMid;
  uniform vec3 uNegHot;
  uniform vec3 uPosHot;
  uniform vec3 uDead;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vRing;
  varying float vDead;
  varying float vHot;

  void main() {
    float absAct = abs(aAct);
    float normAct = clamp(absAct / max(uNormScale, 0.05), 0.0, 1.0);
    float activeRamp = smoothstep(uActThreshold, 1.0, normAct);
    float spikeBoost = clamp(aSpike * 4.5, 0.0, 1.0);
    float energy = clamp(activeRamp + spikeBoost * 0.45, 0.0, 1.0);

    vec3 polarityColor = aAct >= 0.0
      ? mix(uPos, uPosHot, smoothstep(0.45, 1.0, energy))
      : mix(uNeg, uNegHot, smoothstep(0.45, 1.0, energy));
    vColor = mix(uMid, polarityColor, smoothstep(0.05, 0.45, energy));

    float focusDim = (uFocus < -0.5 || abs(aGroup - uFocus) < 0.5) ? 1.0 : 0.06;
    float size = uSize * (0.42 + 1.55 * pow(energy, 0.85));
    vAlpha = mix(uQuietAlpha, 0.96, activeRamp) * focusDim;
    vHot = energy;
    vDead = 0.0;

    if (uFlyActive < 0.5) {
      vColor = uMid;
      vAlpha = 0.28 * focusDim;
      size = uSize * 0.55;
      vHot = 0.0;
    } else if (aAlive < 0.5) {
      vColor = uDead;
      vAlpha = 0.92 * focusDim;
      size = uSize * 1.15;
      vDead = 1.0;
      vHot = 0.0;
    }

    bool picked = abs(aIndex - uSelected) < 0.5 || abs(aIndex - uHover) < 0.5;
    vRing = picked ? 1.0 : 0.0;
    if (picked) {
      size = max(size, uSize * 1.2) * 2.1;
      vAlpha = 1.0;
    }

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * uPixelRatio * (14.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const POINT_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  varying float vRing;
  varying float vDead;
  varying float vHot;

  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c);
    if (r > 0.5 || vAlpha < 0.015) discard;

    if (vRing > 0.5) {
      float ring = smoothstep(0.5, 0.44, r) * smoothstep(0.32, 0.38, r);
      float core = smoothstep(0.26, 0.0, r);
      gl_FragColor = vec4(mix(vColor, vec3(1.0), ring), max(core, ring) * vAlpha);
      return;
    }
    if (vDead > 0.5) {
      float ring = smoothstep(0.5, 0.40, r) * smoothstep(0.18, 0.28, r);
      gl_FragColor = vec4(vColor, ring * vAlpha);
      return;
    }

    float core = smoothstep(0.34, 0.08, r);
    float halo = pow(smoothstep(0.5, 0.0, r), 1.8) * (0.55 + 0.62 * vHot);
    float alpha = max(core, halo) * vAlpha;
    vec3 color = vColor * (1.12 + 1.05 * core * vHot);
    gl_FragColor = vec4(color, alpha);
  }
`;

const EDGE_VERTEX = /* glsl */ `
  attribute float aT;
  attribute float aAct;
  attribute float aWeight;
  attribute float aPolarity;
  attribute float aGroup;
  attribute float aPre;
  attribute float aPost;
  uniform float uSelected;
  uniform float uFocus;
  varying float vT;
  varying float vAct;
  varying float vWeight;
  varying float vPolarity;
  varying float vSel;
  varying float vPhase;
  varying float vFocusDim;

  void main() {
    vT = aT;
    vAct = aAct;
    vWeight = aWeight;
    vPolarity = aPolarity;
    vSel = (abs(aPre - uSelected) < 0.5 || abs(aPost - uSelected) < 0.5) ? 1.0 : 0.0;
    vPhase = fract(aPre * 0.1371 + aPost * 0.0713);
    vFocusDim = (uFocus < -0.5 || abs(aGroup - uFocus) < 0.5 || vSel > 0.5) ? 1.0 : 0.08;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const EDGE_FRAGMENT = /* glsl */ `
  uniform float uTime;
  uniform float uOpacity;
  uniform float uEdgeThreshold;
  uniform vec3 uExcColor;
  uniform vec3 uInhColor;
  uniform vec3 uSelColor;
  varying float vT;
  varying float vAct;
  varying float vWeight;
  varying float vPolarity;
  varying float vSel;
  varying float vPhase;
  varying float vFocusDim;

  void main() {
    float activeStrength = smoothstep(uEdgeThreshold, 1.0, vAct);
    float wave = fract(vT - uTime * 0.65 + vPhase);
    float packet = smoothstep(0.76, 0.96, wave) * smoothstep(1.0, 0.96, wave);
    float trail = smoothstep(0.45, 0.92, wave) * 0.22;

    float baseAlpha = activeStrength * (0.045 + 0.08 * vWeight);
    float pulseAlpha = activeStrength * (0.35 + 0.65 * vWeight) * (packet + trail);
    float alpha = (baseAlpha + pulseAlpha) * uOpacity * vFocusDim;

    vec3 color = vPolarity >= 0.0 ? uExcColor : uInhColor;
    color *= (1.0 + 1.15 * packet);

    if (vSel > 0.5) {
      alpha = max(alpha, (0.28 + 0.68 * packet));
      color = mix(color, uSelColor, 0.45 + 0.4 * packet);
    }

    if (alpha < 0.015) discard;
    gl_FragColor = vec4(color, clamp(alpha, 0.0, 0.95));
  }
`;

const SHELL_VERTEX = /* glsl */ `
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vNormalView = normalize(normalMatrix * normal);
    vViewDir = normalize(-mvPosition.xyz);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const SHELL_FRAGMENT = /* glsl */ `
  uniform vec3 uColor;
  uniform vec3 uGlowColor;
  uniform float uBase;
  uniform float uRim;
  uniform float uPower;
  uniform float uOpacity;
  uniform float uGlow;
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  void main() {
    float facing = abs(dot(normalize(vNormalView), normalize(vViewDir)));
    float fresnel = pow(1.0 - facing, uPower);
    vec3 baseCol = mix(uColor, uGlowColor, uGlow * 0.62);
    vec3 color = baseCol * (uBase + uRim * fresnel) + uGlowColor * uGlow * (0.09 + 0.60 * fresnel);
    float alpha = uOpacity * (uBase + fresnel) + uGlow * fresnel * 0.25;
    gl_FragColor = vec4(color, clamp(alpha, 0.0, 0.60));
  }
`;

export class Brain3DRenderer {
  constructor(rootElement) {
    this.root = rootElement;
    this.stage = rootElement.querySelector('[data-brain3d="stage"]');
    this.canvas = rootElement.querySelector('[data-brain3d="canvas"]');
    this.statusEl = rootElement.querySelector('[data-brain3d="status"]');
    this.labelLayer = rootElement.querySelector('[data-brain3d="labels"]');
    this.inspectorEl = rootElement.querySelector('[data-brain3d="inspector"]');
    this.hudStepEl = rootElement.querySelector('[data-brain3d="hud-step"]');
    this.hudPhaseEl = rootElement.querySelector('[data-brain3d="hud-phase"]');
    this.hudProbsEl = rootElement.querySelector('[data-brain3d="hud-probs"]');
    this.regionListEl = rootElement.querySelector('[data-brain3d="regions"]');

    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.ready = false;
    this.visible = true;
    this.inViewport = true;
    this.flyActive = false;
    this.agentName = '';
    this.layoutMode = 'neuropil';
    this.filterMode = 'active';
    this.focusGroup = -1;
    this.showEdges = true;
    this.showLabels = true;
    this.selectedIndex = -1;
    this.hoverIndex = -1;
    this.lastFrameAt = 0;
    this.smoothingTau = 0.24;
    this.elapsed = 0;
    this.adaptiveNormScale = 0.16;
    this.cameraTween = null;
    this.morph = null;
    this.rafId = null;
    this.clock = new THREE.Clock();
  }

  async init() {
    const [layout, gltf] = await Promise.all([
      fetch(`${ASSET_BASE}/fly_brain_neurons.json`).then((r) => {
        if (!r.ok) throw new Error(`neuron layout HTTP ${r.status}`);
        return r.json();
      }),
      new GLTFLoader().loadAsync(`${ASSET_BASE}/fly_brain_meshes.glb`),
    ]);

    this.layout = layout;
    this.count = layout.num_neurons;
    this.layoutPositions = {
      neuropil: Float32Array.from(layout.layout_neuropil),
      soma: Float32Array.from(layout.layout_soma),
    };
    this.targetAct = new Float32Array(this.count);
    this.prevTargetAct = new Float32Array(this.count);
    this.displayAct = new Float32Array(this.count);
    this.spikeAct = new Float32Array(this.count);
    this.alive = new Float32Array(this.count).fill(1);
    this.neuronRegion = layout.neuropil_idx.map((i) => regionOfNeuropil(layout.neuropil_names[i]));

    this._initRenderer();
    this._buildAnatomy(gltf.scene);
    this._buildNeurons();
    this._buildEdges();
    this._buildLabels();
    this._buildRegionList();
    this._bindPointer();
    this._observeSize();

    this.ready = true;
    this._setStatus(null);
    this._loop();
  }

  // ------------------------------------------------------------------ public API
  applyFrame(frame) {
    if (!this.ready || !frame) return;
    const brain = frame.connectome_telemetry || {};
    const policy = frame.policy_telemetry || {};
    const snapshot = frame.snapshot || {};
    this.agentName = frame.agent_name || '';

    const now = performance.now();
    if (this.lastFrameAt > 0) {
      const interval = (now - this.lastFrameAt) / 1000;
      this.smoothingTau = Math.min(0.30, Math.max(0.05, interval * 0.42));
    }
    this.lastFrameAt = now;

    this.flyActive = Boolean(brain.is_fly_active && brain.activations_b64);
    this.pointMaterial.uniforms.uFlyActive.value = this.flyActive ? 1 : 0;

    if (this.flyActive) {
      const acts = decodeBase64(brain.activations_b64);
      const mask = brain.lesion_mask_b64 ? decodeBase64(brain.lesion_mask_b64) : null;
      let maxAbs = 0.08;
      for (let i = 0; i < this.count; i += 1) {
        const prev = this.targetAct[i];
        const next = ((acts[i] / 255) * 2 - 1) * H_SCALE;
        this.prevTargetAct[i] = prev;
        this.targetAct[i] = next;
        const delta = Math.abs(next - prev);
        if (delta > this.spikeAct[i]) {
          this.spikeAct[i] = delta;
        }
        const isAlive = mask ? (mask[i] > 0 ? 1 : 0) : 1;
        this.alive[i] = isAlive;
        if (isAlive && Math.abs(next) > maxAbs) {
          maxAbs = Math.abs(next);
        }
      }
      // Smoothly adapt display ceiling so top active firing neurons always reach full contrast.
      const targetScale = Math.min(H_SCALE, Math.max(0.12, maxAbs * 0.85));
      this.adaptiveNormScale += (targetScale - this.adaptiveNormScale) * 0.25;
      this.pointMaterial.uniforms.uNormScale.value = this.adaptiveNormScale;
      this.pointGeometry.attributes.aAlive.needsUpdate = true;
      this._setStatus(null);
    } else {
      this.targetAct.fill(0);
      this.spikeAct.fill(0);
      this._setStatus(
        `${this.agentName || 'Baseline controller'} aktif. Controller ini tidak memiliki state 2.500 neuron; anatomi ditampilkan tanpa aktivasi.`
      );
    }

    this._updateRegionActivity();
    this._updateHud(snapshot, policy);
    if (this.selectedIndex >= 0) this._renderInspector();
  }

  setVisible(isVisible) {
    this.visible = Boolean(isVisible);
    if (this.visible && this.ready) {
      this._resize();
      this._loop();
    }
  }

  setLayout(mode) {
    if (!this.ready || mode === this.layoutMode || !this.layoutPositions[mode]) return;
    const from = Float32Array.from(this.pointGeometry.attributes.position.array);
    this.layoutMode = mode;
    this.morph = {
      from,
      to: this.layoutPositions[mode],
      start: performance.now(),
      duration: this.reducedMotion ? 0 : 900,
    };
  }

  setFilterMode(mode) {
    if (!this.ready) return;
    this.filterMode = mode === 'all' ? 'all' : 'active';
    if (this.filterMode === 'active') {
      this.pointMaterial.uniforms.uActThreshold.value = 0.28;
      this.pointMaterial.uniforms.uQuietAlpha.value = 0.04;
      this.edgeMaterial.uniforms.uEdgeThreshold.value = 0.36;
    } else {
      this.pointMaterial.uniforms.uActThreshold.value = 0.14;
      this.pointMaterial.uniforms.uQuietAlpha.value = 0.16;
      this.edgeMaterial.uniforms.uEdgeThreshold.value = 0.22;
    }
  }

  setFocus(group) {
    this.focusGroup = group;
    if (!this.ready) return;
    this.pointMaterial.uniforms.uFocus.value = group;
    this.edgeMaterial.uniforms.uFocus.value = group;
    for (const entry of this.focusMeshes) {
      const dimmed = group >= 0 && entry.group !== group;
      entry.material.uniforms.uOpacity.value = dimmed ? entry.baseOpacity * 0.25 : entry.baseOpacity;
      entry.material.uniforms.uRim.value = dimmed ? entry.baseRim * 0.35 : entry.baseRim;
    }
  }

  setEdgesVisible(show) {
    this.showEdges = Boolean(show);
    if (this.edgeLines) this.edgeLines.visible = this.showEdges;
  }

  setLabelsVisible(show) {
    this.showLabels = Boolean(show);
    if (this.labelLayer) this.labelLayer.hidden = !this.showLabels;
  }

  setAutoRotate(enabled) {
    if (this.controls) this.controls.autoRotate = Boolean(enabled) && !this.reducedMotion;
  }

  setView(preset) {
    const target = CAMERA_PRESETS[preset];
    if (!target || !this.camera) return;
    this.cameraTween = {
      fromPos: this.camera.position.clone(),
      toPos: target.clone(),
      fromTarget: this.controls.target.clone(),
      toTarget: CAMERA_TARGET.clone(),
      start: performance.now(),
      duration: this.reducedMotion ? 0 : 750,
    };
  }

  clearSelection() {
    this.selectedIndex = -1;
    if (this.pointMaterial) this.pointMaterial.uniforms.uSelected.value = -1;
    if (this.edgeMaterial) this.edgeMaterial.uniforms.uSelected.value = -1;
    if (this.inspectorEl) this.inspectorEl.hidden = true;
  }

  async toggleFullscreen() {
    if (document.fullscreenElement === this.stage) {
      await document.exitFullscreen();
    } else if (this.stage.requestFullscreen) {
      await this.stage.requestFullscreen();
    }
  }

  // --------------------------------------------------------------- scene build
  _initRenderer() {
    const renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(new THREE.Color('#090b10'), 1);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 2, 0.1, 200);
    this.camera.position.copy(CAMERA_PRESETS.front);

    const controls = new OrbitControls(this.camera, this.canvas);
    controls.target.copy(CAMERA_TARGET);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 5;
    controls.maxDistance = 38;
    controls.rotateSpeed = 0.7;
    controls.autoRotate = !this.reducedMotion;
    controls.autoRotateSpeed = 0.4;
    controls.addEventListener('start', () => {
      this.cameraTween = null;
    });
    this.controls = controls;

    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    // Balanced bloom: luminous aura on active spikes, synaptic packets, and lit neuropil rims.
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(512, 256), 0.65, 0.50, 0.34);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());
  }

  _fresnelMaterial({ color, base, rim, power, opacity, glowColor = '#f59e0b' }) {
    return new THREE.ShaderMaterial({
      vertexShader: SHELL_VERTEX,
      fragmentShader: SHELL_FRAGMENT,
      uniforms: {
        uColor: { value: new THREE.Color(color) },
        uGlowColor: { value: new THREE.Color(glowColor) },
        uBase: { value: base },
        uRim: { value: rim },
        uPower: { value: power },
        uOpacity: { value: opacity },
        uGlow: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.FrontSide,
      blending: THREE.NormalBlending,
    });
  }

  _buildAnatomy(gltfScene) {
    const cxSet = new Set(this.layout.cx_neuropils);
    const mbSet = new Set(this.layout.mb_neuropils);
    this.anatomyGroup = new THREE.Group();
    this.regionMaterials = {};
    this.focusMeshes = [];
    this.meshCenters = {};

    const meshes = [];
    gltfScene.traverse((obj) => {
      if (obj.isMesh) meshes.push(obj);
    });

    for (const mesh of meshes) {
      const name = mesh.name || (mesh.parent && mesh.parent.name) || '';
      if (!mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
      mesh.geometry.computeBoundingBox();
      this.meshCenters[name] = mesh.geometry.boundingBox.getCenter(new THREE.Vector3());

      let material;
      if (name === 'BRAIN_SHELL') {
        material = this._fresnelMaterial({
          color: '#93adc9',
          base: 0.028,
          rim: 0.62,
          power: 2.25,
          opacity: 0.28,
        });
        mesh.renderOrder = 3;
      } else if (cxSet.has(name) || mbSet.has(name)) {
        const isCx = cxSet.has(name);
        material = this._fresnelMaterial({
          color: isCx ? '#94bfe8' : '#85d1c5',
          glowColor: isCx ? '#f59e0b' : '#38bdf8',
          base: 0.028,
          rim: 0.68,
          power: 1.95,
          opacity: 0.22,
        });
        const group = isCx ? 0 : 1;
        const region = regionOfNeuropil(name);
        (this.regionMaterials[region] ||= []).push(material);
        this.focusMeshes.push({ material, group, baseOpacity: 0.22, baseRim: 0.68 });
        mesh.renderOrder = 2;
      } else {
        material = this._fresnelMaterial({
          color: '#526073',
          base: 0.0,
          rim: 0.18,
          power: 2.8,
          opacity: 0.10,
        });
        mesh.renderOrder = 1;
      }
      mesh.material = material;
      this.anatomyGroup.add(mesh);
    }
    this.scene.add(this.anatomyGroup);
  }

  _buildNeurons() {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(Float32Array.from(this.layoutPositions.neuropil), 3)
    );
    geometry.setAttribute('aAct', new THREE.BufferAttribute(this.displayAct, 1));
    geometry.setAttribute('aSpike', new THREE.BufferAttribute(this.spikeAct, 1));
    geometry.setAttribute('aAlive', new THREE.BufferAttribute(this.alive, 1));
    geometry.setAttribute('aGroup', new THREE.BufferAttribute(Float32Array.from(this.layout.circuit_groups), 1));
    geometry.setAttribute(
      'aIndex',
      new THREE.BufferAttribute(Float32Array.from({ length: this.count }, (_, i) => i), 1)
    );
    geometry.computeBoundingSphere();

    const material = new THREE.ShaderMaterial({
      vertexShader: POINT_VERTEX,
      fragmentShader: POINT_FRAGMENT,
      uniforms: {
        uPixelRatio: { value: this.renderer.getPixelRatio() },
        uSize: { value: 4.8 },
        uFocus: { value: -1 },
        uSelected: { value: -1 },
        uHover: { value: -1 },
        uFlyActive: { value: 0 },
        uActThreshold: { value: 0.28 },
        uQuietAlpha: { value: 0.04 },
        uNormScale: { value: this.adaptiveNormScale },
        uNeg: { value: new THREE.Color('#0ea5e9') },
        uNegHot: { value: new THREE.Color('#7dd3fc') },
        uPos: { value: new THREE.Color('#f97316') },
        uPosHot: { value: new THREE.Color('#fde047') },
        uMid: { value: new THREE.Color('#475569') },
        uDead: { value: new THREE.Color('#ef4444') },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });

    this.pointGeometry = geometry;
    this.pointMaterial = material;
    this.points = new THREE.Points(geometry, material);
    this.points.renderOrder = 5;
    this.scene.add(this.points);
  }

  _buildEdges() {
    const { pre, post, weight_norm: weights, polarity } = this.layout.edges;
    const groups = this.layout.circuit_groups;
    const edgeCount = pre.length;
    const positions = new Float32Array(edgeCount * 6);
    const tValues = new Float32Array(edgeCount * 2);
    const acts = new Float32Array(edgeCount * 2);
    const weightAttr = new Float32Array(edgeCount * 2);
    const polarityAttr = new Float32Array(edgeCount * 2);
    const groupAttr = new Float32Array(edgeCount * 2);
    const preAttr = new Float32Array(edgeCount * 2);
    const postAttr = new Float32Array(edgeCount * 2);

    for (let e = 0; e < edgeCount; e += 1) {
      tValues[e * 2] = 0;
      tValues[e * 2 + 1] = 1;
      weightAttr[e * 2] = weights[e];
      weightAttr[e * 2 + 1] = weights[e];
      const pol = polarity ? polarity[e] : 1;
      polarityAttr[e * 2] = pol;
      polarityAttr[e * 2 + 1] = pol;
      const g = groups[pre[e]];
      groupAttr[e * 2] = g;
      groupAttr[e * 2 + 1] = g;
      preAttr[e * 2] = pre[e];
      preAttr[e * 2 + 1] = pre[e];
      postAttr[e * 2] = post[e];
      postAttr[e * 2 + 1] = post[e];
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aT', new THREE.BufferAttribute(tValues, 1));
    geometry.setAttribute('aAct', new THREE.BufferAttribute(acts, 1));
    geometry.setAttribute('aWeight', new THREE.BufferAttribute(weightAttr, 1));
    geometry.setAttribute('aPolarity', new THREE.BufferAttribute(polarityAttr, 1));
    geometry.setAttribute('aGroup', new THREE.BufferAttribute(groupAttr, 1));
    geometry.setAttribute('aPre', new THREE.BufferAttribute(preAttr, 1));
    geometry.setAttribute('aPost', new THREE.BufferAttribute(postAttr, 1));

    this.edgeMaterial = new THREE.ShaderMaterial({
      vertexShader: EDGE_VERTEX,
      fragmentShader: EDGE_FRAGMENT,
      uniforms: {
        uTime: { value: 0 },
        uOpacity: { value: 0.85 },
        uEdgeThreshold: { value: 0.36 },
        uSelected: { value: -1 },
        uFocus: { value: -1 },
        uExcColor: { value: new THREE.Color('#ff8c2e') },
        uInhColor: { value: new THREE.Color('#38bdf8') },
        uSelColor: { value: new THREE.Color('#ffffff') },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });

    this.edgeGeometry = geometry;
    this.edgeLines = new THREE.LineSegments(geometry, this.edgeMaterial);
    this.edgeLines.renderOrder = 4;
    this.edgeLines.frustumCulled = false;
    this.scene.add(this.edgeLines);
    this._syncEdgePositions();
  }

  _syncEdgePositions() {
    const nodePos = this.pointGeometry.attributes.position.array;
    const edgePos = this.edgeGeometry.attributes.position.array;
    const { pre, post } = this.layout.edges;
    for (let e = 0; e < pre.length; e += 1) {
      const a = pre[e] * 3;
      const b = post[e] * 3;
      const o = e * 6;
      edgePos[o] = nodePos[a];
      edgePos[o + 1] = nodePos[a + 1];
      edgePos[o + 2] = nodePos[a + 2];
      edgePos[o + 3] = nodePos[b];
      edgePos[o + 4] = nodePos[b + 1];
      edgePos[o + 5] = nodePos[b + 2];
    }
    this.edgeGeometry.attributes.position.needsUpdate = true;
  }

  _buildLabels() {
    this.labels = [];
    if (!this.labelLayer) return;
    this.labelLayer.textContent = '';
    for (const [meshName, text] of MESH_LABELS) {
      const center = this.meshCenters[meshName];
      if (!center) continue;
      const el = document.createElement('span');
      el.className = 'brain3d-label';
      el.textContent = text;
      this.labelLayer.appendChild(el);
      this.labels.push({ el, position: center.clone() });
    }
  }

  _buildRegionList() {
    this.regionRows = {};
    if (!this.regionListEl) return;
    this.regionListEl.textContent = '';
    const counts = {};
    for (const region of this.neuronRegion) counts[region] = (counts[region] || 0) + 1;

    REGION_KEYS.forEach((key, idx) => {
      const row = document.createElement('div');
      row.className = 'region-row';
      row.dataset.tooltip = REGION_TOOLTIPS[key] || REGION_LABELS[key];
      row.dataset.tooltipPos = idx % 2 === 1 ? 'left' : 'top';
      row.innerHTML = `
        <span class="region-name tooltip-trigger">${REGION_LABELS[key]}</span>
        <span class="region-count mono-value">n=${counts[key] || 0}</span>
        <span class="region-track"><span class="region-fill"></span></span>
        <span class="region-value mono-value">-</span>`;
      this.regionListEl.appendChild(row);
      this.regionRows[key] = {
        fill: row.querySelector('.region-fill'),
        value: row.querySelector('.region-value'),
      };
    });
  }

  // --------------------------------------------------------------- per-frame
  _updateRegionActivity() {
    const sums = {};
    const counts = {};
    for (let i = 0; i < this.count; i += 1) {
      if (!this.alive[i]) continue;
      const region = this.neuronRegion[i];
      sums[region] = (sums[region] || 0) + Math.abs(this.targetAct[i]);
      counts[region] = (counts[region] || 0) + 1;
    }

    for (const key of REGION_KEYS) {
      const mean = counts[key] ? sums[key] / counts[key] : 0;
      const glow = this.flyActive ? Math.min(mean / 0.14, 1) : 0;
      for (const material of this.regionMaterials[key] || []) {
        material.userData.targetGlow = glow;
      }
      const row = this.regionRows[key];
      if (row) {
        row.fill.style.transform = `scaleX(${Math.min(mean / H_SCALE, 1).toFixed(4)})`;
        row.value.textContent = this.flyActive ? mean.toFixed(4) : '-';
      }
    }
  }

  _updateHud(snapshot, policy) {
    const phaseNames = ['NS Straight', 'NS Turn', 'EW Straight', 'EW Turn'];
    if (this.hudStepEl) {
      const stepLabel = snapshot.continuous_mode
        ? `Step ${snapshot.step ?? '-'}`
        : `Step ${snapshot.step ?? '-'}/${snapshot.max_steps ?? '-'}`;
      this.hudStepEl.textContent = `${stepLabel} · ${snapshot.profile_name || '-'}`;
    }
    const chosen = Number(policy.chosen_action ?? 0);
    if (this.hudPhaseEl) {
      const value = policy.value_estimate;
      const valueText = value === null || value === undefined ? '' : ` · V ${Number(value).toFixed(2)}`;
      this.hudPhaseEl.textContent = `a_t = ${phaseNames[chosen] || chosen}${valueText}`;
    }
    if (this.hudProbsEl) {
      const probs = policy.action_probabilities || [];
      if (this.hudProbsEl.childElementCount !== 4) {
        this.hudProbsEl.innerHTML = phaseNames
          .map(
            (name) =>
              `<div class="hud-prob"><span>${name}</span><span class="hud-prob-track"><span class="hud-prob-fill"></span></span><span class="hud-prob-val mono-value">-</span></div>`
          )
          .join('');
      }
      Array.from(this.hudProbsEl.children).forEach((row, i) => {
        const p = Math.min(Math.max(Number(probs[i] || 0), 0), 1);
        row.classList.toggle('is-selected', i === chosen);
        row.querySelector('.hud-prob-fill').style.transform = `scaleX(${p.toFixed(4)})`;
        row.querySelector('.hud-prob-val').textContent = `${Math.round(p * 100)}%`;
      });
    }
  }

  _tick(dt, now) {
    const k = this.reducedMotion ? 1 : 1 - Math.exp(-dt / this.smoothingTau);
    const spikeDecay = this.reducedMotion ? 0 : Math.exp(-dt / 0.35);
    const act = this.displayAct;
    const spike = this.spikeAct;
    for (let i = 0; i < this.count; i += 1) {
      act[i] += (this.targetAct[i] - act[i]) * k;
      spike[i] *= spikeDecay;
    }
    this.pointGeometry.attributes.aAct.needsUpdate = true;
    this.pointGeometry.attributes.aSpike.needsUpdate = true;

    if (this.showEdges) {
      const edgeAct = this.edgeGeometry.attributes.aAct.array;
      const { pre, post } = this.layout.edges;
      const invNorm = 1 / Math.max(this.adaptiveNormScale, 0.05);
      for (let e = 0; e < pre.length; e += 1) {
        const preIdx = pre[e];
        const postIdx = post[e];
        const alivePair = this.alive[preIdx] * this.alive[postIdx];
        const v = alivePair * Math.min(Math.abs(act[preIdx]) * invNorm, 1);
        edgeAct[e * 2] = v;
        edgeAct[e * 2 + 1] = v;
      }
      this.edgeGeometry.attributes.aAct.needsUpdate = true;
      if (!this.reducedMotion) this.edgeMaterial.uniforms.uTime.value = this.elapsed;
    }

    const glowK = 1 - Math.exp(-dt / 0.25);
    for (const key of REGION_KEYS) {
      for (const material of this.regionMaterials[key] || []) {
        const target = material.userData.targetGlow || 0;
        material.uniforms.uGlow.value += (target - material.uniforms.uGlow.value) * glowK;
      }
    }

    if (this.morph) {
      const { from, to, start, duration } = this.morph;
      const t = duration > 0 ? Math.min((now - start) / duration, 1) : 1;
      const e = easeOutQuart(t);
      const pos = this.pointGeometry.attributes.position.array;
      for (let i = 0; i < pos.length; i += 1) pos[i] = from[i] + (to[i] - from[i]) * e;
      this.pointGeometry.attributes.position.needsUpdate = true;
      this._syncEdgePositions();
      if (t >= 1) {
        this.morph = null;
        this.pointGeometry.computeBoundingSphere();
      }
    }

    if (this.cameraTween) {
      const { fromPos, toPos, fromTarget, toTarget, start, duration } = this.cameraTween;
      const t = duration > 0 ? Math.min((now - start) / duration, 1) : 1;
      const e = easeOutQuart(t);
      this.camera.position.lerpVectors(fromPos, toPos, e);
      this.controls.target.lerpVectors(fromTarget, toTarget, e);
      if (t >= 1) this.cameraTween = null;
    }

    this.controls.update();
    this._updateLabels();
  }

  _updateLabels() {
    if (!this.showLabels || !this.labels.length) return;
    const width = this.stage.clientWidth;
    const height = this.stage.clientHeight;
    const projected = new THREE.Vector3();
    for (const label of this.labels) {
      projected.copy(label.position).project(this.camera);
      const visible = projected.z < 1 && Math.abs(projected.x) < 1.05 && Math.abs(projected.y) < 1.05;
      label.el.style.opacity = visible ? '' : '0';
      const x = (projected.x * 0.5 + 0.5) * width;
      const y = (-projected.y * 0.5 + 0.5) * height;
      label.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    }
  }

  _loop() {
    if (this.rafId !== null) return;
    const step = () => {
      this.rafId = null;
      if (!this.visible || !this.inViewport || document.visibilityState !== 'visible') return;
      const dt = Math.min(this.clock.getDelta(), 0.1);
      this.elapsed += dt;
      this._tick(dt, performance.now());
      this.composer.render();
      this.rafId = requestAnimationFrame(step);
    };
    this.clock.getDelta();
    this.rafId = requestAnimationFrame(step);
  }

  // ------------------------------------------------------------- interaction
  _bindPointer() {
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.08;
    const ndc = new THREE.Vector2();
    let downAt = null;
    let pendingMove = null;

    const pick = (clientX, clientY) => {
      const rect = this.canvas.getBoundingClientRect();
      ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(ndc, this.camera);
      const hits = raycaster.intersectObject(this.points, false);
      let best = -1;
      let bestDist = Infinity;
      for (const hit of hits) {
        const group = this.layout.circuit_groups[hit.index];
        if (this.focusGroup >= 0 && group !== this.focusGroup) continue;
        if (hit.distanceToRay < bestDist) {
          bestDist = hit.distanceToRay;
          best = hit.index;
        }
      }
      return best;
    };

    this.canvas.addEventListener('pointerdown', (event) => {
      downAt = { x: event.clientX, y: event.clientY };
    });

    this.canvas.addEventListener('pointermove', (event) => {
      if (downAt) return;
      if (pendingMove) {
        pendingMove = { x: event.clientX, y: event.clientY };
        return;
      }
      pendingMove = { x: event.clientX, y: event.clientY };
      requestAnimationFrame(() => {
        const idx = pick(pendingMove.x, pendingMove.y);
        pendingMove = null;
        this.hoverIndex = idx;
        this.pointMaterial.uniforms.uHover.value = idx;
        this.canvas.style.cursor = idx >= 0 ? 'pointer' : '';
      });
    });

    this.canvas.addEventListener('pointerleave', () => {
      this.hoverIndex = -1;
      this.pointMaterial.uniforms.uHover.value = -1;
      this.canvas.style.cursor = '';
    });

    this.canvas.addEventListener('pointerup', (event) => {
      if (!downAt) return;
      const moved = Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y);
      downAt = null;
      if (moved > 4) return;
      const idx = pick(event.clientX, event.clientY);
      if (idx < 0) {
        this.clearSelection();
        return;
      }
      this.selectedIndex = idx;
      this.pointMaterial.uniforms.uSelected.value = idx;
      this.edgeMaterial.uniforms.uSelected.value = idx;
      this._renderInspector();
    });

    document.addEventListener('fullscreenchange', () => {
      this.root.classList.toggle('is-fullscreen', document.fullscreenElement === this.stage);
      requestAnimationFrame(() => this._resize());
    });
  }

  _renderInspector() {
    const el = this.inspectorEl;
    const i = this.selectedIndex;
    if (!el || i < 0) return;
    const L = this.layout;
    const h = this.targetAct[i];
    const fields = {
      root: L.root_ids[i],
      type: L.cell_types[i] || 'unlabelled',
      group: L.circuit_groups[i] === 0 ? 'CX' : 'MB',
      neuropil: L.neuropil_names[L.neuropil_idx[i]],
      nt: L.nt_names[L.nt_idx[i]],
      index: String(i),
      h: this.flyActive ? (h >= 0 ? '+' : '') + h.toFixed(3) : '-',
      state: !this.flyActive ? 'no connectome state' : this.alive[i] ? 'active' : 'lesioned',
    };
    for (const [key, value] of Object.entries(fields)) {
      const target = el.querySelector(`[data-field="${key}"]`);
      if (target) target.textContent = value;
    }
    el.hidden = false;
  }

  _observeSize() {
    this._resize();
    new ResizeObserver(() => this._resize()).observe(this.stage);
    new IntersectionObserver((entries) => {
      this.inViewport = entries.some((entry) => entry.isIntersecting);
      if (this.inViewport) this._loop();
    }).observe(this.stage);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') this._loop();
    });
  }

  _resize() {
    if (!this.renderer) return;
    const width = Math.max(this.stage.clientWidth, 1);
    const height = Math.max(this.stage.clientHeight, 1);
    this.renderer.setSize(width, height, false);
    this.composer.setSize(width, height);
    this.bloomPass.setSize(width, height);
    this.camera.aspect = width / height;
    // Keep the whole brain (≈10 units wide) in frame on narrow stages.
    this.camera.fov = width / height < 1.4 ? 38 : 30;
    this.camera.updateProjectionMatrix();
    if (this.pointMaterial) this.pointMaterial.uniforms.uPixelRatio.value = this.renderer.getPixelRatio();
  }

  _setStatus(message) {
    if (!this.statusEl) return;
    this.statusEl.hidden = !message;
    if (message) this.statusEl.textContent = message;
  }
}

export function webglAvailable() {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

function decodeBase64(b64) {
  const binary = atob(b64);
  const output = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) output[i] = binary.charCodeAt(i);
  return output;
}
