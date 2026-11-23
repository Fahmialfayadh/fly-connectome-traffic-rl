/**
 * Intersection3DRenderer — Embedded Three.js 4-way 3D intersection panel
 * integrated into the main Fly Control Traffic web workbench.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { IntersectionWorld } from './sim3d/world.js';
import { VehicleManager3D } from './sim3d/vehicles.js';
import { TrafficSignals3D } from './sim3d/signals.js';

const LANE_BADGE_WORLD_POS = [
  new THREE.Vector3(-4.5, 0.4, -17.0), // 0: N_STR
  new THREE.Vector3(-1.5, 0.4, -17.0), // 1: N_TRN
  new THREE.Vector3(4.5, 0.4, 17.0),   // 2: S_STR
  new THREE.Vector3(1.5, 0.4, 17.0),   // 3: S_TRN
  new THREE.Vector3(17.0, 0.4, -4.5),  // 4: E_STR
  new THREE.Vector3(17.0, 0.4, -1.5),  // 5: E_TRN
  new THREE.Vector3(-17.0, 0.4, 4.5),  // 6: W_STR
  new THREE.Vector3(-17.0, 0.4, 1.5),  // 7: W_TRN
];

const CAMERA_PRESETS = {
  top: { pos: [0, 48, 0.1], target: [0, 0, 0] },
  angle: { pos: [26, 28, 26], target: [0, 0, 0] },
  street: { pos: [-3.0, 2.5, -24], target: [-1.5, 1.0, 0] },
};

const REFERENCE_TOP_DIST = 48.0;

export class Intersection3DRenderer {
  constructor(stageElement, options = {}) {
    this.stage = stageElement;
    this.options = options || {};
    this.onLaneClick = null;
    this.onZoomChange = null;

    this.visible = true;
    this.activePreset = 'top';
    this.latestFrame = null;
    this.needsImmediateSync = false;
    this.animationFrameId = null;
    this.alertHideTimer = null;
    this.cameraTweenRaf = null;
    this.pointerDownPos = null;
    this.cinematicActive = false;
    this.cinematicTime = 0;
    this.cinematicBaseAngle = Math.PI * 0.25;
    this.cinematicBlend = 1.0;
    this.cinematicStartPos = new THREE.Vector3();
    this.cinematicStartTarget = new THREE.Vector3();

    this.host = stageElement.querySelector('[data-sim3d="canvas-host"]') || stageElement;
    this.badgeElements = Array.from({ length: 8 }, (_, i) =>
      stageElement.querySelector(`[data-sim3d-qbadge="${i}"]`)
    );
    this.alertEl = stageElement.querySelector('[data-sim3d="collision-alert"]');
    this.alertTitleEl = stageElement.querySelector('[data-sim3d="alert-title"]');
    this.alertSubtitleEl = stageElement.querySelector('[data-sim3d="alert-subtitle"]');
    this.camButtons = Array.from(stageElement.querySelectorAll('[data-sim3d-cam]'));
    this.fullscreenBtn = stageElement.querySelector('[data-sim3d="fullscreen"]');

    this._initScene();
    this._bindInteractions();
    this._bindResizeObserver();
  }

  _initScene() {
    const w = Math.max(this.host.clientWidth || 700, 200);
    const h = Math.max(this.host.clientHeight || 580, 200);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(w, h, false);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.setClearColor(0x181c24);
    this.renderer.domElement.className = 'sim3d-canvas';
    this.host.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x181c24, 0.004);

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.5, 350);
    this.camera.position.set(...CAMERA_PRESETS.top.pos);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(...CAMERA_PRESETS.top.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2.05;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 140;
    this.controls.update();

    this.world = new IntersectionWorld(this.scene);
    this.vehicles = new VehicleManager3D(this.scene);
    this.signals = new TrafficSignals3D(this.scene);

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.tempV = new THREE.Vector3();
    this.clock = new THREE.Clock();
  }

  _bindInteractions() {
    const dom = this.renderer.domElement;

    dom.addEventListener('pointerdown', (event) => {
      this.pointerDownPos = { x: event.clientX, y: event.clientY };
    });

    dom.addEventListener('mousemove', (event) => {
      const rect = dom.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const hits = this.raycaster.intersectObjects(this.world.laneHitMeshes, false);
      if (hits.length > 0) {
        this.world.highlightLane(hits[0].object.userData.laneId);
        dom.style.cursor = 'pointer';
      } else {
        this.world.clearLaneHighlight();
        dom.style.cursor = 'grab';
      }
    });

    dom.addEventListener('mouseleave', () => {
      this.world.clearLaneHighlight();
      dom.style.cursor = 'default';
    });

    dom.addEventListener('click', (event) => {
      if (this.pointerDownPos) {
        const dist = Math.hypot(
          event.clientX - this.pointerDownPos.x,
          event.clientY - this.pointerDownPos.y
        );
        this.pointerDownPos = null;
        if (dist > 5) return;
      }

      const rect = dom.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const hits = this.raycaster.intersectObjects(this.world.laneHitMeshes, false);
      if (hits.length > 0 && typeof this.onLaneClick === 'function') {
        this.onLaneClick(hits[0].object.userData.laneId);
      }
    });

    this.controls.addEventListener('start', () => {
      if (this.cinematicActive) {
        this._stopCinematic();
      }
      if (this.cameraTweenRaf !== null) {
        cancelAnimationFrame(this.cameraTweenRaf);
        this.cameraTweenRaf = null;
      }
    });

    this.controls.addEventListener('change', () => {
      if (this.visible && typeof this.onZoomChange === 'function') {
        this.onZoomChange(this.getZoomRatio());
      }
    });

    for (const btn of this.camButtons) {
      btn.addEventListener('click', () => {
        if (btn.dataset.sim3dCam === 'cinematic') {
          this.setCinematicMode(!this.cinematicActive);
        } else {
          this.setCameraPreset(btn.dataset.sim3dCam);
        }
      });
    }

    if (this.fullscreenBtn) {
      this.fullscreenBtn.addEventListener('click', () => {
        this.toggleFullscreen();
      });
      document.addEventListener('fullscreenchange', () => {
        const isFs = document.fullscreenElement === this.stage;
        this.fullscreenBtn.textContent = isFs ? 'Exit Fullscreen' : 'Fullscreen';
        this._resize();
      });
    }
  }

  _bindResizeObserver() {
    if (typeof ResizeObserver === 'function') {
      this.resizeObserver = new ResizeObserver(() => {
        this._resize();
      });
      this.resizeObserver.observe(this.stage);
    } else {
      window.addEventListener('resize', () => this._resize());
    }
  }

  _resize() {
    if (!this.visible) return;
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (w <= 0 || h <= 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  setVisible(visible) {
    const next = Boolean(visible);
    if (next === this.visible && (next ? this.animationFrameId !== null : true)) {
      if (next) this._resize();
      return;
    }
    this.visible = next;

    if (this.visible) {
      this._resize();
      this.clock.getDelta();
      if (this.needsImmediateSync && this.latestFrame) {
        this.vehicles.syncImmediate(this.latestFrame);
        this.signals.update(this.latestFrame.snapshot || this.latestFrame);
        this.needsImmediateSync = false;
      }
      if (typeof this.onZoomChange === 'function') {
        this.onZoomChange(this.getZoomRatio());
      }
      this._startLoop();
    } else {
      this.needsImmediateSync = true;
      this._stopLoop();
    }
  }

  _startLoop() {
    if (this.animationFrameId !== null) return;
    const tick = () => {
      if (!this.visible) {
        this.animationFrameId = null;
        return;
      }
      const dt = Math.min(this.clock.getDelta(), 0.05);
      if (this.cinematicActive) {
        this._updateCinematicCamera(dt);
      }
      this.controls.update();
      this.vehicles.animate(dt);
      this._updateBadgePositions();
      this.renderer.render(this.scene, this.camera);
      this.animationFrameId = requestAnimationFrame(tick);
    };
    this.animationFrameId = requestAnimationFrame(tick);
  }

  _stopLoop() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  applyBackendFrame(framePayload) {
    if (!framePayload) return;
    const snapshot = framePayload.snapshot ? framePayload.snapshot : framePayload;
    if (!snapshot || typeof snapshot !== 'object') return;

    this.latestFrame = framePayload;

    if (this.visible) {
      if (this.needsImmediateSync) {
        this.vehicles.syncImmediate(framePayload);
        this.needsImmediateSync = false;
      } else {
        this.vehicles.updateFromFrame(framePayload);
      }
      this.signals.update(snapshot);
    } else {
      this.needsImmediateSync = true;
    }

    this._updateApproachOverlays(snapshot);
    this._updateBadgesText(snapshot);
    this._updateCollisionBanner(snapshot);
  }

  _updateApproachOverlays(snapshot) {
    const stage = snapshot.signal_stage || 'GREEN';
    const greenLanes = new Set(snapshot.phase_green_lanes || []);
    const durations = snapshot.lane_signal_durations_s || [];

    for (let lane = 0; lane < 8; lane++) {
      const isActive = greenLanes.has(lane);
      const rLens = this.stage.querySelector(`[data-sim3d-lens="${lane}-r"]`);
      const yLens = this.stage.querySelector(`[data-sim3d-lens="${lane}-y"]`);
      const gLens = this.stage.querySelector(`[data-sim3d-lens="${lane}-g"]`);
      const timerPill = this.stage.querySelector(`[data-sim3d-timer="${lane}"]`);
      if (!rLens || !yLens || !gLens || !timerPill) continue;

      rLens.classList.remove('active');
      yLens.classList.remove('active');
      gLens.classList.remove('active');
      timerPill.className = 'sim3d-timer-pill mono-value';

      const dur = Math.round(Number(durations[lane] || 0));
      timerPill.textContent = `${dur}s`;

      if (isActive && stage === 'GREEN') {
        gLens.classList.add('active');
        timerPill.classList.add('green');
      } else if (isActive && stage === 'YELLOW') {
        yLens.classList.add('active');
        timerPill.classList.add('yellow');
      } else {
        rLens.classList.add('active');
      }
    }
  }

  _updateBadgesText(snapshot) {
    const queues = snapshot.lane_queues || [];
    for (let i = 0; i < 8; i++) {
      const el = this.badgeElements[i];
      if (!el) continue;
      const count = Number(queues[i] || 0);
      el.textContent = String(count);
      el.classList.toggle('is-high', count >= 12);
    }
  }

  _updateCollisionBanner(snapshot) {
    if (!this.alertEl) return;
    const events = snapshot.step_collision_events;
    if (Array.isArray(events) && events.length > 0) {
      const ev = events[0];
      if (this.alertTitleEl) {
        this.alertTitleEl.textContent = `COLLISION: ${ev.kind || 'CONFLICT'}`;
      }
      if (this.alertSubtitleEl) {
        this.alertSubtitleEl.textContent = `Culprit lane ${ev.culprit_lane} vs Victim lane ${ev.victim_lane}`;
      }
      this.alertEl.classList.remove('is-hidden');
      if (this.alertHideTimer) clearTimeout(this.alertHideTimer);
      this.alertHideTimer = setTimeout(() => {
        if (this.alertEl) this.alertEl.classList.add('is-hidden');
        this.alertHideTimer = null;
      }, 3200);
    }
  }

  _updateBadgePositions() {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (width <= 0 || height <= 0) return;
    const widthHalf = width / 2;
    const heightHalf = height / 2;

    for (let i = 0; i < 8; i++) {
      const badgeEl = this.badgeElements[i];
      if (!badgeEl) continue;

      this.tempV.copy(LANE_BADGE_WORLD_POS[i]);
      this.tempV.project(this.camera);

      const x = this.tempV.x * widthHalf + widthHalf;
      const y = -(this.tempV.y * heightHalf) + heightHalf;

      if (this.tempV.z > 1.0 || x < 16 || x > width - 16 || y < 16 || y > height - 16) {
        badgeEl.style.display = 'none';
        continue;
      }

      badgeEl.style.display = 'flex';
      badgeEl.style.left = `${x}px`;
      badgeEl.style.top = `${y}px`;
    }
  }

  setCameraPreset(presetKey) {
    this.cinematicActive = false;
    const preset = CAMERA_PRESETS[presetKey] || CAMERA_PRESETS.top;
    this.activePreset = CAMERA_PRESETS[presetKey] ? presetKey : 'top';

    this._syncPresetButtons(this.activePreset);

    this._animateCamera(
      new THREE.Vector3(...preset.pos),
      new THREE.Vector3(...preset.target)
    );
  }

  setCinematicMode(enabled) {
    if (!enabled) {
      this._stopCinematic();
      return;
    }
    if (this.cameraTweenRaf !== null) {
      cancelAnimationFrame(this.cameraTweenRaf);
      this.cameraTweenRaf = null;
    }
    this.cinematicActive = true;
    this.activePreset = 'cinematic';
    this.cinematicTime = 0;
    this.cinematicBlend = 0.0;
    const horizDist = Math.hypot(this.camera.position.x, this.camera.position.z);
    this.cinematicBaseAngle =
      horizDist > 1.0
        ? Math.atan2(this.camera.position.z, this.camera.position.x)
        : Math.PI * 0.25;
    this.cinematicStartPos.copy(this.camera.position);
    this.cinematicStartTarget.copy(this.controls.target);
    this._syncPresetButtons('cinematic');
  }

  _stopCinematic() {
    this.cinematicActive = false;
    this.activePreset = null;
    this._syncPresetButtons(null);
  }

  _syncPresetButtons(activeKey) {
    for (const btn of this.camButtons) {
      const active = btn.dataset.sim3dCam === activeKey;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    }
  }

  _updateCinematicCamera(dt) {
    this.cinematicTime += dt;
    const t = this.cinematicTime;
    const angle = this.cinematicBaseAngle + t * 0.22;
    const radius = 32.0 + 7.0 * Math.sin(t * 0.35);
    const height = 17.5 + 7.5 * Math.sin(t * 0.28 + 0.6);

    const desiredPos = new THREE.Vector3(
      Math.cos(angle) * radius,
      height,
      Math.sin(angle) * radius
    );
    const desiredTarget = new THREE.Vector3(
      2.0 * Math.cos(t * 0.19),
      1.0,
      2.0 * Math.sin(t * 0.23)
    );

    if (this.cinematicBlend < 1.0) {
      this.cinematicBlend = Math.min(1.0, this.cinematicBlend + dt / 0.9);
      const b = this.cinematicBlend;
      const ease = b < 0.5 ? 2 * b * b : 1 - Math.pow(-2 * b + 2, 2) / 2;
      this.camera.position.lerpVectors(this.cinematicStartPos, desiredPos, ease);
      this.controls.target.lerpVectors(this.cinematicStartTarget, desiredTarget, ease);
    } else {
      this.camera.position.copy(desiredPos);
      this.controls.target.copy(desiredTarget);
    }
  }

  _animateCamera(endPos, endTarget) {
    if (this.cameraTweenRaf !== null) {
      cancelAnimationFrame(this.cameraTweenRaf);
      this.cameraTweenRaf = null;
    }
    const startPos = this.camera.position.clone();
    const startTarget = this.controls.target.clone();
    const duration = 520;
    const startTime = performance.now();

    const step = (now) => {
      const t = Math.min(1, (now - startTime) / duration);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      this.camera.position.lerpVectors(startPos, endPos, ease);
      this.controls.target.lerpVectors(startTarget, endTarget, ease);
      this.controls.update();

      if (typeof this.onZoomChange === 'function') {
        this.onZoomChange(this.getZoomRatio());
      }
      if (t < 1) {
        this.cameraTweenRaf = requestAnimationFrame(step);
      } else {
        this.cameraTweenRaf = null;
      }
    };
    this.cameraTweenRaf = requestAnimationFrame(step);
  }

  getZoomRatio() {
    const dist = this.camera.position.distanceTo(this.controls.target);
    return Math.min(3.0, Math.max(0.35, REFERENCE_TOP_DIST / Math.max(dist, 1)));
  }

  zoomBy(factor) {
    const offset = this.camera.position.clone().sub(this.controls.target);
    const currentDist = offset.length();
    const nextDist = Math.min(
      this.controls.maxDistance,
      Math.max(this.controls.minDistance, currentDist * factor)
    );
    if (currentDist > 1e-4) {
      offset.multiplyScalar(nextDist / currentDist);
      this.camera.position.copy(this.controls.target).add(offset);
      this.controls.update();
    }
    if (typeof this.onZoomChange === 'function') {
      this.onZoomChange(this.getZoomRatio());
    }
  }

  resetZoom() {
    this.setCameraPreset(this.activePreset || 'top');
  }

  toggleFullscreen() {
    if (!document.fullscreenElement) {
      if (typeof this.stage.requestFullscreen === 'function') {
        this.stage.requestFullscreen().catch(() => {});
      }
    } else if (typeof document.exitFullscreen === 'function') {
      document.exitFullscreen().catch(() => {});
    }
  }
}
