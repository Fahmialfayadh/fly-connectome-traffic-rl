/**
 * main.js — Main 3D simulation controller.
 * Connects Three.js world, vehicles, signals, real-time WebSocket telemetry,
 * raycasting for lane clicks (spawn vehicles), and camera presets.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { IntersectionWorld } from '/sim3d/world.js';
import { VehicleManager3D } from '/sim3d/vehicles.js';
import { TrafficSignals3D } from '/sim3d/signals.js';

/* ======== Scene & Renderer ======== */
const container = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.setClearColor(0x181c24);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x181c24, 0.004);

const camera = new THREE.PerspectiveCamera(
  45,
  container.clientWidth / container.clientHeight,
  0.5,
  350
);
// Default Top-down 3D overview looking squarely at the intersection
camera.position.set(0, 48, 0.1);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI / 2.05;
controls.minDistance = 6;
controls.maxDistance = 140;
controls.update();

/* ======== Instantiate 3D Subsystems ======== */
const world = new IntersectionWorld(scene);
const vehicles = new VehicleManager3D(scene);
const signals = new TrafficSignals3D(scene);

/* ======== 3D Coordinates for 8 RHT Lane Queue Badges ======== */
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

const badgeElements = Array.from({ length: 8 }, (_, i) =>
  document.getElementById(`qbadge-${i}`)
);

/* ======== WebSocket Connection to Real Python Simulation ======== */
let socket = null;
let isPaused = false;
let alertHideTimer = null;

const connPill = document.getElementById('conn-pill');
const btnPlayPause = document.getElementById('btn-sim-play-pause');
const btnStep = document.getElementById('btn-sim-step');
const btnReset = document.getElementById('btn-sim-reset');

function connectSimulationWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/simulation`;

  connPill.textContent = 'Connecting...';
  connPill.className = 'conn-pill connecting';

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    connPill.textContent = 'Python Env Connected';
    connPill.className = 'conn-pill connected';
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'simulation_frame' && data.snapshot) {
        handleSimulationFrame(data);
      } else if (data.lane_queues) {
        handleSimulationFrame({ snapshot: data });
      }
    } catch (err) {
      console.error('Failed to parse frame:', err);
    }
  };

  socket.onclose = () => {
    connPill.textContent = 'Offline (Reconnecting...)';
    connPill.className = 'conn-pill offline';
    setTimeout(connectSimulationWebSocket, 2000);
  };

  socket.onerror = () => {
    socket.close();
  };
}

function sendSimulationCommand(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

connectSimulationWebSocket();

/* ======== Frame Handler ======== */
function handleSimulationFrame(frame) {
  const snapshot = frame.snapshot;
  if (!snapshot) return;

  if (frame.running !== undefined) {
    isPaused = !frame.running;
    btnPlayPause.textContent = isPaused ? 'Play' : 'Pause';
  }

  // 1. Update 3D vehicles and signals with full frame (includes steps_per_second)
  vehicles.updateFromFrame(frame);
  signals.update(snapshot);

  // 2. Update Overlay Approach Signals and Timers
  updateApproachOverlays(snapshot);

  // 3. Update Queue Badges Text
  const queues = snapshot.lane_queues || [];
  for (let i = 0; i < 8; i++) {
    if (badgeElements[i]) {
      badgeElements[i].textContent = queues[i] || 0;
    }
  }

  // 4. Update Footer Meta Readouts
  document.getElementById('meta-step').textContent = snapshot.step || 0;
  document.getElementById('meta-phase').textContent = `${snapshot.phase_label || ''} · ${snapshot.signal_stage || ''}`;
  document.getElementById('meta-queue').textContent = `${snapshot.total_queue || 0} veh`;
  document.getElementById('meta-throughput').textContent = `${snapshot.total_throughput || 0} veh`;
  document.getElementById('meta-collisions').textContent = snapshot.total_collisions || 0;

  // 5. Collision Alert Banner
  const collisionAlert = document.getElementById('collision-alert');
  if (snapshot.step_collision_events && snapshot.step_collision_events.length > 0) {
    const ev = snapshot.step_collision_events[0];
    document.getElementById('alert-title').textContent = `COLLISION: ${ev.kind || 'CONFLICT'}`;
    document.getElementById('alert-subtitle').textContent = `Culprit lane ${ev.culprit_lane} vs Victim lane ${ev.victim_lane}`;
    collisionAlert.classList.remove('hidden');
    if (alertHideTimer) clearTimeout(alertHideTimer);
    alertHideTimer = setTimeout(() => {
      collisionAlert.classList.add('hidden');
      alertHideTimer = null;
    }, 3500);
  }
}

/* ======== Approach Overlays Update ======== */
function updateApproachOverlays(snapshot) {
  const stage = snapshot.signal_stage || 'GREEN';
  const greenLanes = new Set(snapshot.phase_green_lanes || []);
  const durations = snapshot.lane_signal_durations_s || [];

  for (let lane = 0; lane < 8; lane++) {
    const isActive = greenLanes.has(lane);
    const rLens = document.getElementById(`lens-${lane}-r`);
    const yLens = document.getElementById(`lens-${lane}-y`);
    const gLens = document.getElementById(`lens-${lane}-g`);
    const timerPill = document.getElementById(`timer-pill-${lane}`);

    if (!rLens || !yLens || !gLens || !timerPill) continue;

    rLens.classList.remove('active');
    yLens.classList.remove('active');
    gLens.classList.remove('active');
    timerPill.className = 'sig-timer-pill';

    const dur = Math.round(durations[lane] || 0);
    timerPill.textContent = `${dur}s`;

    if (isActive) {
      if (stage === 'GREEN') {
        gLens.classList.add('active');
        timerPill.classList.add('green');
      } else if (stage === 'YELLOW') {
        yLens.classList.add('active');
        timerPill.classList.add('yellow');
      } else {
        rLens.classList.add('active');
      }
    } else {
      rLens.classList.add('active');
    }
  }
}

/* ======== Project 3D Badges to 2D Screen Positions ======== */
const tempV = new THREE.Vector3();
function updateBadgePositions() {
  const width = container.clientWidth;
  const height = container.clientHeight;
  const widthHalf = width / 2;
  const heightHalf = height / 2;

  for (let i = 0; i < 8; i++) {
    const badgeEl = badgeElements[i];
    if (!badgeEl) continue;

    tempV.copy(LANE_BADGE_WORLD_POS[i]);
    tempV.project(camera);

    const x = tempV.x * widthHalf + widthHalf;
    const y = -(tempV.y * heightHalf) + heightHalf;

    if (tempV.z > 1.0 || x < 14 || x > width - 14 || y < 14 || y > height - 14) {
      badgeEl.style.display = 'none';
      continue;
    }

    badgeEl.style.display = 'flex';
    badgeEl.style.left = `${x}px`;
    badgeEl.style.top = `${y}px`;
  }
}

/* ======== Raycasting for Lane Clicks (Ignore Orbit Drags) ======== */
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let pointerDownPos = null;

container.addEventListener('pointerdown', (event) => {
  pointerDownPos = { x: event.clientX, y: event.clientY };
});

container.addEventListener('mousemove', (event) => {
  const rect = container.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(world.laneHitMeshes, false);

  if (intersects.length > 0) {
    const hitLane = intersects[0].object.userData.laneId;
    world.highlightLane(hitLane);
    container.style.cursor = 'pointer';
  } else {
    world.clearLaneHighlight();
    container.style.cursor = 'default';
  }
});

container.addEventListener('mouseleave', () => {
  world.clearLaneHighlight();
  container.style.cursor = 'default';
});

container.addEventListener('click', (event) => {
  if (pointerDownPos) {
    const dragDist = Math.hypot(
      event.clientX - pointerDownPos.x,
      event.clientY - pointerDownPos.y
    );
    pointerDownPos = null;
    if (dragDist > 5) return; // Ignore orbit camera drags
  }

  const rect = container.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(world.laneHitMeshes, false);

  if (intersects.length > 0) {
    const hitLane = intersects[0].object.userData.laneId;
    sendSimulationCommand({
      action: 'spawn_vehicles',
      lane_id: hitLane,
      count: 1,
    });
  }
});

/* ======== Transport Controls ======== */
btnPlayPause.addEventListener('click', () => {
  isPaused = !isPaused;
  btnPlayPause.textContent = isPaused ? 'Play' : 'Pause';
  sendSimulationCommand({ action: 'set_running', running: !isPaused });
});

btnStep.addEventListener('click', () => {
  sendSimulationCommand({ action: 'step' });
});

btnReset.addEventListener('click', () => {
  sendSimulationCommand({ action: 'reset' });
});

/* ======== Camera Presets ======== */
const camButtons = {
  top: document.getElementById('btn-cam-top'),
  angle: document.getElementById('btn-cam-angle'),
  street: document.getElementById('btn-cam-street'),
  cinematic: document.getElementById('btn-cam-cinematic'),
};

let cinematicActive = false;
let cinematicTime = 0;
let cinematicBaseAngle = Math.PI * 0.25;
let cinematicBlend = 1.0;
const cinematicStartPos = new THREE.Vector3();
const cinematicStartTarget = new THREE.Vector3();
let cameraTweenRaf = null;

function activateCamButton(activeKey) {
  for (const [key, btn] of Object.entries(camButtons)) {
    if (btn) btn.classList.toggle('active', key === activeKey);
  }
}

controls.addEventListener('start', () => {
  if (cinematicActive) {
    cinematicActive = false;
    activateCamButton(null);
  }
  if (cameraTweenRaf !== null) {
    cancelAnimationFrame(cameraTweenRaf);
    cameraTweenRaf = null;
  }
});

camButtons.top.addEventListener('click', () => {
  cinematicActive = false;
  _animateCamera(0, 48, 0.1, 0, 0, 0);
  activateCamButton('top');
});

camButtons.angle.addEventListener('click', () => {
  cinematicActive = false;
  _animateCamera(26, 28, 26, 0, 0, 0);
  activateCamButton('angle');
});

camButtons.street.addEventListener('click', () => {
  cinematicActive = false;
  _animateCamera(-3.0, 2.5, -24, -1.5, 1.0, 0);
  activateCamButton('street');
});

if (camButtons.cinematic) {
  camButtons.cinematic.addEventListener('click', () => {
    cinematicActive = !cinematicActive;
    if (!cinematicActive) {
      activateCamButton(null);
      return;
    }
    if (cameraTweenRaf !== null) {
      cancelAnimationFrame(cameraTweenRaf);
      cameraTweenRaf = null;
    }
    cinematicTime = 0;
    cinematicBlend = 0.0;
    const horizDist = Math.hypot(camera.position.x, camera.position.z);
    cinematicBaseAngle =
      horizDist > 1.0
        ? Math.atan2(camera.position.z, camera.position.x)
        : Math.PI * 0.25;
    cinematicStartPos.copy(camera.position);
    cinematicStartTarget.copy(controls.target);
    activateCamButton('cinematic');
  });
}

function _updateCinematicCamera(dt) {
  cinematicTime += dt;
  const t = cinematicTime;
  const angle = cinematicBaseAngle + t * 0.22;
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

  if (cinematicBlend < 1.0) {
    cinematicBlend = Math.min(1.0, cinematicBlend + dt / 0.9);
    const b = cinematicBlend;
    const ease = b < 0.5 ? 2 * b * b : 1 - Math.pow(-2 * b + 2, 2) / 2;
    camera.position.lerpVectors(cinematicStartPos, desiredPos, ease);
    controls.target.lerpVectors(cinematicStartTarget, desiredTarget, ease);
  } else {
    camera.position.copy(desiredPos);
    controls.target.copy(desiredTarget);
  }
}

function _animateCamera(x, y, z, tx, ty, tz) {
  if (cameraTweenRaf !== null) {
    cancelAnimationFrame(cameraTweenRaf);
    cameraTweenRaf = null;
  }
  const startPos = camera.position.clone();
  const endPos = new THREE.Vector3(x, y, z);
  const startTarget = controls.target.clone();
  const endTarget = new THREE.Vector3(tx, ty, tz);
  const duration = 600;
  const startTime = performance.now();

  function step(now) {
    const elapsed = now - startTime;
    const t = Math.min(1, elapsed / duration);
    const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

    camera.position.lerpVectors(startPos, endPos, ease);
    controls.target.lerpVectors(startTarget, endTarget, ease);
    controls.update();

    if (t < 1) {
      cameraTweenRaf = requestAnimationFrame(step);
    } else {
      cameraTweenRaf = null;
    }
  }
  cameraTweenRaf = requestAnimationFrame(step);
}

/* ======== Window Resize ======== */
window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

/* ======== Animation Loop ======== */
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);

  if (cinematicActive) {
    _updateCinematicCamera(dt);
  }
  controls.update();
  vehicles.animate(dt);
  updateBadgePositions();

  renderer.render(scene, camera);
}
animate();
