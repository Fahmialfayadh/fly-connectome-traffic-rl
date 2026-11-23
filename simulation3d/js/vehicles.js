/**
 * vehicles.js — 3D vehicle kinematics, queuing, smooth circular-arc turns,
 * straggler handling, and path-synchronized collision physics.
 *
 * Matches Python TrafficIntersectionEnv (traffic_env.py) and 2D IntersectionRenderer:
 * - Right-hand traffic (RHT) lane coordinates 0..7
 * - Step / mutation_seq deduplication so non-step actions never replay departures
 * - Newly arriving & manually spawned vehicles enter from the far back of the lane
 * - Car-following spacing guarantees zero overlapping / clipping inside queues or exits
 * - Turn vehicles follow exact C1-continuous tangent circular arcs into their true exit lanes
 */
import * as THREE from 'three';
import { ROAD_HALF, STOP_LINE_OFFSET } from './world.js';

const CAR_LENGTH = 1.9;
const CAR_WIDTH = 0.95;
const CAR_HEIGHT = 0.55;
const QUEUE_SPACING = 2.8; // 1.9m car + 0.9m clear gap
const STOP_OFFSET = STOP_LINE_OFFSET + 0.35 + CAR_LENGTH / 2; // 10.3m (bumper behind 9.0m bar)
const SPAWN_MIN_DIST = 34.0; // Off-screen in Top View & back horizon in 3D view
const EXIT_DESPAWN_DIST = 88.0;
const MAX_EXITING_CARS = 140;
const MAX_COLLIDING_CARS = 20;

const STRAIGHT_COLORS = [0x2563eb, 0x1d4ed8, 0x3b82f6, 0x1e40af, 0x475569];
const TURN_COLORS = [0xd97706, 0xb45309, 0xf59e0b, 0xea580c, 0xc2410c];

// Physical crossing conflict adjacency matching traffic_env.CONFLICTING_LANES
const CONFLICTING_LANES = [
  new Set([3, 4, 6, 7]), // 0: N_STR
  new Set([2, 4, 5, 7]), // 1: N_TRN
  new Set([1, 4, 5, 6]), // 2: S_STR
  new Set([0, 5, 6, 7]), // 3: S_TRN
  new Set([0, 1, 2, 7]), // 4: E_STR
  new Set([1, 2, 3, 6]), // 5: E_TRN
  new Set([0, 2, 3, 5]), // 6: W_STR
  new Set([0, 1, 3, 4]), // 7: W_TRN
];

/**
 * Right-hand traffic (RHT) specifications for lanes 0..7.
 * Local car +Z is front; world forward (dx, dz) maps to mesh.rotation.y = atan2(dx, dz).
 * For turns, arcCenter = (exitBoxCoord, entryBoxCoord) with radius R = 7.5m connects
 * the inner incoming lane (offset ±1.5m) smoothly to the inner outgoing lane (offset ±1.5m).
 */
const LANE_CONFIGS = [
  { id: 0, isTurn: false, axis: 'z', laneCoord: -4.5, queueSign: -1, dirX: 0, dirZ: 1 },
  {
    id: 1,
    isTurn: true,
    axis: 'z',
    laneCoord: -1.5,
    queueSign: -1,
    dirX: 0,
    dirZ: 1,
    arcCenter: [ROAD_HALF, -ROAD_HALF],
    arcRadius: 7.5,
    a0: Math.PI,
    a1: Math.PI / 2,
    exitDirX: 1,
    exitDirZ: 0,
    exitCoord: 1.5,
  },
  { id: 2, isTurn: false, axis: 'z', laneCoord: 4.5, queueSign: 1, dirX: 0, dirZ: -1 },
  {
    id: 3,
    isTurn: true,
    axis: 'z',
    laneCoord: 1.5,
    queueSign: 1,
    dirX: 0,
    dirZ: -1,
    arcCenter: [-ROAD_HALF, ROAD_HALF],
    arcRadius: 7.5,
    a0: 0,
    a1: -Math.PI / 2,
    exitDirX: -1,
    exitDirZ: 0,
    exitCoord: -1.5,
  },
  { id: 4, isTurn: false, axis: 'x', laneCoord: -4.5, queueSign: 1, dirX: -1, dirZ: 0 },
  {
    id: 5,
    isTurn: true,
    axis: 'x',
    laneCoord: -1.5,
    queueSign: 1,
    dirX: -1,
    dirZ: 0,
    arcCenter: [ROAD_HALF, ROAD_HALF],
    arcRadius: 7.5,
    a0: -Math.PI / 2,
    a1: -Math.PI,
    exitDirX: 0,
    exitDirZ: 1,
    exitCoord: -1.5,
  },
  { id: 6, isTurn: false, axis: 'x', laneCoord: 4.5, queueSign: -1, dirX: 1, dirZ: 0 },
  {
    id: 7,
    isTurn: true,
    axis: 'x',
    laneCoord: 1.5,
    queueSign: -1,
    dirX: 1,
    dirZ: 0,
    arcCenter: [-ROAD_HALF, -ROAD_HALF],
    arcRadius: 7.5,
    a0: Math.PI / 2,
    a1: 0,
    exitDirX: 0,
    exitDirZ: -1,
    exitCoord: 1.5,
  },
];

export class VehicleManager3D {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = 'vehicles';
    scene.add(this.group);

    this.queuedByLane = Array.from({ length: 8 }, () => []);
    this.exitingVehicles = [];
    this.collidingVehicles = [];
    this.collisionRings = [];
    this.collisionParticles = [];

    this.hasSyncedOnce = false;
    this.lastStepSeen = -1;
    this.lastMutationSeq = -1;
    this.currentStepsPerSecond = 2.0;
    this.clockSeconds = 0;

    // Shared geometries (never disposed per-car)
    this.bodyGeo = new THREE.BoxGeometry(CAR_WIDTH, CAR_HEIGHT * 0.62, CAR_LENGTH);
    this.cabinGeo = new THREE.BoxGeometry(CAR_WIDTH * 0.84, CAR_HEIGHT * 0.44, CAR_LENGTH * 0.54);
    this.wheelGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.12, 10);
    this.wheelGeo.rotateZ(Math.PI / 2);
    this.lightGeo = new THREE.BoxGeometry(0.18, 0.08, 0.04);
    this.turnGeo = new THREE.BoxGeometry(0.08, 0.08, 0.14);
    this.ringGeo = new THREE.RingGeometry(0.65, 0.95, 28);
    this.ringGeo.rotateX(-Math.PI / 2);
    this.hazardRingGeo = new THREE.RingGeometry(1.15, 1.32, 24);
    this.hazardRingGeo.rotateX(-Math.PI / 2);
    this.sparkGeo = new THREE.BoxGeometry(0.11, 0.11, 0.11);

    this.wheelMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.85 });
    this.cabinMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.2,
      metalness: 0.8,
    });
    this.headlightMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xfffae0,
      emissiveIntensity: 0.65,
      roughness: 0.3,
    });

    this.rng = this._seededRng(42);
  }

  /**
   * Backwards-compatible entry point if caller passes only snapshot.
   */
  updateFromSnapshot(snapshot) {
    this.updateFromFrame({ snapshot });
  }

  /**
   * Immediately reconcile scene queues & signals to a snapshot without animating old exiting cars.
   * Useful when a hidden 3D viewport becomes visible again.
   */
  syncImmediate(framePayload) {
    if (!framePayload) return;
    const snapshot = framePayload.snapshot ? framePayload.snapshot : framePayload;
    if (!snapshot || typeof snapshot !== 'object') return;

    if (
      framePayload.steps_per_second !== undefined &&
      framePayload.steps_per_second !== null
    ) {
      const parsedSps = Number(framePayload.steps_per_second);
      if (Number.isFinite(parsedSps) && parsedSps > 0) {
        this.currentStepsPerSecond = Math.min(Math.max(parsedSps, 0.5), 10.0);
      }
    }

    const currentStep = Number(snapshot.step || 0);
    const hasMutationSeq =
      snapshot.mutation_seq !== undefined && snapshot.mutation_seq !== null;
    const mutationSeq = hasMutationSeq ? Number(snapshot.mutation_seq) : -1;
    const laneQueues = snapshot.lane_queues || new Array(8).fill(0);

    this._resetScene(laneQueues, 0, []);
    this.hasSyncedOnce = true;
    this.lastStepSeen = currentStep;
    this.lastMutationSeq = mutationSeq;
  }

  /**
   * Sync 3D vehicles from a WebSocket frame payload ({ snapshot, steps_per_second }).
   */
  updateFromFrame(framePayload) {
    if (!framePayload) return;
    const snapshot = framePayload.snapshot ? framePayload.snapshot : framePayload;
    if (
      !snapshot ||
      typeof snapshot !== 'object' ||
      (snapshot.step === undefined && !Array.isArray(snapshot.lane_queues))
    ) {
      return;
    }

    if (
      framePayload.steps_per_second !== undefined &&
      framePayload.steps_per_second !== null
    ) {
      const parsedSps = Number(framePayload.steps_per_second);
      if (Number.isFinite(parsedSps) && parsedSps > 0) {
        this.currentStepsPerSecond = Math.min(Math.max(parsedSps, 0.5), 10.0);
      }
    }

    const stepDurationSec = 1.0 / Math.max(this.currentStepsPerSecond, 0.5);
    const currentStep = Number(snapshot.step || 0);
    const hasMutationSeq =
      snapshot.mutation_seq !== undefined && snapshot.mutation_seq !== null;
    const mutationSeq = hasMutationSeq ? Number(snapshot.mutation_seq) : -1;

    const laneQueues = snapshot.lane_queues || new Array(8).fill(0);
    const stepArrivals = snapshot.step_arrivals_per_lane || new Array(8).fill(0);
    const stepDepartures = snapshot.step_departures_per_lane || new Array(8).fill(0);
    const manualSpawns = snapshot.manual_spawns_per_lane || new Array(8).fill(0);
    const collisionEvents = snapshot.step_collision_events || [];
    const activeStragglerLanes = Array.isArray(snapshot.active_straggler_lanes)
      ? snapshot.active_straggler_lanes
          .map((lane) => Number(lane))
          .filter((lane) => Number.isInteger(lane) && lane >= 0 && lane <= 7)
      : [];

    const hasManualSpawns = manualSpawns.some((c) => Number(c) > 0);
    const mutationChanged = hasMutationSeq && mutationSeq !== this.lastMutationSeq;

    // 1. Initial connection or reset
    if (!this.hasSyncedOnce || currentStep < this.lastStepSeen) {
      this._resetScene(laneQueues, currentStep, collisionEvents);
      this._syncStragglers(activeStragglerLanes, stepDurationSec);
      this.hasSyncedOnce = true;
      this.lastStepSeen = currentStep;
      this.lastMutationSeq = mutationSeq;
      return;
    }

    // 2. Same step (e.g. manual click spawn, pause/play, agent/scenario switch)
    if (currentStep === this.lastStepSeen) {
      if (currentStep === 0 && mutationChanged && !hasManualSpawns) {
        const anyGrew = laneQueues.some(
          (target, idx) => Number(target || 0) > this.queuedByLane[idx].length
        );
        if (!anyGrew) {
          this._resetScene(laneQueues, currentStep, collisionEvents);
          this._syncStragglers(activeStragglerLanes, stepDurationSec);
          this.lastMutationSeq = mutationSeq;
          return;
        }
      }

      const queueDiffers =
        !hasMutationSeq &&
        laneQueues.some(
          (target, idx) => Number(target || 0) !== this.queuedByLane[idx].length
        );

      if (!mutationChanged && !queueDiffers) {
        return;
      }

      this._reconcileManualSpawns(laneQueues);
      this.lastMutationSeq = mutationSeq;
      return;
    }

    // 3. Normal step advance: extract red_light_conflict culprits first (env discharged them before step_departures)
    const redLightCulpritsByLane = Array.from({ length: 8 }, () => []);
    for (const ev of collisionEvents) {
      if (!ev || ev.kind !== 'red_light_conflict') continue;
      const culpritLane = Number(ev.culprit_lane);
      if (
        Number.isInteger(culpritLane) &&
        culpritLane >= 0 &&
        culpritLane <= 7 &&
        this.queuedByLane[culpritLane].length > 0
      ) {
        redLightCulpritsByLane[culpritLane].push(this.queuedByLane[culpritLane].shift());
      }
    }

    for (let lane = 0; lane < 8; lane++) {
      const queue = this.queuedByLane[lane];
      const departedCount = Math.min(Number(stepDepartures[lane] || 0), queue.length);

      for (let d = 0; d < departedCount; d++) {
        const departingCar = queue.shift();
        this._startExiting(departingCar, lane, d, stepDurationSec);
      }

      const arrivedCount = Number(stepArrivals[lane] || 0);
      for (let a = 0; a < arrivedCount; a++) {
        const tailDist = queue.length > 0 ? queue[queue.length - 1].distFromStop : 0;
        const spawnDist = Math.max(SPAWN_MIN_DIST, tailDist + QUEUE_SPACING * 1.45);
        queue.push(this._createQueuedCar(lane, spawnDist, 14.0));
      }

      const targetCount = Math.max(0, Number(laneQueues[lane] || 0));
      while (queue.length > targetCount) {
        this._removeCarFromScene(queue.pop());
      }
      while (queue.length < targetCount) {
        const tailDist = queue.length > 0 ? queue[queue.length - 1].distFromStop : 0;
        const spawnDist = Math.max(SPAWN_MIN_DIST, tailDist + QUEUE_SPACING * 1.45);
        queue.push(this._createQueuedCar(lane, spawnDist, 14.0));
      }
    }

    this._processCollisionEvents(collisionEvents, stepDurationSec, redLightCulpritsByLane);
    this._syncStragglers(activeStragglerLanes, stepDurationSec);

    this.lastStepSeen = currentStep;
    this.lastMutationSeq = mutationSeq;
  }

  _resetScene(laneQueues, currentStep, collisionEvents) {
    for (let lane = 0; lane < 8; lane++) {
      for (const car of this.queuedByLane[lane]) {
        this._removeCarFromScene(car);
      }
      this.queuedByLane[lane] = [];
    }
    for (const v of this.exitingVehicles) this._removeCarFromScene(v);
    for (const c of this.collidingVehicles) this._removeCarFromScene(c);
    for (const r of this.collisionRings) {
      this.group.remove(r.mesh);
      r.mesh.material.dispose();
    }
    for (const p of this.collisionParticles) {
      this.group.remove(p.mesh);
      p.mesh.material.dispose();
    }

    this.exitingVehicles = [];
    this.collidingVehicles = [];
    this.collisionRings = [];
    this.collisionParticles = [];

    for (let lane = 0; lane < 8; lane++) {
      const initialCount = Math.max(0, Number(laneQueues[lane] || 0));
      for (let slot = 0; slot < initialCount; slot++) {
        const car = this._createQueuedCar(lane, slot * QUEUE_SPACING, 0);
        car.braking = true;
        this._applyQueuedTransform(car);
        this.queuedByLane[lane].push(car);
      }
    }

    if (currentStep > 0 && Array.isArray(collisionEvents) && collisionEvents.length > 0) {
      const stepDurationSec = 1.0 / Math.max(this.currentStepsPerSecond, 0.5);
      this._processCollisionEvents(collisionEvents, stepDurationSec, null);
    }
  }

  _reconcileManualSpawns(laneQueues) {
    for (let lane = 0; lane < 8; lane++) {
      const queue = this.queuedByLane[lane];
      const targetCount = Math.max(0, Number(laneQueues[lane] || 0));

      while (queue.length > targetCount) {
        this._removeCarFromScene(queue.pop());
      }
      while (queue.length < targetCount) {
        const tailDist = queue.length > 0 ? queue[queue.length - 1].distFromStop : 0;
        const spawnDist = Math.max(SPAWN_MIN_DIST, tailDist + QUEUE_SPACING * 1.45);
        queue.push(this._createQueuedCar(lane, spawnDist, 16.0));
      }
    }
  }

  _createQueuedCar(lane, distFromStop, initialSpeed) {
    const car = this._createCarMesh(lane);
    car.distFromStop = distFromStop;
    car.speed = initialSpeed;
    car.braking = initialSpeed < 1.0;
    this._applyQueuedTransform(car);
    return car;
  }

  _applyQueuedTransform(car) {
    const cfg = LANE_CONFIGS[car.lane];
    const distFromCenter = STOP_OFFSET + car.distFromStop;
    if (cfg.axis === 'z') {
      car.mesh.position.set(cfg.laneCoord, 0, cfg.queueSign * distFromCenter);
    } else {
      car.mesh.position.set(cfg.queueSign * distFromCenter, 0, cfg.laneCoord);
    }
    car.mesh.rotation.y = Math.atan2(cfg.dirX, cfg.dirZ);
  }

  _buildCrossingPath(lane, startDistFromStop) {
    const cfg = LANE_CONFIGS[lane];
    const startCenterOffset = STOP_OFFSET + startDistFromStop;
    const entryLen = Math.max(0, startCenterOffset - ROAD_HALF);

    if (!cfg.isTurn) {
      const startX = cfg.axis === 'z' ? cfg.laneCoord : cfg.queueSign * startCenterOffset;
      const startZ = cfg.axis === 'z' ? cfg.queueSign * startCenterOffset : cfg.laneCoord;
      const endX = cfg.axis === 'z' ? cfg.laneCoord : -cfg.queueSign * EXIT_DESPAWN_DIST;
      const endZ = cfg.axis === 'z' ? -cfg.queueSign * EXIT_DESPAWN_DIST : cfg.laneCoord;
      const totalLen = Math.hypot(endX - startX, endZ - startZ);
      return {
        kind: 'line',
        x0: startX,
        z0: startZ,
        x1: endX,
        z1: endZ,
        dx: cfg.dirX,
        dz: cfg.dirZ,
        totalLen,
      };
    }

    const startX = cfg.axis === 'z' ? cfg.laneCoord : cfg.queueSign * startCenterOffset;
    const startZ = cfg.axis === 'z' ? cfg.queueSign * startCenterOffset : cfg.laneCoord;
    const boxInX = cfg.axis === 'z' ? cfg.laneCoord : cfg.queueSign * ROAD_HALF;
    const boxInZ = cfg.axis === 'z' ? cfg.queueSign * ROAD_HALF : cfg.laneCoord;

    const [cx, cz] = cfg.arcCenter;
    const arcLen = cfg.arcRadius * Math.abs(cfg.a1 - cfg.a0);

    const boxOutX = cx + cfg.arcRadius * Math.cos(cfg.a1);
    const boxOutZ = cz + cfg.arcRadius * Math.sin(cfg.a1);
    const endX = boxOutX + cfg.exitDirX * (EXIT_DESPAWN_DIST - ROAD_HALF);
    const endZ = boxOutZ + cfg.exitDirZ * (EXIT_DESPAWN_DIST - ROAD_HALF);
    const exitLen = Math.hypot(endX - boxOutX, endZ - boxOutZ);

    return {
      kind: 'turn_arc',
      entry: { x0: startX, z0: startZ, x1: boxInX, z1: boxInZ, dx: cfg.dirX, dz: cfg.dirZ, len: entryLen },
      arc: { cx, cz, radius: cfg.arcRadius, a0: cfg.a0, a1: cfg.a1, len: arcLen },
      exit: { x0: boxOutX, z0: boxOutZ, x1: endX, z1: endZ, dx: cfg.exitDirX, dz: cfg.exitDirZ, len: exitLen },
      totalLen: entryLen + arcLen + exitLen,
    };
  }

  _samplePathPose(path, progressM) {
    if (path.kind === 'line') {
      const t = Math.min(Math.max(progressM / Math.max(path.totalLen, 0.001), 0), 1);
      return {
        x: path.x0 + (path.x1 - path.x0) * t,
        z: path.z0 + (path.z1 - path.z0) * t,
        dx: path.dx,
        dz: path.dz,
      };
    }

    if (progressM <= path.entry.len) {
      const t = path.entry.len > 0 ? Math.min(Math.max(progressM / path.entry.len, 0), 1) : 1;
      return {
        x: path.entry.x0 + (path.entry.x1 - path.entry.x0) * t,
        z: path.entry.z0 + (path.entry.z1 - path.entry.z0) * t,
        dx: path.entry.dx,
        dz: path.entry.dz,
      };
    }

    const afterEntry = progressM - path.entry.len;
    if (afterEntry <= path.arc.len) {
      const t = Math.min(Math.max(afterEntry / path.arc.len, 0), 1);
      const angleDelta = path.arc.a1 - path.arc.a0;
      const angle = path.arc.a0 + angleDelta * t;
      const sign = angleDelta >= 0 ? 1 : -1;
      const dx = -sign * Math.sin(angle);
      const dz = sign * Math.cos(angle);
      return {
        x: path.arc.cx + path.arc.radius * Math.cos(angle),
        z: path.arc.cz + path.arc.radius * Math.sin(angle),
        dx,
        dz,
      };
    }

    const afterArc = afterEntry - path.arc.len;
    const t = Math.min(Math.max(afterArc / Math.max(path.exit.len, 0.001), 0), 1);
    return {
      x: path.exit.x0 + (path.exit.x1 - path.exit.x0) * t,
      z: path.exit.z0 + (path.exit.z1 - path.exit.z0) * t,
      dx: path.exit.dx,
      dz: path.exit.dz,
    };
  }

  _computeBoxClearM(path, startDist) {
    if (path.kind === 'line') {
      return startDist + (STOP_OFFSET - ROAD_HALF) + ROAD_HALF * 2 + 1.6;
    }
    return path.entry.len + path.arc.len + 1.6;
  }

  _computeBoxEntryHoldM(startDist) {
    // Hold right at the box edge (front bumper just past crosswalk / stop line)
    return Math.max(0, startDist + (STOP_OFFSET - ROAD_HALF) - 1.2);
  }

  _startExiting(car, lane, slotInBatch, stepDurationSec) {
    if (this.exitingVehicles.length >= MAX_EXITING_CARS) {
      this._removeCarFromScene(this.exitingVehicles.shift());
    }

    const startDist = Math.max(0, Number(car.distFromStop || 0));
    const path = this._buildCrossingPath(lane, startDist);
    const boxClearM = this._computeBoxClearM(path, startDist);
    const boxEntryHoldM = this._computeBoxEntryHoldM(startDist);
    const delayRemaining = slotInBatch * Math.min(0.08, stepDurationSec * 0.14);
    const targetClearSec = Math.max(0.08, stepDurationSec * 1.15 - delayRemaining);
    const maxSyncSpeed = Math.max(48.0, 20.0 * this.currentStepsPerSecond);
    const cruiseSpeed = Math.min(maxSyncSpeed, Math.max(15.0, boxClearM / targetClearSec));

    this.exitingVehicles.push({
      mesh: car.mesh,
      lane,
      isTurn: car.isTurn,
      isStraggler: false,
      startDist,
      path,
      progressM: 0,
      boxClearM,
      boxEntryHoldM,
      boxEntryM: startDist + (STOP_OFFSET - ROAD_HALF),
      cruiseSpeed,
      speed: cruiseSpeed,
      delayRemaining,
      braking: false,
    });
  }

  _syncStragglers(activeStragglerLanes, stepDurationSec) {
    const stragglerSet = new Set(activeStragglerLanes);

    for (const v of this.exitingVehicles) {
      if (v.isStraggler && !stragglerSet.has(v.lane)) {
        v.isStraggler = false;
        v.delayRemaining = 0;
        const remainingClearM = Math.max(2.0, v.boxClearM - v.progressM);
        v.cruiseSpeed = Math.min(
          42.0,
          Math.max(18.0, remainingClearM / Math.max(0.08, stepDurationSec * 0.75))
        );
        v.speed = Math.max(v.speed, v.cruiseSpeed);
        this._setHazardRingVisible(v, false);
      }
    }

    for (const lane of stragglerSet) {
      if (this.exitingVehicles.some((v) => v.lane === lane && v.isStraggler)) {
        continue;
      }
      let candidate = null;
      for (let i = this.exitingVehicles.length - 1; i >= 0; i--) {
        const item = this.exitingVehicles[i];
        if (item.lane === lane && item.progressM <= item.boxEntryHoldM + 2.0) {
          candidate = item;
          break;
        }
      }
      if (candidate) {
        candidate.isStraggler = true;
        candidate.delayRemaining = 0;
        candidate.holdTargetM = Math.max(candidate.progressM, candidate.boxEntryHoldM);
        this._setHazardRingVisible(candidate, true, 0xf59e0b);
      }
    }
  }

  _processCollisionEvents(events, stepDurationSec, redLightCulpritsByLane) {
    if (!Array.isArray(events) || events.length === 0) return;

    const approachSec = Math.min(0.24, stepDurationSec * 0.42);
    const haltSec = Math.max(0.25, Math.min(1.3, stepDurationSec * 1.8));

    for (const ev of events) {
      if (!ev || typeof ev !== 'object') continue;
      const culpritLane = Number(ev.culprit_lane);
      const victimLane = Number(ev.victim_lane);
      if (!Number.isInteger(culpritLane) || culpritLane < 0 || culpritLane > 7) continue;

      const kind = String(ev.kind || 'red_light_conflict');
      const hasValidVictim =
        Number.isInteger(victimLane) &&
        victimLane >= 0 &&
        victimLane <= 7 &&
        victimLane !== culpritLane;

      let culpritSource = null;
      if (kind === 'box_clearance_conflict') {
        culpritSource = this._takeStragglerCulprit(culpritLane);
      } else if (
        redLightCulpritsByLane &&
        Array.isArray(redLightCulpritsByLane[culpritLane]) &&
        redLightCulpritsByLane[culpritLane].length > 0
      ) {
        culpritSource = redLightCulpritsByLane[culpritLane].shift();
      }

      const culpritStartDist = culpritSource
        ? Number(culpritSource.startDist ?? culpritSource.distFromStop ?? 0)
        : 0;
      const culpritPath =
        culpritSource?.path || this._buildCrossingPath(culpritLane, culpritStartDist);
      const culpritStartProgress = culpritSource?.path ? Number(culpritSource.progressM || 0) : 0;
      const culpritMesh = culpritSource?.mesh || this._createCarMesh(culpritLane).mesh;
      if (culpritSource) this._setHazardRingVisible(culpritSource, false);

      const victimSource = hasValidVictim ? this._takeVictimCandidate(victimLane) : null;
      const victimStartDist = victimSource
        ? Number(victimSource.startDist ?? victimSource.distFromStop ?? 0)
        : 0;
      const victimPath =
        victimSource?.path ||
        (hasValidVictim ? this._buildCrossingPath(victimLane, victimStartDist) : null);
      const victimStartProgress = victimSource?.path ? Number(victimSource.progressM || 0) : 0;

      const conflictGeo = victimPath
        ? this._computeConflictGeometry(culpritPath, victimPath)
        : this._fallbackConflictGeometry(culpritPath);

      const standoff = CAR_LENGTH * 0.48;
      const culpritImpactM = Math.max(
        culpritStartProgress,
        Math.max(0, conflictGeo.culpritProgressM - standoff)
      );
      const victimImpactM = Math.max(
        victimStartProgress,
        Math.max(0, conflictGeo.victimProgressM - standoff)
      );

      this._pushCollidingVehicle({
        mesh: culpritMesh,
        lane: culpritLane,
        isTurn: LANE_CONFIGS[culpritLane].isTurn,
        role: 'culprit',
        startDist: culpritStartDist,
        path: culpritPath,
        startProgressM: culpritStartProgress,
        progressM: culpritStartProgress,
        impactProgressM: culpritImpactM,
        approachSec,
        haltSec,
        phaseElapsed: 0,
        state: 'approaching',
        braking: false,
        yawOffset: 0,
        impactYawOffset: -0.18,
        speed: 18.0,
      });

      if (hasValidVictim && victimPath) {
        const victimMesh = victimSource?.mesh || this._createCarMesh(victimLane).mesh;
        this._pushCollidingVehicle({
          mesh: victimMesh,
          lane: victimLane,
          isTurn: LANE_CONFIGS[victimLane].isTurn,
          role: 'victim',
          startDist: victimStartDist,
          path: victimPath,
          startProgressM: victimStartProgress,
          progressM: victimStartProgress,
          impactProgressM: victimImpactM,
          approachSec,
          haltSec,
          phaseElapsed: 0,
          state: 'approaching',
          braking: false,
          yawOffset: 0,
          impactYawOffset: 0.18,
          speed: 18.0,
        });
      }

      // Ground hazard pulse ring at the exact impact coordinates
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0xef4444,
        transparent: true,
        opacity: 0.9,
        side: THREE.DoubleSide,
      });
      const ringMesh = new THREE.Mesh(this.ringGeo, ringMat);
      ringMesh.position.set(conflictGeo.x, 0.03, conflictGeo.z);
      this.group.add(ringMesh);
      this.collisionRings.push({
        mesh: ringMesh,
        delayRemaining: approachSec,
        elapsed: 0,
        totalSec: haltSec + 0.8,
        cx: conflictGeo.x,
        cz: conflictGeo.z,
        sparksSpawned: false,
      });
    }
  }

  _pushCollidingVehicle(entry) {
    if (this.collidingVehicles.length >= MAX_COLLIDING_CARS) {
      this._removeCarFromScene(this.collidingVehicles.shift());
    }
    this._setHazardRingVisible(entry, true, entry.role === 'culprit' ? 0xef4444 : 0xf59e0b);
    this.collidingVehicles.push(entry);
  }

  _takeStragglerCulprit(lane) {
    for (let i = 0; i < this.exitingVehicles.length; i++) {
      const cand = this.exitingVehicles[i];
      if (cand.lane === lane && cand.isStraggler) {
        this.exitingVehicles.splice(i, 1);
        return cand;
      }
    }
    for (let i = this.exitingVehicles.length - 1; i >= 0; i--) {
      const cand = this.exitingVehicles[i];
      if (cand.lane === lane && cand.progressM <= cand.boxEntryM + 1.0) {
        this.exitingVehicles.splice(i, 1);
        return cand;
      }
    }
    return null;
  }

  _takeVictimCandidate(victimLane) {
    let bestIdx = -1;
    let lowestProgress = Infinity;
    for (let i = 0; i < this.exitingVehicles.length; i++) {
      const cand = this.exitingVehicles[i];
      if (cand.lane === victimLane && !cand.isStraggler && cand.progressM < lowestProgress) {
        lowestProgress = cand.progressM;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      return this.exitingVehicles.splice(bestIdx, 1)[0];
    }
    return null;
  }

  _sampleBoxPoints(path, stepM = 0.25) {
    const margin = ROAD_HALF + 0.6;
    const maxSample = Math.min(path.totalLen, 50.0);
    const pts = [];
    for (let m = 0; m <= maxSample; m += stepM) {
      const pose = this._samplePathPose(path, m);
      if (Math.abs(pose.x) <= margin && Math.abs(pose.z) <= margin) {
        pts.push({ x: pose.x, z: pose.z, progress: m });
      }
    }
    return pts;
  }

  _computeConflictGeometry(culpritPath, victimPath) {
    const cPts = this._sampleBoxPoints(culpritPath, 0.25);
    const vPts = this._sampleBoxPoints(victimPath, 0.25);
    let bestDistSq = Infinity;
    let bestPair = null;

    for (const pA of cPts) {
      for (const pB of vPts) {
        const dx = pA.x - pB.x;
        const dz = pA.z - pB.z;
        const d2 = dx * dx + dz * dz;
        if (d2 < bestDistSq) {
          bestDistSq = d2;
          bestPair = { pA, pB };
        }
      }
    }

    if (bestPair) {
      return {
        x: (bestPair.pA.x + bestPair.pB.x) / 2,
        z: (bestPair.pA.z + bestPair.pB.z) / 2,
        culpritProgressM: bestPair.pA.progress,
        victimProgressM: bestPair.pB.progress,
      };
    }
    return this._fallbackConflictGeometry(culpritPath);
  }

  _fallbackConflictGeometry(culpritPath) {
    const pts = this._sampleBoxPoints(culpritPath, 0.25);
    if (pts.length > 0) {
      const mid = pts[Math.floor(pts.length / 2)];
      return {
        x: mid.x,
        z: mid.z,
        culpritProgressM: mid.progress,
        victimProgressM: mid.progress,
      };
    }
    return { x: 0, z: 0, culpritProgressM: 8.0, victimProgressM: 8.0 };
  }

  /**
   * 60 FPS continuous kinematics & anti-collision animation loop.
   */
  animate(dt) {
    if (dt <= 0) return;
    this.clockSeconds += dt;
    const blinkOn = Math.floor(this.clockSeconds * 3.2) % 2 === 0;
    const stepDurationSec = 1.0 / Math.max(this.currentStepsPerSecond, 0.5);

    // Track minimum sPastStop of any exiting/colliding vehicle near the stop line per lane
    // so queued cars behind them never drive into a departing car that hasn't pulled away yet.
    const nearestBlockingPastStop = new Array(8).fill(Infinity);
    for (const v of this.exitingVehicles) {
      const s = v.progressM - v.startDist;
      if (s < QUEUE_SPACING * 1.1 && s < nearestBlockingPastStop[v.lane]) {
        nearestBlockingPastStop[v.lane] = s;
      }
    }
    for (const cv of this.collidingVehicles) {
      const s = cv.progressM - cv.startDist;
      if (cv.state !== 'clearing' && s < QUEUE_SPACING * 1.1 && s < nearestBlockingPastStop[cv.lane]) {
        nearestBlockingPastStop[cv.lane] = s;
      }
    }

    // 1. Update queued vehicles with car-following spacing (zero overlap)
    const maxApproachSpeed = Math.max(22.0, 14.0 * this.currentStepsPerSecond);
    for (let lane = 0; lane < 8; lane++) {
      const queue = this.queuedByLane[lane];
      for (let slot = 0; slot < queue.length; slot++) {
        const car = queue[slot];
        const nominalTarget = slot * QUEUE_SPACING;
        let minSafeDist = nominalTarget;

        if (slot === 0 && nearestBlockingPastStop[lane] < QUEUE_SPACING) {
          minSafeDist = Math.max(minSafeDist, QUEUE_SPACING - nearestBlockingPastStop[lane]);
        }
        if (slot > 0) {
          minSafeDist = Math.max(minSafeDist, queue[slot - 1].distFromStop + QUEUE_SPACING);
        }

        const delta = car.distFromStop - minSafeDist;
        if (delta > 0.03) {
          const desiredSpeed = Math.min(maxApproachSpeed, delta * 2.8 + 1.8);
          car.speed += (desiredSpeed - car.speed) * Math.min(1, dt * 8.0);
          car.distFromStop = Math.max(minSafeDist, car.distFromStop - car.speed * dt);
          car.braking = delta < 3.5;
        } else {
          car.distFromStop = minSafeDist;
          car.speed = 0;
          car.braking = true;
        }

        this._applyQueuedTransform(car);
        this._updateCarLights(car.mesh, car.braking, car.isTurn && blinkOn);
      }
    }

    // 2. Determine which lanes currently have an active vehicle strictly inside the box
    const activeBoxOccupiedLanes = new Set();
    for (const v of this.exitingVehicles) {
      if (v.delayRemaining > 0 || v.isStraggler) continue;
      if (v.progressM >= v.boxEntryM + 0.3 && v.progressM <= v.boxClearM - 0.4) {
        activeBoxOccupiedLanes.add(v.lane);
      }
    }
    for (const cv of this.collidingVehicles) {
      if (cv.state !== 'clearing') {
        const entryM = cv.startDist + (STOP_OFFSET - ROAD_HALF);
        const clearM = this._computeBoxClearM(cv.path, cv.startDist);
        if (cv.progressM >= entryM + 0.3 && cv.progressM <= clearM - 0.4) {
          activeBoxOccupiedLanes.add(cv.lane);
        }
      }
    }

    // Track halted/approaching colliding vehicles per lane so same-lane followers strictly behind them don't clip
    const activeCollidingBlockSByLane = Array.from({ length: 8 }, () => []);
    for (const cv of this.collidingVehicles) {
      if (cv.state !== 'clearing') {
        activeCollidingBlockSByLane[cv.lane].push(cv.progressM - cv.startDist);
      }
    }

    // 3. Advance exiting vehicles grouped by lane (frontmost first) to guarantee same-lane gap
    const exitingByLane = Array.from({ length: 8 }, () => []);
    for (const v of this.exitingVehicles) {
      exitingByLane[v.lane].push(v);
    }

    const survivingExiting = [];
    const maxSyncCap = Math.max(48.0, 20.0 * this.currentStepsPerSecond);
    for (let lane = 0; lane < 8; lane++) {
      const laneCars = exitingByLane[lane];
      laneCars.sort((a, b) => (b.progressM - b.startDist) - (a.progressM - a.startDist));

      let leaderS = Infinity;
      const collBlockers = activeCollidingBlockSByLane[lane];
      for (const v of laneCars) {
        const myS = v.progressM - v.startDist;
        let effectiveLeaderS = leaderS;
        for (const cvS of collBlockers) {
          if (cvS > myS - 0.3 && cvS < effectiveLeaderS) {
            effectiveLeaderS = cvS;
          }
        }

        if (v.delayRemaining > 0) {
          v.delayRemaining -= dt;
          this._applyExitingTransform(v);
          this._updateCarLights(v.mesh, true, v.isTurn && blinkOn);
          survivingExiting.push(v);
          leaderS = Math.min(leaderS, myS);
          continue;
        }

        if (v.isStraggler) {
          const prevM = v.progressM;
          const holdM = v.holdTargetM ?? v.boxEntryHoldM;
          if (v.progressM < holdM - 0.15) {
            v.speed = Math.min(12.0, Math.max(4.0, (holdM - v.progressM) * 2.5));
            v.progressM = Math.min(holdM, v.progressM + v.speed * dt);
          } else {
            v.speed = 0.5;
            v.progressM = Math.min(holdM + 0.4, v.progressM + v.speed * dt);
          }
          const maxAllowedM = effectiveLeaderS - QUEUE_SPACING + v.startDist;
          if (v.progressM > maxAllowedM) {
            v.progressM = Math.max(prevM, maxAllowedM);
          }
          leaderS = v.progressM - v.startDist;

          this._applyExitingTransform(v);
          this._updateCarLights(v.mesh, true, blinkOn);
          survivingExiting.push(v);
          continue;
        }

        // Courtesy box-entry yield if a conflicting lane still has a vehicle inside the box
        const nearBoxEntry =
          v.progressM >= v.boxEntryM - 2.5 && v.progressM <= v.boxEntryM + 0.1;
        let yieldingForBox = false;
        if (nearBoxEntry) {
          for (const occLane of activeBoxOccupiedLanes) {
            if (CONFLICTING_LANES[v.lane].has(occLane)) {
              yieldingForBox = true;
              break;
            }
          }
        }

        let nextM;
        if (yieldingForBox) {
          const holdLineM = Math.max(v.progressM, v.boxEntryM - 0.15);
          const distToHold = Math.max(0, holdLineM - v.progressM);
          v.speed = Math.min(v.speed, distToHold * 6.0);
          v.braking = true;
          nextM = Math.min(holdLineM, v.progressM + v.speed * dt);
        } else {
          const minSyncSpeed = Math.min(
            maxSyncCap,
            Math.max(15.0, v.boxClearM / Math.max(0.08, stepDurationSec * 1.15))
          );
          v.speed = Math.max(v.cruiseSpeed, minSyncSpeed);
          v.braking = false;
          nextM = v.progressM + v.speed * dt;
          // If crossing into box entry while conflicting lane occupies box, clamp before crossing
          if (v.progressM < v.boxEntryM + 0.2 && nextM >= v.boxEntryM - 0.15) {
            let blocked = false;
            for (const occLane of activeBoxOccupiedLanes) {
              if (CONFLICTING_LANES[v.lane].has(occLane)) {
                nextM = Math.min(v.progressM, v.boxEntryM - 0.15);
                v.speed = 0;
                v.braking = true;
                blocked = true;
                break;
              }
            }
            if (!blocked) {
              activeBoxOccupiedLanes.add(v.lane);
            }
          }
        }

        const maxAllowedM = effectiveLeaderS - QUEUE_SPACING + v.startDist;
        if (nextM > maxAllowedM) {
          nextM = Math.max(v.progressM, maxAllowedM);
          v.braking = true;
        }
        v.progressM = nextM;
        if (v.progressM >= v.boxEntryM + 0.3 && v.progressM <= v.boxClearM - 0.4) {
          activeBoxOccupiedLanes.add(v.lane);
        }
        leaderS = v.progressM - v.startDist;

        if (v.progressM < v.path.totalLen) {
          this._applyExitingTransform(v);
          const turningNow =
            v.isTurn && v.progressM < v.path.entry.len + v.path.arc.len;
          this._updateCarLights(v.mesh, v.braking, turningNow && blinkOn);
          survivingExiting.push(v);
        } else {
          this._removeCarFromScene(v);
        }
      }
    }
    this.exitingVehicles = survivingExiting;

    // 4. Advance colliding vehicles along their real lanes
    const survivingColliding = [];
    for (const cv of this.collidingVehicles) {
      if (cv.state === 'approaching') {
        cv.phaseElapsed += dt;
        const ratio = Math.min(1, cv.phaseElapsed / Math.max(cv.approachSec, 0.01));
        cv.progressM = cv.startProgressM + (cv.impactProgressM - cv.startProgressM) * ratio;
        cv.braking = ratio > 0.72;
        cv.yawOffset = 0;
        if (ratio >= 1) {
          cv.state = 'halted';
          cv.phaseElapsed = 0;
          cv.progressM = cv.impactProgressM;
          cv.braking = true;
          cv.yawOffset = cv.impactYawOffset || 0;
        }
        this._applyExitingTransform(cv, cv.yawOffset);
        this._updateCarLights(cv.mesh, cv.braking, blinkOn);
        survivingColliding.push(cv);
        continue;
      }

      if (cv.state === 'halted') {
        cv.phaseElapsed += dt;
        cv.braking = true;
        if (cv.phaseElapsed >= cv.haltSec) {
          cv.state = 'clearing';
          cv.phaseElapsed = 0;
          cv.braking = false;
          cv.yawOffset = 0;
          this._setHazardRingVisible(cv, false);
        }
        this._applyExitingTransform(cv, cv.yawOffset);
        this._updateCarLights(cv.mesh, true, blinkOn);
        survivingColliding.push(cv);
        continue;
      }

      const clearMinSpeed = Math.min(
        Math.max(46.0, 20.0 * this.currentStepsPerSecond),
        Math.max(16.0, 22.0 / Math.max(0.08, stepDurationSec * 0.9))
      );
      cv.speed = Math.max(cv.speed || 16.0, clearMinSpeed);
      cv.progressM += cv.speed * dt;
      if (cv.progressM < cv.path.totalLen) {
        this._applyExitingTransform(cv, 0);
        this._updateCarLights(cv.mesh, false, cv.isTurn && blinkOn);
        survivingColliding.push(cv);
      } else {
        this._removeCarFromScene(cv);
      }
    }
    this.collidingVehicles = survivingColliding;

    // 5. Update collision rings & spark particles
    for (let i = this.collisionRings.length - 1; i >= 0; i--) {
      const ring = this.collisionRings[i];
      if (ring.delayRemaining > 0) {
        ring.delayRemaining -= dt;
        ring.mesh.visible = false;
        continue;
      }
      if (!ring.sparksSpawned) {
        ring.sparksSpawned = true;
        this._spawnSparks(ring.cx, ring.cz);
      }
      ring.mesh.visible = true;
      ring.elapsed += dt;
      const t = ring.elapsed / Math.max(ring.totalSec, 0.1);
      if (t >= 1) {
        this.group.remove(ring.mesh);
        ring.mesh.material.dispose();
        this.collisionRings.splice(i, 1);
        continue;
      }
      const pulse = 1.0 + ((ring.elapsed * 2.2) % 1.0) * 1.9;
      ring.mesh.scale.set(pulse, 1, pulse);
      ring.mesh.material.opacity = (1 - t) * 0.85;
    }

    for (let i = this.collisionParticles.length - 1; i >= 0; i--) {
      const p = this.collisionParticles[i];
      p.mesh.position.addScaledVector(p.velocity, dt);
      p.velocity.y -= 12.0 * dt;
      if (p.mesh.position.y < 0.05) {
        p.mesh.position.y = 0.05;
        p.velocity.y *= -0.3;
      }
      p.life -= dt;
      p.mesh.scale.multiplyScalar(0.95);
      if (p.life <= 0) {
        this.group.remove(p.mesh);
        p.mesh.material.dispose();
        this.collisionParticles.splice(i, 1);
      }
    }
  }

  _applyExitingTransform(vehicle, yawOffset = 0) {
    const pose = this._samplePathPose(vehicle.path, vehicle.progressM);
    vehicle.mesh.position.set(pose.x, 0, pose.z);
    vehicle.mesh.rotation.y = Math.atan2(pose.dx, pose.dz) + yawOffset;
  }

  _spawnSparks(cx, cz) {
    for (let p = 0; p < 16; p++) {
      const mat = new THREE.MeshBasicMaterial({ color: p % 2 === 0 ? 0xffaa00 : 0xef4444 });
      const spark = new THREE.Mesh(this.sparkGeo, mat);
      spark.position.set(cx, 0.45, cz);
      const vel = new THREE.Vector3(
        (this.rng() - 0.5) * 8.0,
        2.5 + this.rng() * 5.0,
        (this.rng() - 0.5) * 8.0
      );
      this.group.add(spark);
      this.collisionParticles.push({
        mesh: spark,
        velocity: vel,
        life: 0.45 + this.rng() * 0.35,
      });
    }
  }

  _setHazardRingVisible(car, visible, colorHex = 0xf59e0b) {
    const ring = car?.mesh?.userData?.hazardRing;
    if (!ring) return;
    ring.visible = Boolean(visible);
    if (visible) {
      ring.material.color.setHex(colorHex);
    }
  }

  _updateCarLights(carMesh, braking, turnBlinkOn) {
    const brakeLights = carMesh.userData.brakeLights || [];
    for (const bl of brakeLights) {
      bl.material.emissiveIntensity = braking ? 0.95 : 0.18;
    }
    const turnLights = carMesh.userData.turnSignals || [];
    for (const tl of turnLights) {
      tl.material.emissiveIntensity = turnBlinkOn ? 1.2 : 0.0;
    }
  }

  _createCarMesh(lane) {
    const isTurn = lane % 2 === 1;
    const palette = isTurn ? TURN_COLORS : STRAIGHT_COLORS;
    const colorHex = palette[Math.floor(this.rng() * palette.length)];

    const carGroup = new THREE.Group();
    carGroup.name = `car_lane_${lane}`;

    const bodyMat = new THREE.MeshStandardMaterial({
      color: colorHex,
      roughness: 0.35,
      metalness: 0.25,
    });
    const body = new THREE.Mesh(this.bodyGeo, bodyMat);
    body.position.y = CAR_HEIGHT * 0.44;
    body.castShadow = true;
    body.receiveShadow = true;
    carGroup.add(body);
    carGroup.userData.bodyMesh = body;

    const cabin = new THREE.Mesh(this.cabinGeo, this.cabinMat);
    cabin.position.set(0, CAR_HEIGHT * 0.86, -0.1);
    cabin.castShadow = true;
    carGroup.add(cabin);

    // 4 wheels (shared geometry & material)
    for (const wx of [-CAR_WIDTH * 0.48, CAR_WIDTH * 0.48]) {
      for (const wz of [-CAR_LENGTH * 0.3, CAR_LENGTH * 0.3]) {
        const wheel = new THREE.Mesh(this.wheelGeo, this.wheelMat);
        wheel.position.set(wx, 0.18, wz);
        carGroup.add(wheel);
      }
    }

    // Headlights (front: +Z local)
    for (const side of [-0.32, 0.32]) {
      const hl = new THREE.Mesh(this.lightGeo, this.headlightMat);
      hl.position.set(side, CAR_HEIGHT * 0.44, CAR_LENGTH / 2 + 0.01);
      carGroup.add(hl);
    }

    // Brake lights (rear: -Z local, per-car material so brake brightness is independent)
    const tailMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0xef4444,
      emissiveIntensity: 0.9,
      roughness: 0.3,
    });
    const brakeLights = [];
    for (const side of [-0.32, 0.32]) {
      const tl = new THREE.Mesh(this.lightGeo, tailMat);
      tl.position.set(side, CAR_HEIGHT * 0.44, -CAR_LENGTH / 2 - 0.01);
      carGroup.add(tl);
      brakeLights.push(tl);
    }
    carGroup.userData.brakeLights = brakeLights;

    // Left-turn signal blinkers on true driver's LEFT (+X local) front & rear
    const turnMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xf59e0b,
      emissiveIntensity: 0.0,
    });
    const turnSignals = [];
    for (const zOff of [CAR_LENGTH / 2 - 0.12, -CAR_LENGTH / 2 + 0.12]) {
      const sig = new THREE.Mesh(this.turnGeo, turnMat);
      sig.position.set(CAR_WIDTH / 2 + 0.01, CAR_HEIGHT * 0.44, zOff);
      carGroup.add(sig);
      turnSignals.push(sig);
    }
    carGroup.userData.turnSignals = turnSignals;

    // Optional ground hazard ring (hidden unless straggler or collision)
    const hazardRingMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    });
    const hazardRing = new THREE.Mesh(this.hazardRingGeo, hazardRingMat);
    hazardRing.position.y = 0.025;
    hazardRing.visible = false;
    carGroup.add(hazardRing);
    carGroup.userData.hazardRing = hazardRing;

    carGroup.userData.perCarMaterials = [bodyMat, tailMat, turnMat, hazardRingMat];

    this.group.add(carGroup);
    return { mesh: carGroup, isTurn, lane };
  }

  _removeCarFromScene(car) {
    if (!car || !car.mesh) return;
    this.group.remove(car.mesh);
    const mats = car.mesh.userData?.perCarMaterials || [];
    for (const mat of mats) {
      if (mat && typeof mat.dispose === 'function') {
        mat.dispose();
      }
    }
  }

  _seededRng(seed) {
    let s = seed;
    return () => {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  }
}
