/**
 * Realistic 60 FPS 2D 4-Way Intersection Canvas Renderer.
 * Renders concrete curbs, matte asphalt, zebra crosswalks, lane arrows,
 * 3-lens Red/Yellow/Green signal housings, interactive camera zoom (45%-125%),
 * direct per-lane click/hover hit testing, and 1-to-1 synced vehicles
 * with braking lights, turn signals, circular-arc left turns, and
 * synchronized culprit-victim collision physics.
 */

const MAX_EXITING_SPRITES = 160;
const MAX_COLLIDING_SPRITES = 24;

const LANE_SPECIFICATIONS = [
  // 0: N_STRAIGHT (heading South +y, outer right-hand lane x = cx - 72)
  { id: 0, approach: 'N', isTurn: false, startHeading: Math.PI / 2, label: 'N Straight' },
  // 1: N_TURN (heading South +y -> turns East +x)
  { id: 1, approach: 'N', isTurn: true, startHeading: Math.PI / 2, label: 'N Turn' },
  // 2: S_STRAIGHT (heading North -y, outer right-hand lane x = cx + 72)
  { id: 2, approach: 'S', isTurn: false, startHeading: -Math.PI / 2, label: 'S Straight' },
  // 3: S_TURN (heading North -y -> turns West -x)
  { id: 3, approach: 'S', isTurn: true, startHeading: -Math.PI / 2, label: 'S Turn' },
  // 4: E_STRAIGHT (heading West -x, outer right-hand lane y = cy - 72)
  { id: 4, approach: 'E', isTurn: false, startHeading: Math.PI, label: 'E Straight' },
  // 5: E_TURN (heading West -x -> turns South +y)
  { id: 5, approach: 'E', isTurn: true, startHeading: Math.PI, label: 'E Turn' },
  // 6: W_STRAIGHT (heading East +x, outer right-hand lane y = cy + 72)
  { id: 6, approach: 'W', isTurn: false, startHeading: 0, label: 'W Straight' },
  // 7: W_TURN (heading East +x -> turns North -y)
  { id: 7, approach: 'W', isTurn: true, startHeading: 0, label: 'W Turn' },
];

const VEHICLE_BODY_COLORS = [
  '#2563EB', // Sapphire Blue
  '#DC2626', // Crimson Red
  '#D97706', // Warm Amber
  '#16A34A', // Emerald Green
  '#1D1D1F', // Apple Space Black
  '#475569', // Slate Gray
];

export class IntersectionRenderer {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.options = options || {};
    this.compact = Boolean(this.options.compact);
    this.logicalWidth = 700;
    this.logicalHeight = 580;
    this.centerX = 350;
    this.centerY = 290;
    this.roadHalfWidth = 96;
    this.laneWidth = 48;
    this.stopOffset = 124;

    this.zoomLevel = 1.0;
    this.minZoom = 0.45;
    this.maxZoom = 1.25;
    this.onZoomChange = null;
    this.onLaneClick = null;
    this.hoveredLaneId = null;

    this.queuedByLane = Array.from({ length: 8 }, () => []);
    this.exitingVehicles = [];
    this.collidingVehicles = [];
    this.collisionEffects = [];
    this.nextVehicleId = 1;
    this.hasSyncedOnce = false;
    this.lastStepSeen = -1;
    this.lastMutationSeq = -1;
    this.currentStepsPerSecond = 2.0;
    this.latestSnapshot = null;
    this.animationTimestampMs = performance.now();

    this._configureHiDpi();
    if (!this.compact) {
      this._bindCanvasInteractions();
    }
  }

  _configureHiDpi() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = this.logicalWidth * ratio;
    this.canvas.height = this.logicalHeight * ratio;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  _bindCanvasInteractions() {
    if (!this.canvas || typeof this.canvas.addEventListener !== 'function') {
      return;
    }

    this.canvas.addEventListener('mousemove', (event) => {
      const world = this.screenToWorld(event.clientX, event.clientY);
      const hitLane = this._hitTestLane(world.x, world.y);
      this.hoveredLaneId = hitLane;
      this.canvas.style.cursor = hitLane !== null ? 'pointer' : 'default';
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.hoveredLaneId = null;
      this.canvas.style.cursor = 'default';
    });

    this.canvas.addEventListener('click', (event) => {
      const world = this.screenToWorld(event.clientX, event.clientY);
      const hitLane = this._hitTestLane(world.x, world.y);
      this.hoveredLaneId = hitLane;
      if (hitLane !== null && typeof this.onLaneClick === 'function') {
        this.onLaneClick(hitLane);
      }
    });

    this.canvas.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        const delta = event.deltaY < 0 ? 0.1 : -0.1;
        const targetZoom = Math.round((this.zoomLevel + delta) * 100) / 100;
        const previousZoom = this.zoomLevel;
        this.setZoom(targetZoom);
        if (
          Math.abs(this.zoomLevel - previousZoom) <= 1e-6 &&
          typeof this.onZoomChange === 'function'
        ) {
          this.onZoomChange(this.zoomLevel);
        }
      },
      { passive: false }
    );
  }

  getZoom() {
    return this.zoomLevel;
  }

  setZoom(level) {
    const numeric = Number(level);
    if (!Number.isFinite(numeric)) {
      return this.zoomLevel;
    }
    const clamped = Math.min(this.maxZoom, Math.max(this.minZoom, numeric));
    const rounded = Math.round(clamped * 1000) / 1000;
    const changed = Math.abs(rounded - this.zoomLevel) > 1e-6;
    this.zoomLevel = rounded;
    if (changed && typeof this.onZoomChange === 'function') {
      this.onZoomChange(this.zoomLevel);
    }
    return this.zoomLevel;
  }

  screenToWorld(clientX, clientY) {
    const rect =
      this.canvas && typeof this.canvas.getBoundingClientRect === 'function'
        ? this.canvas.getBoundingClientRect()
        : null;
    const rectWidth = rect && rect.width > 0 ? rect.width : this.logicalWidth;
    const rectHeight = rect && rect.height > 0 ? rect.height : this.logicalHeight;
    const left = rect ? rect.left : 0;
    const top = rect ? rect.top : 0;

    const sx = ((clientX - left) / rectWidth) * this.logicalWidth;
    const sy = ((clientY - top) / rectHeight) * this.logicalHeight;
    const wx = this.centerX + (sx - this.centerX) / this.zoomLevel;
    const wy = this.centerY + (sy - this.centerY) / this.zoomLevel;
    return { x: wx, y: wy };
  }

  _hitTestLane(wx, wy) {
    const cx = this.centerX;
    const cy = this.centerY;
    const stopOffset = this.stopOffset;

    if (wy < cy - stopOffset && wx >= cx - 96 && wx < cx - 48) {
      return 0;
    }
    if (wy < cy - stopOffset && wx >= cx - 48 && wx < cx) {
      return 1;
    }
    if (wy > cy + stopOffset && wx > cx + 48 && wx <= cx + 96) {
      return 2;
    }
    if (wy > cy + stopOffset && wx >= cx && wx <= cx + 48) {
      return 3;
    }
    if (wx > cx + stopOffset && wy >= cy - 96 && wy < cy - 48) {
      return 4;
    }
    if (wx > cx + stopOffset && wy >= cy - 48 && wy < cy) {
      return 5;
    }
    if (wx < cx - stopOffset && wy > cy + 48 && wy <= cy + 96) {
      return 6;
    }
    if (wx < cx - stopOffset && wy >= cy && wy <= cy + 48) {
      return 7;
    }
    return null;
  }

  _getWorldBounds() {
    const worldHalfW = (this.logicalWidth / 2) / this.zoomLevel;
    const worldHalfH = (this.logicalHeight / 2) / this.zoomLevel;
    return {
      halfW: worldHalfW,
      halfH: worldHalfH,
      minX: this.centerX - worldHalfW,
      maxX: this.centerX + worldHalfW,
      minY: this.centerY - worldHalfH,
      maxY: this.centerY + worldHalfH,
    };
  }

  applyBackendFrame(framePayload) {
    if (!framePayload) {
      return;
    }

    const snapshot = framePayload.snapshot ? framePayload.snapshot : framePayload;
    if (
      !snapshot ||
      typeof snapshot !== 'object' ||
      (snapshot.step === undefined && !Array.isArray(snapshot.lane_queues))
    ) {
      return;
    }

    this.latestSnapshot = snapshot;

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

    const hasManualSpawns = manualSpawns.some((count) => Number(count) > 0);
    const mutationChanged = hasMutationSeq && mutationSeq !== this.lastMutationSeq;

    if (!this.hasSyncedOnce || currentStep < this.lastStepSeen) {
      this._resetSceneQueues(laneQueues, currentStep, collisionEvents, snapshot);
      this._syncStragglers(activeStragglerLanes, stepDurationSec);
      this.hasSyncedOnce = true;
      this.lastStepSeen = currentStep;
      this.lastMutationSeq = mutationSeq;
      return;
    }

    if (currentStep === this.lastStepSeen) {
      if (currentStep === 0 && mutationChanged && !hasManualSpawns) {
        const anyGrew = laneQueues.some(
          (target, idx) => Number(target || 0) > this.queuedByLane[idx].length
        );
        if (!anyGrew) {
          this._resetSceneQueues(laneQueues, currentStep, collisionEvents, snapshot);
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

      this._reconcileManualSpawns(laneQueues, manualSpawns);
      this.lastMutationSeq = mutationSeq;
      return;
    }

    // Shift head-of-line culprits for red_light_conflict before queue reconciliation.
    const redLightCulpritsByLane = Array.from({ length: 8 }, () => []);
    for (const ev of collisionEvents) {
      if (!ev || ev.kind !== 'red_light_conflict') {
        continue;
      }
      const culpritLane = Number(ev.culprit_lane);
      if (
        Number.isInteger(culpritLane) &&
        culpritLane >= 0 &&
        culpritLane <= 7 &&
        this.queuedByLane[culpritLane].length > 0
      ) {
        redLightCulpritsByLane[culpritLane].push(
          this.queuedByLane[culpritLane].shift()
        );
      }
    }

    for (let laneIndex = 0; laneIndex < 8; laneIndex += 1) {
      const queue = this.queuedByLane[laneIndex];
      const departedCount = Math.min(Number(stepDepartures[laneIndex] || 0), queue.length);

      for (let i = 0; i < departedCount; i += 1) {
        const departingVehicle = queue.shift();
        this._spawnExitingVehicle(departingVehicle, laneIndex, i, stepDurationSec);
      }

      const arrivedCount = Number(stepArrivals[laneIndex] || 0);
      for (let i = 0; i < arrivedCount; i += 1) {
        const tailDistance = queue.length > 0 ? queue[queue.length - 1].distFromStop : 180;
        const spawnDistance = Math.max(210, tailDistance + 30);
        queue.push(this._createQueuedVehicle(laneIndex, spawnDistance, 110));
      }

      const targetLength = Number(laneQueues[laneIndex] || 0);
      while (queue.length > targetLength) {
        queue.pop();
      }
      while (queue.length < targetLength) {
        const tailDistance = queue.length > 0 ? queue[queue.length - 1].distFromStop : 160;
        const spawnDistance = Math.max(210, tailDistance + 30);
        queue.push(this._createQueuedVehicle(laneIndex, spawnDistance, 115));
      }
    }

    this._processCollisionEvents(
      collisionEvents,
      stepDurationSec,
      redLightCulpritsByLane
    );
    this._syncStragglers(activeStragglerLanes, stepDurationSec);
    this.lastStepSeen = currentStep;
    this.lastMutationSeq = mutationSeq;
  }

  _resetSceneQueues(laneQueues, currentStep, collisionEvents, snapshot = null) {
    this.exitingVehicles = [];
    this.collidingVehicles = [];
    this.collisionEffects = [];
    for (let laneIndex = 0; laneIndex < 8; laneIndex += 1) {
      this.queuedByLane[laneIndex] = [];
      const initialCount = Number(laneQueues[laneIndex] || 0);
      for (let slot = 0; slot < initialCount; slot += 1) {
        this.queuedByLane[laneIndex].push(
          this._createQueuedVehicle(laneIndex, 12 + slot * 26, 0)
        );
      }
    }
    if (currentStep > 0) {
      const stepDurationSec = 1.0 / Math.max(this.currentStepsPerSecond, 0.5);
      this._processCollisionEvents(collisionEvents, stepDurationSec);
    }
  }

  _reconcileManualSpawns(laneQueues, manualSpawns) {
    const bounds = this._getWorldBounds();
    const minSpawnBase = Math.max(
      210,
      Math.round(Math.max(bounds.halfW, bounds.halfH) - this.stopOffset + 24)
    );

    for (let laneIndex = 0; laneIndex < 8; laneIndex += 1) {
      const queue = this.queuedByLane[laneIndex];
      const targetLength = Math.max(0, Number(laneQueues[laneIndex] || 0));
      const hintCount = Math.max(0, Number(manualSpawns[laneIndex] || 0));

      while (queue.length > targetLength) {
        queue.pop();
      }

      const needed = targetLength - queue.length;
      const smoothBatch = Math.max(needed, hintCount);
      for (let i = 0; i < needed; i += 1) {
        const tailDistance = queue.length > 0 ? queue[queue.length - 1].distFromStop : 150;
        const entryOffset = i === 0 && smoothBatch > 0 ? minSpawnBase : tailDistance + 30;
        const spawnDistance = Math.max(210, tailDistance + 30, entryOffset);
        queue.push(this._createQueuedVehicle(laneIndex, spawnDistance, 125));
      }
    }
  }

  _syncStragglers(activeStragglerLanes, stepDurationSec) {
    const stragglerSet = new Set(activeStragglerLanes);

    for (const vehicle of this.exitingVehicles) {
      if (vehicle.isStraggler && !stragglerSet.has(vehicle.laneIndex)) {
        vehicle.isStraggler = false;
        vehicle.delayRemaining = 0;
        const remainingClearPx = Math.max(
          20,
          (vehicle.boxClearPx || 260) - vehicle.progressPx
        );
        vehicle.cruiseSpeed = Math.max(
          220,
          remainingClearPx / Math.max(0.05, stepDurationSec * 0.75)
        );
        vehicle.speed = Math.max(vehicle.speed, vehicle.cruiseSpeed);
      }
    }

    for (const laneIndex of stragglerSet) {
      const alreadyTracked = this.exitingVehicles.some(
        (v) => v.laneIndex === laneIndex && v.isStraggler
      );
      if (alreadyTracked) {
        continue;
      }

      let candidate = null;
      for (let i = this.exitingVehicles.length - 1; i >= 0; i -= 1) {
        const item = this.exitingVehicles[i];
        if (
          item.laneIndex === laneIndex &&
          item.progressPx <= (item.boxEntryHoldPx || 9999) + 18
        ) {
          candidate = item;
          break;
        }
      }

      if (candidate) {
        candidate.isStraggler = true;
        candidate.delayRemaining = 0;
        candidate.progressPx = Math.min(
          candidate.progressPx,
          candidate.boxEntryHoldPx || candidate.progressPx
        );
        continue;
      }

      const startDist = 12;
      const path = this._buildCrossingPath(laneIndex, startDist);
      const boxClearPx = this._computeBoxClearDistance(path, startDist);
      const boxEntryHoldPx = this._computeBoxEntryHoldDistance(startDist);
      if (this.exitingVehicles.length >= MAX_EXITING_SPRITES) {
        this.exitingVehicles.shift();
      }
      this.exitingVehicles.push({
        id: this.nextVehicleId,
        laneIndex,
        color: VEHICLE_BODY_COLORS[(this.nextVehicleId + laneIndex) % VEHICLE_BODY_COLORS.length],
        isTurn: Boolean(LANE_SPECIFICATIONS[laneIndex]?.isTurn),
        isStraggler: true,
        path,
        progressPx: Math.max(0, boxEntryHoldPx - 28),
        boxClearPx,
        boxEntryHoldPx,
        cruiseSpeed: 120,
        speed: 90,
        delayRemaining: 0,
      });
      this.nextVehicleId += 1;
    }
  }

  _processCollisionEvents(
    events,
    stepDurationSec = 0.5,
    redLightCulpritsByLane = null
  ) {
    if (!Array.isArray(events) || events.length === 0) {
      return;
    }

    const approachSec = Math.min(0.24, stepDurationSec * 0.42);
    const haltSec = Math.max(0.85, Math.min(1.5, stepDurationSec * 2.1));

    for (const ev of events) {
      if (!ev || typeof ev !== 'object') {
        continue;
      }
      const culpritLane = Number(ev.culprit_lane);
      const victimLane = Number(ev.victim_lane);
      if (!Number.isInteger(culpritLane) || culpritLane < 0 || culpritLane > 7) {
        continue;
      }
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

      const culpritStartDist = culpritSource ? Number(culpritSource.distFromStop || 12) : 12;
      const culpritPath = culpritSource?.path || this._buildCrossingPath(culpritLane, culpritStartDist);
      const culpritStartProgress = culpritSource?.path ? Number(culpritSource.progressPx || 0) : 0;

      const victimSource = hasValidVictim ? this._takeVictimCandidate(victimLane) : null;
      const victimStartDist = victimSource ? Number(victimSource.distFromStop || 12) : 12;
      const victimPath = victimSource?.path || (
        hasValidVictim ? this._buildCrossingPath(victimLane, victimStartDist) : null
      );
      const victimStartProgress = victimSource?.path ? Number(victimSource.progressPx || 0) : 0;

      const conflictGeo = victimPath
        ? this._computeConflictGeometry(culpritPath, victimPath)
        : this._fallbackConflictGeometry(culpritPath);

      const culpritImpactProgress = Math.max(0, conflictGeo.culpritProgressPx - 9);
      const victimImpactProgress = Math.max(0, conflictGeo.victimProgressPx - 9);

      this._pushCollidingVehicle({
        id: culpritSource?.id || this._allocateVehicleId(),
        laneIndex: culpritLane,
        role: 'culprit',
        isTurn: Boolean(LANE_SPECIFICATIONS[culpritLane]?.isTurn),
        path: culpritPath,
        startProgressPx: Math.min(culpritStartProgress, culpritImpactProgress),
        progressPx: Math.min(culpritStartProgress, culpritImpactProgress),
        impactProgressPx: culpritImpactProgress,
        approachSec,
        haltSec,
        phaseElapsed: 0,
        state: 'approaching',
        braking: false,
        yawOffset: 0,
        impactYawOffset: -0.14,
        speed: 220,
        color: '#dc2626',
      });

      if (hasValidVictim && victimPath) {
        this._pushCollidingVehicle({
          id: victimSource?.id || this._allocateVehicleId(),
          laneIndex: victimLane,
          role: 'victim',
          isTurn: Boolean(LANE_SPECIFICATIONS[victimLane]?.isTurn),
          path: victimPath,
          startProgressPx: Math.min(victimStartProgress, victimImpactProgress),
          progressPx: Math.min(victimStartProgress, victimImpactProgress),
          impactProgressPx: victimImpactProgress,
          approachSec,
          haltSec,
          phaseElapsed: 0,
          state: 'approaching',
          braking: false,
          yawOffset: 0,
          impactYawOffset: 0.14,
          speed: 220,
          color:
            victimSource?.color ||
            VEHICLE_BODY_COLORS[(victimLane + 1) % VEHICLE_BODY_COLORS.length],
        });
      }

      const culpritLabel = LANE_SPECIFICATIONS[culpritLane]?.label || `Lane ${culpritLane}`;
      const victimLabel = hasValidVictim
        ? LANE_SPECIFICATIONS[victimLane]?.label || `Lane ${victimLane}`
        : 'Box';

      this.collisionEffects.push({
        x: conflictGeo.x,
        y: conflictGeo.y,
        kind,
        culpritLane,
        victimLane,
        culpritLabel,
        victimLabel,
        delayRemaining: approachSec,
        remainingSec: 2.2,
        totalSec: 2.2,
      });
    }
  }

  _allocateVehicleId() {
    const id = this.nextVehicleId;
    this.nextVehicleId += 1;
    return id;
  }

  _pushCollidingVehicle(entry) {
    if (this.collidingVehicles.length >= MAX_COLLIDING_SPRITES) {
      this.collidingVehicles.shift();
    }
    this.collidingVehicles.push(entry);
  }

  _takeStragglerCulprit(laneIndex) {
    for (let i = 0; i < this.exitingVehicles.length; i += 1) {
      const candidate = this.exitingVehicles[i];
      if (candidate.laneIndex === laneIndex && candidate.isStraggler) {
        this.exitingVehicles.splice(i, 1);
        return candidate;
      }
    }
    for (let i = this.exitingVehicles.length - 1; i >= 0; i -= 1) {
      const candidate = this.exitingVehicles[i];
      if (
        candidate.laneIndex === laneIndex &&
        candidate.progressPx <= (candidate.boxClearPx || 260) * 0.75
      ) {
        this.exitingVehicles.splice(i, 1);
        return candidate;
      }
    }
    return null;
  }

  _takeVictimCandidate(victimLane) {
    let bestIndex = -1;
    let lowestProgress = Infinity;
    for (let i = 0; i < this.exitingVehicles.length; i += 1) {
      const candidate = this.exitingVehicles[i];
      if (
        candidate.laneIndex === victimLane &&
        !candidate.isStraggler &&
        candidate.progressPx < lowestProgress
      ) {
        lowestProgress = candidate.progressPx;
        bestIndex = i;
      }
    }
    if (bestIndex >= 0) {
      const [picked] = this.exitingVehicles.splice(bestIndex, 1);
      return picked;
    }
    return null;
  }

  _samplePathPointsWithProgress(path, stepPx = 2) {
    const cx = this.centerX;
    const cy = this.centerY;
    const totalLength = Math.max(this._estimatePathLength(path), 1);
    const boxMargin = this.roadHalfWidth + 10;
    const boxPoints = [];
    const maxSamplePx = Math.min(totalLength, 520);

    for (let px = 0; px <= maxSamplePx; px += stepPx) {
      const pose = this._samplePathPose(path, px);
      if (
        Math.abs(pose.x - cx) <= boxMargin &&
        Math.abs(pose.y - cy) <= boxMargin
      ) {
        boxPoints.push({ x: pose.x, y: pose.y, progress: px });
      }
    }
    return boxPoints;
  }

  _computeConflictGeometry(culpritPath, victimPath) {
    const culpritPoints = this._samplePathPointsWithProgress(culpritPath, 2);
    const victimPoints = this._samplePathPointsWithProgress(victimPath, 2);

    let bestDistSq = Infinity;
    let bestPair = null;
    for (const pA of culpritPoints) {
      for (const pB of victimPoints) {
        const dx = pA.x - pB.x;
        const dy = pA.y - pB.y;
        const distSq = dx * dx + dy * dy;
        if (distSq < bestDistSq) {
          bestDistSq = distSq;
          bestPair = { pA, pB };
        }
      }
    }

    if (bestPair) {
      return {
        x: (bestPair.pA.x + bestPair.pB.x) / 2,
        y: (bestPair.pA.y + bestPair.pB.y) / 2,
        culpritProgressPx: bestPair.pA.progress,
        victimProgressPx: bestPair.pB.progress,
      };
    }

    return this._fallbackConflictGeometry(culpritPath);
  }

  _fallbackConflictGeometry(culpritPath) {
    const culpritPoints = this._samplePathPointsWithProgress(culpritPath, 2);
    if (culpritPoints.length > 0) {
      const mid = culpritPoints[Math.floor(culpritPoints.length / 2)];
      return {
        x: mid.x,
        y: mid.y,
        culpritProgressPx: mid.progress,
        victimProgressPx: mid.progress,
      };
    }
    return {
      x: this.centerX,
      y: this.centerY,
      culpritProgressPx: 96,
      victimProgressPx: 96,
    };
  }

  _computeConflictPoint(culpritLane, victimLane) {
    const culpritPath = this._buildCrossingPath(culpritLane, 12);
    const hasValidVictim =
      Number.isInteger(victimLane) &&
      victimLane >= 0 &&
      victimLane <= 7 &&
      victimLane !== culpritLane;
    if (hasValidVictim) {
      const victimPath = this._buildCrossingPath(victimLane, 12);
      const geo = this._computeConflictGeometry(culpritPath, victimPath);
      return { x: geo.x, y: geo.y };
    }
    const geo = this._fallbackConflictGeometry(culpritPath);
    return { x: geo.x, y: geo.y };
  }

  _createQueuedVehicle(laneIndex, distFromStop, initialSpeed) {
    const spec = LANE_SPECIFICATIONS[laneIndex];
    const color =
      VEHICLE_BODY_COLORS[(this.nextVehicleId + laneIndex) % VEHICLE_BODY_COLORS.length];
    const vehicle = {
      id: this.nextVehicleId,
      laneIndex,
      isTurn: spec.isTurn,
      distFromStop,
      speed: initialSpeed,
      braking: initialSpeed < 10,
      color,
    };
    this.nextVehicleId += 1;
    return vehicle;
  }

  _computeBoxEntryHoldDistance(startDistFromStop) {
    return Math.max(0, Number(startDistFromStop || 0) + (this.stopOffset - 92));
  }

  _computeBoxClearDistance(path, startDistFromStop) {
    if (path.kind === 'line') {
      return Number(startDistFromStop || 0) + this.stopOffset + (this.roadHalfWidth + 18);
    }
    if (path.kind === 'turn_arc') {
      return path.entry.len + path.arc.len + 18;
    }
    return Math.min(this._estimatePathLength(path), 320);
  }

  _spawnExitingVehicle(
    queuedVehicle,
    laneIndex,
    slotInBatch = 0,
    stepDurationSec = 0.5
  ) {
    if (this.exitingVehicles.length >= MAX_EXITING_SPRITES) {
      this.exitingVehicles.shift();
    }
    const startDist = Number(queuedVehicle.distFromStop || 12);
    const path = this._buildCrossingPath(laneIndex, startDist);
    const boxClearPx = this._computeBoxClearDistance(path, startDist);
    const boxEntryHoldPx = this._computeBoxEntryHoldDistance(startDist);
    const delaySeconds =
      Number(slotInBatch || 0) * Math.min(0.1, stepDurationSec * 0.16);
    const targetClearSec = Math.max(0.05, stepDurationSec * 1.15 - delaySeconds);
    const cruiseSpeed = Math.max(185, boxClearPx / targetClearSec);

    this.exitingVehicles.push({
      id: queuedVehicle.id,
      laneIndex,
      distFromStop: startDist,
      color: queuedVehicle.color,
      isTurn: queuedVehicle.isTurn,
      isStraggler: false,
      path,
      progressPx: 0,
      boxClearPx,
      boxEntryHoldPx,
      cruiseSpeed,
      speed: cruiseSpeed,
      delayRemaining: delaySeconds,
    });
  }

  _buildCrossingPath(laneIndex, startDistFromStop) {
    const cx = this.centerX;
    const cy = this.centerY;
    const stopOffset = this.stopOffset;
    const boxEdge = this.roadHalfWidth;
    const turnRadius = 120;
    const stop = stopOffset + startDistFromStop;
    const bounds = this._getWorldBounds();
    const exitMinX = Math.min(bounds.minX, 0) - 48;
    const exitMaxX = Math.max(bounds.maxX, this.logicalWidth) + 48;
    const exitMinY = Math.min(bounds.minY, 0) - 48;
    const exitMaxY = Math.max(bounds.maxY, this.logicalHeight) + 48;

    if (laneIndex === 0) {
      return {
        kind: 'line',
        x0: cx - 72,
        y0: cy - stop,
        x1: cx - 72,
        y1: exitMaxY,
      };
    }
    if (laneIndex === 2) {
      return {
        kind: 'line',
        x0: cx + 72,
        y0: cy + stop,
        x1: cx + 72,
        y1: exitMinY,
      };
    }
    if (laneIndex === 4) {
      return {
        kind: 'line',
        x0: cx + stop,
        y0: cy - 72,
        x1: exitMinX,
        y1: cy - 72,
      };
    }
    if (laneIndex === 6) {
      return {
        kind: 'line',
        x0: cx - stop,
        y0: cy + 72,
        x1: exitMaxX,
        y1: cy + 72,
      };
    }

    // Left turns: straight approach -> exact 90-degree R=120 circular arc -> straight exit.
    if (laneIndex === 1) {
      return this._makeTurnArc(
        { x: cx - 24, y: cy - stop },
        { x: cx - 24, y: cy - boxEdge },
        { x: cx + boxEdge, y: cy - boxEdge },
        turnRadius,
        Math.PI,
        Math.PI / 2,
        { x: cx + boxEdge, y: cy + 24 },
        { x: exitMaxX, y: cy + 24 }
      );
    }
    if (laneIndex === 3) {
      return this._makeTurnArc(
        { x: cx + 24, y: cy + stop },
        { x: cx + 24, y: cy + boxEdge },
        { x: cx - boxEdge, y: cy + boxEdge },
        turnRadius,
        0,
        -Math.PI / 2,
        { x: cx - boxEdge, y: cy - 24 },
        { x: exitMinX, y: cy - 24 }
      );
    }
    if (laneIndex === 5) {
      return this._makeTurnArc(
        { x: cx + stop, y: cy - 24 },
        { x: cx + boxEdge, y: cy - 24 },
        { x: cx + boxEdge, y: cy + boxEdge },
        turnRadius,
        -Math.PI / 2,
        -Math.PI,
        { x: cx - 24, y: cy + boxEdge },
        { x: cx - 24, y: exitMaxY }
      );
    }

    return this._makeTurnArc(
      { x: cx - stop, y: cy + 24 },
      { x: cx - boxEdge, y: cy + 24 },
      { x: cx - boxEdge, y: cy - boxEdge },
      turnRadius,
      Math.PI / 2,
      0,
      { x: cx + 24, y: cy - boxEdge },
      { x: cx + 24, y: exitMinY }
    );
  }

  _makeTurnArc(startPt, arcStartPt, arcCenter, radius, a0, a1, arcEndPt, endPt) {
    const entryLen = Math.hypot(arcStartPt.x - startPt.x, arcStartPt.y - startPt.y);
    const arcLen = radius * Math.abs(a1 - a0);
    const exitLen = Math.hypot(endPt.x - arcEndPt.x, endPt.y - arcEndPt.y);

    return {
      kind: 'turn_arc',
      entry: {
        x0: startPt.x,
        y0: startPt.y,
        x1: arcStartPt.x,
        y1: arcStartPt.y,
        len: entryLen,
      },
      arc: {
        cx: arcCenter.x,
        cy: arcCenter.y,
        radius,
        a0,
        a1,
        len: Math.max(arcLen, 1),
      },
      exit: {
        x0: arcEndPt.x,
        y0: arcEndPt.y,
        x1: endPt.x,
        y1: endPt.y,
        len: Math.max(exitLen, 1),
      },
    };
  }

  renderFrame(timestampMs, drawScene = true) {
    const dtSeconds = Math.min(
      Math.max((timestampMs - this.animationTimestampMs) / 1000, 0),
      0.05
    );
    this.animationTimestampMs = timestampMs;

    this._stepVehicleKinematics(dtSeconds);
    if (drawScene) {
      this._drawScene(timestampMs);
    }
  }

  _stepVehicleKinematics(dtSeconds) {
    const stepDurationSec = 1.0 / Math.max(this.currentStepsPerSecond, 0.5);
    const slotGap = 26;
    const maxApproachSpeed = Math.max(165, 115 * this.currentStepsPerSecond);

    for (let laneIndex = 0; laneIndex < 8; laneIndex += 1) {
      const queue = this.queuedByLane[laneIndex];
      for (let slot = 0; slot < queue.length; slot += 1) {
        const vehicle = queue[slot];
        const targetDist = 12 + slot * slotGap;
        const delta = vehicle.distFromStop - targetDist;

        if (delta > 0.6) {
          const desiredSpeed = Math.min(maxApproachSpeed, delta * 3.8 + 18);
          vehicle.speed += (desiredSpeed - vehicle.speed) * Math.min(1, dtSeconds * 8);
          vehicle.distFromStop = Math.max(
            targetDist,
            vehicle.distFromStop - vehicle.speed * dtSeconds
          );
          vehicle.braking = delta < 22;
        } else {
          vehicle.distFromStop = targetDist;
          vehicle.speed = 0;
          vehicle.braking = true;
        }
      }
    }

    const activeExiting = [];
    for (const vehicle of this.exitingVehicles) {
      if (vehicle.delayRemaining > 0) {
        vehicle.delayRemaining -= dtSeconds;
        activeExiting.push(vehicle);
        continue;
      }

      if (vehicle.isStraggler) {
        const holdPx = vehicle.boxEntryHoldPx || 64;
        if (vehicle.progressPx < holdPx - 2) {
          vehicle.speed = Math.min(
            130,
            Math.max(55, (holdPx - vehicle.progressPx) * 2.5)
          );
          vehicle.progressPx = Math.min(
            holdPx,
            vehicle.progressPx + vehicle.speed * dtSeconds
          );
        } else {
          vehicle.speed = 10;
          vehicle.progressPx = Math.min(
            holdPx + 6,
            vehicle.progressPx + vehicle.speed * dtSeconds
          );
        }
        activeExiting.push(vehicle);
        continue;
      }

      const minSyncSpeed = Math.max(
        185,
        (vehicle.boxClearPx || 260) / Math.max(0.05, stepDurationSec * 1.15)
      );
      vehicle.speed = Math.max(vehicle.cruiseSpeed || 185, minSyncSpeed);
      const approxTotalLength = this._estimatePathLength(vehicle.path);
      vehicle.progressPx += vehicle.speed * dtSeconds;
      if (vehicle.progressPx < approxTotalLength) {
        activeExiting.push(vehicle);
      }
    }
    if (activeExiting.length > MAX_EXITING_SPRITES) {
      this.exitingVehicles = activeExiting.slice(-MAX_EXITING_SPRITES);
    } else {
      this.exitingVehicles = activeExiting;
    }

    const activeColliding = [];
    for (const vehicle of this.collidingVehicles) {
      const approxTotalLength = this._estimatePathLength(vehicle.path);
      if (vehicle.state === 'approaching') {
        vehicle.phaseElapsed += dtSeconds;
        const ratio = Math.min(
          1,
          vehicle.phaseElapsed / Math.max(vehicle.approachSec, 0.01)
        );
        vehicle.progressPx =
          vehicle.startProgressPx +
          (vehicle.impactProgressPx - vehicle.startProgressPx) * ratio;
        vehicle.braking = ratio > 0.72;
        vehicle.yawOffset = 0;
        if (ratio >= 1) {
          vehicle.state = 'halted';
          vehicle.phaseElapsed = 0;
          vehicle.progressPx = vehicle.impactProgressPx;
          vehicle.braking = true;
          vehicle.yawOffset = vehicle.impactYawOffset || 0;
        }
        activeColliding.push(vehicle);
        continue;
      }

      if (vehicle.state === 'halted') {
        vehicle.phaseElapsed += dtSeconds;
        vehicle.braking = true;
        if (vehicle.phaseElapsed >= vehicle.haltSec) {
          vehicle.state = 'clearing';
          vehicle.phaseElapsed = 0;
          vehicle.braking = false;
          vehicle.yawOffset = 0;
        }
        activeColliding.push(vehicle);
        continue;
      }

      const clearMinSpeed = Math.max(
        220,
        260 / Math.max(0.05, stepDurationSec * 0.9)
      );
      vehicle.speed = Math.max(vehicle.speed || 220, clearMinSpeed);
      vehicle.progressPx += vehicle.speed * dtSeconds;
      if (vehicle.progressPx < approxTotalLength) {
        activeColliding.push(vehicle);
      }
    }
    if (activeColliding.length > MAX_COLLIDING_SPRITES) {
      this.collidingVehicles = activeColliding.slice(-MAX_COLLIDING_SPRITES);
    } else {
      this.collidingVehicles = activeColliding;
    }

    const activeEffects = [];
    for (const effect of this.collisionEffects) {
      if (effect.delayRemaining && effect.delayRemaining > 0) {
        effect.delayRemaining = Math.max(0, effect.delayRemaining - dtSeconds);
        activeEffects.push(effect);
        continue;
      }
      effect.remainingSec -= dtSeconds;
      if (effect.remainingSec > 0) {
        activeEffects.push(effect);
      }
    }
    this.collisionEffects = activeEffects;
  }

  _estimatePathLength(path) {
    if (path.kind === 'line') {
      return Math.hypot(path.x1 - path.x0, path.y1 - path.y0);
    }
    if (path.kind === 'turn_arc') {
      return path.entry.len + path.arc.len + path.exit.len;
    }
    if (path.kind === 'turn_compound') {
      return path.entry.len + path.bezier.len + path.exit.len;
    }
    const seg1 = Math.hypot(path.p1.x - path.p0.x, path.p1.y - path.p0.y);
    const seg2 = Math.hypot(path.p2.x - path.p1.x, path.p2.y - path.p1.y);
    return (seg1 + seg2) * 0.88;
  }

  _drawScene(timestampMs) {
    const ctx = this.ctx;
    const width = this.logicalWidth;
    const height = this.logicalHeight;
    const cx = this.centerX;
    const cy = this.centerY;
    const zoom = this.zoomLevel;
    const bounds = this._getWorldBounds();

    ctx.clearRect(0, 0, width, height);

    // Concrete plaza / sidewalk base fills viewport
    ctx.fillStyle = '#e8eaef';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(zoom, zoom);
    ctx.translate(-cx, -cy);

    // Corner landscape blocks
    this._drawCornerBlocks(ctx, bounds);

    // Matte asphalt crossroad
    this._drawAsphaltAndMarkings(ctx, bounds);

    // Hover highlight over incoming lane corridor
    if (!this.compact && this.hoveredLaneId !== null) {
      this._drawHoveredLaneHighlight(ctx, this.hoveredLaneId, bounds);
    }

    // Signal stop-line color bars + 3-lens housings
    this._drawStopLinesAndSignalHeads(ctx);

    // Queued, exiting, and violating vehicles
    this._drawAllVehicles(ctx, timestampMs);

    // Concentric collision hazard rings and factual conflict banner
    this._drawCollisionEffects(ctx);

    // Crisp click hint label above vehicles when hovering a lane
    if (!this.compact && this.hoveredLaneId !== null) {
      this._drawHoveredLaneLabel(ctx, this.hoveredLaneId, bounds);
    }

    // Lane queue count tags on outer approach edges
    this._drawQueueBadges(ctx, bounds);

    ctx.restore();
  }

  _drawCornerBlocks(ctx, bounds) {
    const cx = this.centerX;
    const cy = this.centerY;
    const half = this.roadHalfWidth + 10;

    const blocks = [
      {
        x: bounds.minX,
        y: bounds.minY,
        w: cx - half - bounds.minX,
        h: cy - half - bounds.minY,
      },
      {
        x: cx + half,
        y: bounds.minY,
        w: bounds.maxX - (cx + half),
        h: cy - half - bounds.minY,
      },
      {
        x: bounds.minX,
        y: cy + half,
        w: cx - half - bounds.minX,
        h: bounds.maxY - (cy + half),
      },
      {
        x: cx + half,
        y: cy + half,
        w: bounds.maxX - (cx + half),
        h: bounds.maxY - (cy + half),
      },
    ];

    for (const blk of blocks) {
      ctx.fillStyle = '#dfe4dc';
      ctx.fillRect(blk.x, blk.y, blk.w, blk.h);
      ctx.strokeStyle = '#c4cad3';
      ctx.lineWidth = 2;
      ctx.strokeRect(blk.x, blk.y, blk.w, blk.h);
    }
  }

  _drawAsphaltAndMarkings(ctx, bounds) {
    const cx = this.centerX;
    const cy = this.centerY;
    const half = this.roadHalfWidth;
    const worldWidth = bounds.maxX - bounds.minX;
    const worldHeight = bounds.maxY - bounds.minY;

    ctx.fillStyle = '#252a33';
    ctx.fillRect(cx - half, bounds.minY, half * 2, worldHeight);
    ctx.fillRect(bounds.minX, cy - half, worldWidth, half * 2);

    // Intersection center asphalt box
    ctx.fillStyle = '#2b313c';
    ctx.fillRect(cx - half, cy - half, half * 2, half * 2);

    // Double yellow center medians
    ctx.strokeStyle = '#eab308';
    ctx.lineWidth = 2;
    const stop = this.stopOffset;

    const medians = [
      [cx - 2, bounds.minY, cx - 2, cy - stop],
      [cx + 2, bounds.minY, cx + 2, cy - stop],
      [cx - 2, cy + stop, cx - 2, bounds.maxY],
      [cx + 2, cy + stop, cx + 2, bounds.maxY],
      [bounds.minX, cy - 2, cx - stop, cy - 2],
      [bounds.minX, cy + 2, cx - stop, cy + 2],
      [cx + stop, cy - 2, bounds.maxX, cy - 2],
      [cx + stop, cy + 2, bounds.maxX, cy + 2],
    ];
    for (const [x1, y1, x2, y2] of medians) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Dashed white lane dividers
    ctx.strokeStyle = 'rgba(241, 245, 249, 0.72)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([10, 10]);

    const dividers = [
      [cx - 48, bounds.minY, cx - 48, cy - stop],
      [cx + 48, bounds.minY, cx + 48, cy - stop],
      [cx - 48, cy + stop, cx - 48, bounds.maxY],
      [cx + 48, cy + stop, cx + 48, bounds.maxY],
      [bounds.minX, cy - 48, cx - stop, cy - 48],
      [bounds.minX, cy + 48, cx - stop, cy + 48],
      [cx + stop, cy - 48, bounds.maxX, cy - 48],
      [cx + stop, cy + 48, bounds.maxX, cy + 48],
    ];
    for (const [x1, y1, x2, y2] of dividers) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Zebra crosswalk stripes around intersection box
    ctx.fillStyle = 'rgba(248, 250, 252, 0.85)';
    for (let offset = -86; offset <= 74; offset += 20) {
      ctx.fillRect(cx + offset, cy - 116, 12, 18);
      ctx.fillRect(cx + offset, cy + 98, 12, 18);
      ctx.fillRect(cx - 116, cy + offset, 18, 12);
      ctx.fillRect(cx + 98, cy + offset, 18, 12);
    }

    // Painted lane directional arrows (straight vs left turn)
    this._drawLaneArrows(ctx);

    // Approach labels
    const textScale = Math.min(1.6, Math.max(1, 1 / this.zoomLevel));
    const fontPx = this.compact ? 10.5 : Math.round(12 * textScale * 10) / 10;
    ctx.fillStyle = '#334155';
    ctx.font = `600 ${fontPx}px "IBM Plex Sans", sans-serif`;
    ctx.textAlign = 'left';
    ctx.fillText('NORTH APPROACH', cx + half + 16, bounds.minY + 22 * textScale);
    ctx.fillText('SOUTH APPROACH', cx + half + 16, bounds.maxY - 14 * textScale);
    ctx.fillText('WEST APPROACH', bounds.minX + 14 * textScale, cy - half - 14 * textScale);
    ctx.textAlign = 'right';
    ctx.fillText('EAST APPROACH', bounds.maxX - 14 * textScale, cy - half - 14 * textScale);
    ctx.textAlign = 'left';
  }

  _getLaneCorridorRect(laneId, bounds) {
    const cx = this.centerX;
    const cy = this.centerY;
    const stop = this.stopOffset;

    switch (laneId) {
      case 0:
        return { x: cx - 96, y: bounds.minY, w: 48, h: cy - stop - bounds.minY };
      case 1:
        return { x: cx - 48, y: bounds.minY, w: 48, h: cy - stop - bounds.minY };
      case 2:
        return { x: cx + 48, y: cy + stop, w: 48, h: bounds.maxY - (cy + stop) };
      case 3:
        return { x: cx, y: cy + stop, w: 48, h: bounds.maxY - (cy + stop) };
      case 4:
        return { x: cx + stop, y: cy - 96, w: bounds.maxX - (cx + stop), h: 48 };
      case 5:
        return { x: cx + stop, y: cy - 48, w: bounds.maxX - (cx + stop), h: 48 };
      case 6:
        return { x: bounds.minX, y: cy + 48, w: cx - stop - bounds.minX, h: 48 };
      case 7:
        return { x: bounds.minX, y: cy, w: cx - stop - bounds.minX, h: 48 };
      default:
        return null;
    }
  }

  _drawHoveredLaneHighlight(ctx, laneId, bounds) {
    const rect = this._getLaneCorridorRect(laneId, bounds);
    if (!rect) {
      return;
    }
    ctx.save();
    ctx.fillStyle = 'rgba(37, 99, 235, 0.24)';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.strokeStyle = 'rgba(96, 165, 250, 0.85)';
    ctx.lineWidth = 2;
    ctx.strokeRect(rect.x + 1, rect.y + 1, Math.max(0, rect.w - 2), Math.max(0, rect.h - 2));
    ctx.restore();
  }

  _drawHoveredLaneLabel(ctx, laneId, bounds) {
    const spec = LANE_SPECIFICATIONS[laneId];
    if (!spec) {
      return;
    }
    const cx = this.centerX;
    const cy = this.centerY;
    const stop = this.stopOffset;
    const scale = Math.min(1.75, Math.max(1, 1 / this.zoomLevel));
    const labelText = `${spec.label}: Click to add vehicle(s)`;

    ctx.save();
    ctx.font = `600 ${Math.round(11 * scale)}px "IBM Plex Sans", sans-serif`;
    const textW = ctx.measureText(labelText).width;
    const padX = 8 * scale;
    const boxW = textW + padX * 2;
    const boxH = 22 * scale;

    let anchorX = cx;
    let anchorY = cy;
    if (laneId === 0 || laneId === 1) {
      anchorX = cx - 48;
      anchorY = cy - stop - 26 * scale;
    } else if (laneId === 2 || laneId === 3) {
      anchorX = cx + 48;
      anchorY = cy + stop + 26 * scale;
    } else if (laneId === 4 || laneId === 5) {
      anchorX = cx + stop + boxW / 2 + 16 * scale;
      anchorY = laneId === 4 ? cy - 72 : cy - 24;
    } else {
      anchorX = cx - stop - boxW / 2 - 16 * scale;
      anchorY = laneId === 6 ? cy + 72 : cy + 24;
    }

    const boxX = Math.min(
      Math.max(bounds.minX + 8, anchorX - boxW / 2),
      bounds.maxX - boxW - 8
    );
    const boxY = Math.min(
      Math.max(bounds.minY + 8, anchorY - boxH / 2),
      bounds.maxY - boxH - 8
    );

    ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
    this._roundRect(ctx, boxX, boxY, boxW, boxH, 4 * scale);
    ctx.fill();

    ctx.strokeStyle = 'rgba(96, 165, 250, 0.9)';
    ctx.lineWidth = 1.25 * scale;
    this._roundRect(ctx, boxX, boxY, boxW, boxH, 4 * scale);
    ctx.stroke();

    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, boxX + boxW / 2, boxY + boxH / 2 + 0.5);
    ctx.restore();
  }

  _drawLaneArrows(ctx) {
    const cx = this.centerX;
    const cy = this.centerY;

    const arrowList = [
      { x: cx - 72, y: cy - 155, angle: Math.PI / 2, isTurn: false },
      { x: cx - 24, y: cy - 155, angle: Math.PI / 2, isTurn: true },
      { x: cx + 72, y: cy + 155, angle: -Math.PI / 2, isTurn: false },
      { x: cx + 24, y: cy + 155, angle: -Math.PI / 2, isTurn: true },
      { x: cx + 155, y: cy - 72, angle: Math.PI, isTurn: false },
      { x: cx + 155, y: cy - 24, angle: Math.PI, isTurn: true },
      { x: cx - 155, y: cy + 72, angle: 0, isTurn: false },
      { x: cx - 155, y: cy + 24, angle: 0, isTurn: true },
    ];

    ctx.strokeStyle = 'rgba(226, 232, 240, 0.45)';
    ctx.fillStyle = 'rgba(226, 232, 240, 0.45)';
    ctx.lineWidth = 2;

    for (const item of arrowList) {
      ctx.save();
      ctx.translate(item.x, item.y);
      ctx.rotate(item.angle);
      ctx.beginPath();
      if (!item.isTurn) {
        ctx.moveTo(-10, 0);
        ctx.lineTo(10, 0);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(12, 0);
        ctx.lineTo(5, -4.5);
        ctx.lineTo(5, 4.5);
        ctx.closePath();
        ctx.fill();
      } else {
        ctx.moveTo(-8, 4);
        ctx.lineTo(2, 4);
        ctx.lineTo(2, -6);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(2, -10);
        ctx.lineTo(-2.5, -3);
        ctx.lineTo(6.5, -3);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }
  }

  _getLaneSignalColor(laneId) {
    if (!this.latestSnapshot) {
      return 'RED';
    }
    const stage = this.latestSnapshot.signal_stage;
    const phaseLanes = this.latestSnapshot.phase_green_lanes || [];
    if (!phaseLanes.includes(laneId)) {
      return 'RED';
    }
    if (stage === 'GREEN') {
      return 'GREEN';
    }
    if (stage === 'YELLOW') {
      return 'YELLOW';
    }
    return 'RED';
  }

  _drawStopLinesAndSignalHeads(ctx) {
    const cx = this.centerX;
    const cy = this.centerY;
    const stop = this.stopOffset;

    const stopBars = [
      { laneId: 0, x1: cx - 94, y1: cy - stop, x2: cx - 50, y2: cy - stop },
      { laneId: 1, x1: cx - 46, y1: cy - stop, x2: cx - 4, y2: cy - stop },
      { laneId: 3, x1: cx + 4, y1: cy + stop, x2: cx + 46, y2: cy + stop },
      { laneId: 2, x1: cx + 50, y1: cy + stop, x2: cx + 94, y2: cy + stop },
      { laneId: 4, x1: cx + stop, y1: cy - 94, x2: cx + stop, y2: cy - 50 },
      { laneId: 5, x1: cx + stop, y1: cy - 46, x2: cx + stop, y2: cy - 4 },
      { laneId: 7, x1: cx - stop, y1: cy + 4, x2: cx - stop, y2: cy + 46 },
      { laneId: 6, x1: cx - stop, y1: cy + 50, x2: cx - stop, y2: cy + 94 },
    ];

    ctx.lineWidth = 4;
    for (const bar of stopBars) {
      const state = this._getLaneSignalColor(bar.laneId);
      ctx.strokeStyle =
        state === 'GREEN'
          ? '#22c55e'
          : state === 'YELLOW'
          ? '#f59e0b'
          : '#ef4444';
      ctx.beginPath();
      ctx.moveTo(bar.x1, bar.y1);
      ctx.lineTo(bar.x2, bar.y2);
      ctx.stroke();
    }

    // Real 3-lens housings positioned cleanly on the 4 intersection corners
    const headSpecs = [
      // Northbound approach signals (top-left corner for N Straight & N Turn)
      { x: cx - 156, y: cy - 178, vertical: true, laneId: 0, tag: 'N-STR' },
      { x: cx - 128, y: cy - 178, vertical: true, laneId: 1, tag: 'N-TRN' },
      // Southbound approach signals (bottom-right corner)
      { x: cx + 112, y: cy + 124, vertical: true, laneId: 3, tag: 'S-TRN' },
      { x: cx + 140, y: cy + 124, vertical: true, laneId: 2, tag: 'S-STR' },
      // Eastbound approach signals (top-right corner for E Straight & E Turn)
      { x: cx + 124, y: cy - 156, vertical: false, laneId: 4, tag: 'E-STR' },
      { x: cx + 124, y: cy - 128, vertical: false, laneId: 5, tag: 'E-TRN' },
      // Westbound approach signals (bottom-left corner)
      { x: cx - 178, y: cy + 112, vertical: false, laneId: 7, tag: 'W-TRN' },
      { x: cx - 178, y: cy + 140, vertical: false, laneId: 6, tag: 'W-STR' },
    ];

    for (const head of headSpecs) {
      this._drawThreeLensHousing(
        ctx,
        head.x,
        head.y,
        head.vertical,
        this._getLaneSignalColor(head.laneId),
        head.tag
      );
    }
  }

  _drawThreeLensHousing(ctx, x, y, vertical, activeState, labelText) {
    const w = vertical ? 20 : 50;
    const h = vertical ? 50 : 20;

    ctx.fillStyle = '#14171d';
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1.5;
    this._roundRect(ctx, x, y, w, h, 5);
    ctx.fill();
    ctx.stroke();

    const lenses = [
      { key: 'RED', on: '#ef4444', off: '#3f1d1d', glow: 'rgba(239, 68, 68, 0.42)' },
      { key: 'YELLOW', on: '#f59e0b', off: '#423012', glow: 'rgba(245, 158, 11, 0.42)' },
      { key: 'GREEN', on: '#22c55e', off: '#173824', glow: 'rgba(34, 197, 94, 0.42)' },
    ];

    lenses.forEach((lens, index) => {
      const lx = vertical ? x + 10 : x + 10 + index * 15;
      const ly = vertical ? y + 10 + index * 15 : y + 10;
      const isActive = activeState === lens.key;

      if (isActive) {
        ctx.fillStyle = lens.glow;
        ctx.beginPath();
        ctx.arc(lx, ly, 8.5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = isActive ? lens.on : lens.off;
      ctx.beginPath();
      ctx.arc(lx, ly, 5.2, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = '#475569';
    ctx.font = '600 9px "IBM Plex Mono", monospace';
    ctx.fillText(labelText, x - 2, y + h + 11);
  }

  _drawAllVehicles(ctx, timestampMs) {
    const blinkOn = Math.floor(timestampMs / 320) % 2 === 0;

    for (let laneIndex = 0; laneIndex < 8; laneIndex += 1) {
      const queue = this.queuedByLane[laneIndex];
      const spec = LANE_SPECIFICATIONS[laneIndex];

      for (const vehicle of queue) {
        const pose = this._queuedVehiclePose(laneIndex, vehicle.distFromStop);
        if (!this._isInsideCanvas(pose.x, pose.y)) {
          continue;
        }
        this._drawCarSprite(
          ctx,
          pose.x,
          pose.y,
          spec.startHeading,
          vehicle.color,
          vehicle.braking,
          vehicle.isTurn && blinkOn
        );
      }
    }

    for (const vehicle of this.exitingVehicles) {
      if (vehicle.delayRemaining > 0) {
        continue;
      }
      const pose = this._samplePathPose(vehicle.path, vehicle.progressPx);
      if (!this._isInsideCanvas(pose.x, pose.y)) {
        continue;
      }
      if (vehicle.isStraggler) {
        ctx.save();
        ctx.strokeStyle = blinkOn ? '#f59e0b' : '#d97706';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(pose.x, pose.y, 14, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
      this._drawCarSprite(
        ctx,
        pose.x,
        pose.y,
        pose.heading,
        vehicle.color,
        Boolean(vehicle.isStraggler),
        (vehicle.isTurn || vehicle.isStraggler) && blinkOn
      );
    }

    for (const vehicle of this.collidingVehicles) {
      const pose = this._samplePathPose(vehicle.path, vehicle.progressPx);
      if (!this._isInsideCanvas(pose.x, pose.y)) {
        continue;
      }
      if (vehicle.role === 'culprit' || vehicle.state === 'halted') {
        ctx.save();
        ctx.strokeStyle =
          vehicle.role === 'culprit'
            ? blinkOn
              ? '#ef4444'
              : '#dc2626'
            : blinkOn
            ? '#f59e0b'
            : '#ef4444';
        ctx.lineWidth = 1.75;
        ctx.beginPath();
        ctx.arc(pose.x, pose.y, 15, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      this._drawCarSprite(
        ctx,
        pose.x,
        pose.y,
        pose.heading + (vehicle.yawOffset || 0),
        vehicle.color,
        Boolean(vehicle.braking || vehicle.state === 'halted'),
        blinkOn
      );
    }
  }

  _drawCollisionEffects(ctx) {
    if (!this.collisionEffects || this.collisionEffects.length === 0) {
      return;
    }

    let latestEffect = null;

    for (const effect of this.collisionEffects) {
      if (effect.delayRemaining && effect.delayRemaining > 0) {
        continue;
      }
      const total = Math.max(effect.totalSec || 2.2, 0.1);
      const remaining = Math.max(0, effect.remainingSec);
      const alpha = Math.min(1, remaining / total);
      const elapsed = Math.max(0, total - remaining);
      const pulsePhase = (elapsed * 1.8) % 1;

      ctx.save();

      // Translucent hazard core
      ctx.globalAlpha = alpha * 0.24;
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, 12, 0, Math.PI * 2);
      ctx.fill();

      // Inner fixed hazard ring
      ctx.globalAlpha = alpha * 0.95;
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, 12, 0, Math.PI * 2);
      ctx.stroke();

      // Middle expanding ring
      ctx.globalAlpha = alpha * (1 - pulsePhase * 0.65);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2.25;
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, 15 + pulsePhase * 22, 0, Math.PI * 2);
      ctx.stroke();

      // Outer faint ring
      const outerPhase = (pulsePhase + 0.5) % 1;
      ctx.globalAlpha = alpha * (1 - outerPhase * 0.75) * 0.75;
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, 16 + outerPhase * 28, 0, Math.PI * 2);
      ctx.stroke();

      // Center impact point marker
      ctx.globalAlpha = alpha;
      ctx.fillStyle = '#dc2626';
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
      latestEffect = effect;
    }

    if (latestEffect) {
      const total = Math.max(latestEffect.totalSec || 2.2, 0.1);
      const alpha = Math.min(1, Math.max(0, latestEffect.remainingSec) / total);
      const pairTag =
        latestEffect.culpritLabel && latestEffect.victimLabel
          ? ` (${latestEffect.culpritLabel} x ${latestEffect.victimLabel})`
          : '';
      const bannerText = `COLLISION: ${latestEffect.kind || 'red_light_conflict'}${pairTag}`;

      ctx.save();
      ctx.globalAlpha = Math.min(1, alpha * 1.15);
      ctx.font = '600 10.5px "IBM Plex Mono", monospace';
      const textWidth = ctx.measureText(bannerText).width;
      const boxW = Math.max(180, textWidth + 18);
      const boxH = 19;
      const boxX = this.centerX - boxW / 2;
      const boxY = this.centerY - this.roadHalfWidth;

      ctx.fillStyle = 'rgba(20, 23, 29, 0.92)';
      this._roundRect(ctx, boxX, boxY, boxW, boxH, 4);
      ctx.fill();

      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      this._roundRect(ctx, boxX, boxY, boxW, boxH, 4);
      ctx.stroke();

      ctx.fillStyle = '#fef2f2';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(bannerText, this.centerX, boxY + boxH / 2 + 0.5);
      ctx.restore();
    }
  }

  _queuedVehiclePose(laneIndex, distFromStop) {
    const cx = this.centerX;
    const cy = this.centerY;
    const offset = this.stopOffset + distFromStop;

    switch (laneIndex) {
      case 0:
        return { x: cx - 72, y: cy - offset };
      case 1:
        return { x: cx - 24, y: cy - offset };
      case 2:
        return { x: cx + 72, y: cy + offset };
      case 3:
        return { x: cx + 24, y: cy + offset };
      case 4:
        return { x: cx + offset, y: cy - 72 };
      case 5:
        return { x: cx + offset, y: cy - 24 };
      case 6:
        return { x: cx - offset, y: cy + 72 };
      default:
        return { x: cx - offset, y: cy + 24 };
    }
  }

  _samplePathPose(path, progressPx) {
    if (path.kind === 'line') {
      const totalLength = Math.max(this._estimatePathLength(path), 1);
      const t = Math.min(Math.max(progressPx / totalLength, 0), 1);
      const dx = path.x1 - path.x0;
      const dy = path.y1 - path.y0;
      return {
        x: path.x0 + dx * t,
        y: path.y0 + dy * t,
        heading: Math.atan2(dy, dx),
      };
    }

    if (path.kind === 'turn_arc') {
      if (path.entry.len > 0 && progressPx <= path.entry.len) {
        const t = Math.min(Math.max(progressPx / path.entry.len, 0), 1);
        const dx = path.entry.x1 - path.entry.x0;
        const dy = path.entry.y1 - path.entry.y0;
        return {
          x: path.entry.x0 + dx * t,
          y: path.entry.y0 + dy * t,
          heading: Math.atan2(dy, dx),
        };
      }
      const afterEntry = Math.max(0, progressPx - path.entry.len);
      if (afterEntry <= path.arc.len) {
        const t = Math.min(Math.max(afterEntry / path.arc.len, 0), 1);
        const angleDelta = path.arc.a1 - path.arc.a0;
        const angle = path.arc.a0 + angleDelta * t;
        const sign = angleDelta >= 0 ? 1 : -1;
        const dx = -sign * Math.sin(angle);
        const dy = sign * Math.cos(angle);
        return {
          x: path.arc.cx + path.arc.radius * Math.cos(angle),
          y: path.arc.cy + path.arc.radius * Math.sin(angle),
          heading: Math.atan2(dy, dx),
        };
      }
      const afterArc = Math.max(0, afterEntry - path.arc.len);
      const t = Math.min(Math.max(afterArc / path.exit.len, 0), 1);
      const dx = path.exit.x1 - path.exit.x0;
      const dy = path.exit.y1 - path.exit.y0;
      return {
        x: path.exit.x0 + dx * t,
        y: path.exit.y0 + dy * t,
        heading: Math.atan2(dy, dx),
      };
    }

    if (path.kind === 'turn_compound') {
      if (path.entry.len > 0 && progressPx <= path.entry.len) {
        const t = Math.min(Math.max(progressPx / path.entry.len, 0), 1);
        const dx = path.entry.x1 - path.entry.x0;
        const dy = path.entry.y1 - path.entry.y0;
        return {
          x: path.entry.x0 + dx * t,
          y: path.entry.y0 + dy * t,
          heading: Math.atan2(dy, dx),
        };
      }
      const afterEntry = Math.max(0, progressPx - path.entry.len);
      if (afterEntry <= path.bezier.len) {
        const t = Math.min(Math.max(afterEntry / path.bezier.len, 0), 1);
        return this._sampleQuadraticBezier(path.bezier, t);
      }
      const afterBezier = Math.max(0, afterEntry - path.bezier.len);
      const t = Math.min(Math.max(afterBezier / path.exit.len, 0), 1);
      const dx = path.exit.x1 - path.exit.x0;
      const dy = path.exit.y1 - path.exit.y0;
      return {
        x: path.exit.x0 + dx * t,
        y: path.exit.y0 + dy * t,
        heading: Math.atan2(dy, dx),
      };
    }

    const totalLength = Math.max(this._estimatePathLength(path), 1);
    const t = Math.min(Math.max(progressPx / totalLength, 0), 1);
    return this._sampleQuadraticBezier(path, t);
  }

  _sampleQuadraticBezier(bezier, t) {
    const inv = 1 - t;
    const x =
      inv * inv * bezier.p0.x + 2 * inv * t * bezier.p1.x + t * t * bezier.p2.x;
    const y =
      inv * inv * bezier.p0.y + 2 * inv * t * bezier.p1.y + t * t * bezier.p2.y;
    const dx =
      2 * inv * (bezier.p1.x - bezier.p0.x) + 2 * t * (bezier.p2.x - bezier.p1.x);
    const dy =
      2 * inv * (bezier.p1.y - bezier.p0.y) + 2 * t * (bezier.p2.y - bezier.p1.y);

    return { x, y, heading: Math.atan2(dy, dx) };
  }

  _drawCarSprite(ctx, x, y, heading, bodyColor, braking, turnSignalOn) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(heading);

    // Subtle ground contact shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.28)';
    this._roundRect(ctx, -10, -6, 20, 14, 4);
    ctx.fill();

    // Car body centered at (0,0), nose pointing +x
    ctx.fillStyle = bodyColor;
    this._roundRect(ctx, -10, -6.5, 20, 13, 3.5);
    ctx.fill();

    // Cabin / windshield roof
    ctx.fillStyle = '#0f172a';
    this._roundRect(ctx, -4.5, -4.5, 10, 9, 2);
    ctx.fill();

    // Headlights (front +x)
    ctx.fillStyle = '#fef9c3';
    ctx.fillRect(8.2, -4.8, 2, 2.2);
    ctx.fillRect(8.2, 2.6, 2, 2.2);

    // Red brake lights (rear -x)
    ctx.fillStyle = braking ? '#ff2d2d' : '#7f1d1d';
    ctx.fillRect(-10.2, -5.0, 2, 2.5);
    ctx.fillRect(-10.2, 2.5, 2, 2.5);

    // Left-turn amber blinker on front-left & rear-left
    if (turnSignalOn) {
      ctx.fillStyle = '#fbbf24';
      ctx.fillRect(7.5, -6.5, 2.5, 1.8);
      ctx.fillRect(-9.5, -6.5, 2.5, 1.8);
    }

    ctx.restore();
  }

  _drawQueueBadges(ctx, bounds) {
    if (!this.latestSnapshot) {
      return;
    }
    const queues = this.latestSnapshot.lane_queues || [];
    const cx = this.centerX;
    const cy = this.centerY;

    // Cap inverse badge scale at 44/28 so adjacent 48px-wide lanes never overlap
    const badgeScale = Math.min(44 / 28, Math.max(1.0, 1 / this.zoomLevel));
    const badgeW = 28 * badgeScale;
    const badgeH = 20 * badgeScale;
    const insetY = 18 * badgeScale;
    const insetX = 22 * badgeScale;
    const fontPx = Math.round(11 * badgeScale * 10) / 10;

    const badgePositions = [
      { x: cx - 72, y: bounds.minY + insetY },
      { x: cx - 24, y: bounds.minY + insetY },
      { x: cx + 72, y: bounds.maxY - insetY },
      { x: cx + 24, y: bounds.maxY - insetY },
      { x: bounds.maxX - insetX, y: cy - 72 },
      { x: bounds.maxX - insetX, y: cy - 24 },
      { x: bounds.minX + insetX, y: cy + 72 },
      { x: bounds.minX + insetX, y: cy + 24 },
    ];

    ctx.font = `600 ${fontPx}px "IBM Plex Mono", monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    badgePositions.forEach((pos, idx) => {
      const count = Number(queues[idx] || 0);
      ctx.fillStyle = count >= 12 ? '#dc2626' : '#0f172a';
      this._roundRect(
        ctx,
        pos.x - badgeW / 2,
        pos.y - badgeH / 2,
        badgeW,
        badgeH,
        4 * badgeScale
      );
      ctx.fill();

      ctx.fillStyle = '#f8fafc';
      ctx.fillText(String(count), pos.x, pos.y + 0.5);
    });

    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  _isInsideCanvas(x, y) {
    const bounds = this._getWorldBounds();
    return (
      x > bounds.minX - 35 &&
      x < bounds.maxX + 35 &&
      y > bounds.minY - 35 &&
      y < bounds.maxY + 35
    );
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  }
}
