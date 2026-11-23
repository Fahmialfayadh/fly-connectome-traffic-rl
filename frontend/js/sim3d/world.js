/**
 * world.js — Realistic daylight 3D environment for the 4-way intersection.
 * Uses right-hand traffic (RHT) geometry matching the Python TrafficIntersectionEnv
 * CONFLICTING_LANES, Pygame 2D panel, and 2D Web Canvas renderer.
 */
import * as THREE from 'three';

// Dimensions matching environment proportions
export const ROAD_WIDTH = 12.0;
export const ROAD_HALF = ROAD_WIDTH / 2.0; // 6.0m (intersection box is [-6, +6] x [-6, +6])
export const LANE_WIDTH = 3.0;             // 4 lanes per road corridor (2 incoming, 2 outgoing)
export const ROAD_LENGTH = 100.0;          // Long corridors for full queues & off-screen entry
export const INTERSECTION_SIZE = ROAD_WIDTH;
export const SIDEWALK_WIDTH = 3.2;
export const SIDEWALK_HEIGHT = 0.22;
export const CROSSWALK_CENTER_OFFSET = ROAD_HALF + 1.4; // 7.4m (spans 6.4m .. 8.4m)
export const STOP_LINE_OFFSET = 9.0;                    // Stop bar cleanly behind crosswalk

// Clean daylight palette (bright architecture, clean asphalt)
const COLOR_GROUND = 0x5a9a4b;
const COLOR_ASPHALT = 0x2e323b;
const COLOR_INTERSECT = 0x272a32;
const COLOR_SIDEWALK = 0xd8dbe0;
const COLOR_MARKING_WHITE = 0xffffff;
const COLOR_MARKING_YELLOW = 0xf5b041;

const BUILDING_COLORS = [
  0xf8f9fa,
  0xf1f3f5,
  0xede8df,
  0xe9ecef,
  0xe5e0d8,
  0xdde2e6,
];

export class IntersectionWorld {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = 'world';

    this.laneHitMeshes = [];
    this._buildEnvironment();
    this._buildLighting();

    scene.add(this.group);
  }

  _buildEnvironment() {
    this._createGround();
    this._createRoadCorridors();
    this._createIntersectionBox();
    this._createCompassRose();
    this._createRoadMarkings();
    this._createLaneArrows();
    this._createSidewalksAndCurbs();
    this._createBrightBuildings();
    this._createTreesAndFoliage();
    this._createLaneHitSurfaces();
  }

  _buildLighting() {
    const ambientLight = new THREE.AmbientLight(0xdbeafe, 0.65);
    this.scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x4ade80, 0.45);
    hemiLight.position.set(0, 50, 0);
    this.scene.add(hemiLight);

    const sunLight = new THREE.DirectionalLight(0xfffbf0, 1.35);
    sunLight.position.set(45, 65, 30);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.left = -75;
    sunLight.shadow.camera.right = 75;
    sunLight.shadow.camera.top = 75;
    sunLight.shadow.camera.bottom = -75;
    sunLight.shadow.camera.near = 5;
    sunLight.shadow.camera.far = 190;
    sunLight.shadow.bias = -0.0008;
    this.scene.add(sunLight);
  }

  _createGround() {
    const groundGeo = new THREE.PlaneGeometry(360, 360);
    const groundMat = new THREE.MeshStandardMaterial({
      color: COLOR_GROUND,
      roughness: 0.9,
      metalness: 0.05,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.02;
    ground.receiveShadow = true;
    this.group.add(ground);
  }

  _createRoadCorridors() {
    const asphaltMat = new THREE.MeshStandardMaterial({
      color: COLOR_ASPHALT,
      roughness: 0.82,
      metalness: 0.1,
    });

    const totalLen = ROAD_LENGTH * 2 + INTERSECTION_SIZE;

    const nsRoad = new THREE.Mesh(new THREE.PlaneGeometry(ROAD_WIDTH, totalLen), asphaltMat);
    nsRoad.rotation.x = -Math.PI / 2;
    nsRoad.position.y = 0.005;
    nsRoad.receiveShadow = true;
    this.group.add(nsRoad);

    const ewRoad = new THREE.Mesh(new THREE.PlaneGeometry(totalLen, ROAD_WIDTH), asphaltMat);
    ewRoad.rotation.x = -Math.PI / 2;
    ewRoad.position.y = 0.005;
    ewRoad.receiveShadow = true;
    this.group.add(ewRoad);
  }

  _createIntersectionBox() {
    const boxMat = new THREE.MeshStandardMaterial({
      color: COLOR_INTERSECT,
      roughness: 0.8,
      metalness: 0.1,
    });
    const box = new THREE.Mesh(
      new THREE.PlaneGeometry(INTERSECTION_SIZE, INTERSECTION_SIZE),
      boxMat
    );
    box.rotation.x = -Math.PI / 2;
    box.position.y = 0.008;
    box.receiveShadow = true;
    this.group.add(box);
  }

  _createCompassRose() {
    const compassSize = 8.0;

    if (typeof document === 'undefined') {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(8.0, 8.25, 32),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.011;
      ring.name = 'compass_rose_fallback';
      this.group.add(ring);
      return;
    }

    const size = 1000;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const c = size / 2;

    ctx.clearRect(0, 0, size, size);

    // Outer & inner black-and-white compass rings
    ctx.lineWidth = 5;
    ctx.strokeStyle = 'rgba(10, 10, 12, 0.9)';
    ctx.beginPath();
    ctx.arc(c, c, 316, 0, Math.PI * 2);
    ctx.stroke();

    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.92)';
    ctx.beginPath();
    ctx.arc(c, c, 310, 0, Math.PI * 2);
    ctx.stroke();

    ctx.lineWidth = 2.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.beginPath();
    ctx.arc(c, c, 232, 0, Math.PI * 2);
    ctx.stroke();

    const drawSplitArm = (angleRad, outerR, sideR, lightColor, darkColor) => {
      const tipX = c + Math.sin(angleRad) * outerR;
      const tipY = c - Math.cos(angleRad) * outerR;
      const leftX = c + Math.sin(angleRad - Math.PI / 4) * sideR;
      const leftY = c - Math.cos(angleRad - Math.PI / 4) * sideR;
      const rightX = c + Math.sin(angleRad + Math.PI / 4) * sideR;
      const rightY = c - Math.cos(angleRad + Math.PI / 4) * sideR;

      // Left half
      ctx.beginPath();
      ctx.moveTo(c, c);
      ctx.lineTo(tipX, tipY);
      ctx.lineTo(leftX, leftY);
      ctx.closePath();
      ctx.fillStyle = lightColor;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      // Right half
      ctx.beginPath();
      ctx.moveTo(c, c);
      ctx.lineTo(tipX, tipY);
      ctx.lineTo(rightX, rightY);
      ctx.closePath();
      ctx.fillStyle = darkColor;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    };

    // Minor diagonal points (NE, SE, SW, NW)
    for (let i = 0; i < 4; i++) {
      const ang = (Math.PI / 4) + i * (Math.PI / 2);
      drawSplitArm(ang, 205, 42, '#e2e8f0', '#0b0b0d');
    }

    // Major cardinal points (N, E, S, W)
    for (let i = 0; i < 4; i++) {
      const ang = i * (Math.PI / 2);
      drawSplitArm(ang, 292, 60, '#ffffff', '#09090b');
    }

    // Center hub
    ctx.beginPath();
    ctx.arc(c, c, 24, 0, Math.PI * 2);
    ctx.fillStyle = '#09090b';
    ctx.fill();
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    // Cardinal direction badges (N inverted white-on-black contrast, E/S/W crisp white-on-black)
    const cardinals = [
      { label: 'N', x: c, y: c - 395, isNorth: true },
      { label: 'E', x: c + 395, y: c, isNorth: false },
      { label: 'S', x: c, y: c + 395, isNorth: false },
      { label: 'W', x: c - 395, y: c, isNorth: false },
    ];

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const item of cardinals) {
      const r = item.isNorth ? 56 : 48;
      ctx.beginPath();
      ctx.arc(item.x, item.y, r, 0, Math.PI * 2);
      ctx.fillStyle = item.isNorth ? '#ffffff' : '#09090b';
      ctx.fill();
      ctx.lineWidth = item.isNorth ? 5 : 4;
      ctx.strokeStyle = item.isNorth ? '#09090b' : '#ffffff';
      ctx.stroke();

      ctx.font = `800 ${item.isNorth ? 70 : 58}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
      ctx.fillStyle = item.isNorth ? '#09090b' : '#ffffff';
      ctx.fillText(item.label, item.x, item.y + 3);
    }

    // Intercardinal labels (NE, SE, SW, NW)
    const inters = [
      { label: 'NE', ang: Math.PI / 4 },
      { label: 'SE', ang: (3 * Math.PI) / 4 },
      { label: 'SW', ang: (5 * Math.PI) / 4 },
      { label: 'NW', ang: (7 * Math.PI) / 4 },
    ];
    ctx.font = '700 38px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    for (const item of inters) {
      const lx = c + Math.sin(item.ang) * 282;
      const ly = c - Math.cos(item.ang) * 282;
      ctx.lineWidth = 7;
      ctx.strokeStyle = '#09090b';
      ctx.strokeText(item.label, lx, ly + 2);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(item.label, lx, ly + 2);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 8;

    const compassMat = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0.86,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const compassMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(compassSize, compassSize),
      compassMat
    );
    compassMesh.rotation.x = -Math.PI / 2;
    compassMesh.position.y = 0.011;
    compassMesh.name = 'compass_rose';
    this.group.add(compassMesh);
  }

  _createRoadMarkings() {
    const whiteMat = new THREE.MeshBasicMaterial({ color: COLOR_MARKING_WHITE });
    const yellowMat = new THREE.MeshBasicMaterial({ color: COLOR_MARKING_YELLOW });

    // Double yellow center medians starting cleanly behind the stop line
    for (const [isNS, sign] of [[true, -1], [true, 1], [false, -1], [false, 1]]) {
      const startD = STOP_LINE_OFFSET + 0.25;
      const endD = ROAD_LENGTH + ROAD_HALF;
      const len = endD - startD;
      const centerPos = startD + len / 2;

      for (const offset of [-0.09, 0.09]) {
        const geo = isNS
          ? new THREE.PlaneGeometry(0.11, len)
          : new THREE.PlaneGeometry(len, 0.11);
        const line = new THREE.Mesh(geo, yellowMat);
        line.rotation.x = -Math.PI / 2;
        if (isNS) {
          line.position.set(offset, 0.015, sign * centerPos);
        } else {
          line.position.set(sign * centerPos, 0.015, offset);
        }
        this.group.add(line);
      }
    }

    // Dashed white lane dividers between straight and turn lanes (both incoming & outgoing)
    const dashLen = 2.2;
    const dashGap = 2.2;
    for (const [isNS, sign] of [[true, -1], [true, 1], [false, -1], [false, 1]]) {
      const geo = isNS
        ? new THREE.PlaneGeometry(0.11, dashLen)
        : new THREE.PlaneGeometry(dashLen, 0.11);
      for (let d = STOP_LINE_OFFSET + 0.8; d < ROAD_LENGTH + ROAD_HALF; d += dashLen + dashGap) {
        for (const laneSepOffset of [-LANE_WIDTH, LANE_WIDTH]) {
          const dash = new THREE.Mesh(geo, whiteMat);
          dash.rotation.x = -Math.PI / 2;
          if (isNS) {
            dash.position.set(laneSepOffset, 0.014, sign * (d + dashLen / 2));
          } else {
            dash.position.set(sign * (d + dashLen / 2), 0.014, laneSepOffset);
          }
          this.group.add(dash);
        }
      }
    }

    // RHT Solid white stop lines across incoming half of each arm (at STOP_LINE_OFFSET = 9.0)
    const stopBarLen = ROAD_HALF - 0.15;
    const stopBarThick = 0.4;
    // North incoming (x in [-6, 0], z = -9.0)
    this._addStopBar(-LANE_WIDTH, -STOP_LINE_OFFSET, stopBarLen, stopBarThick, whiteMat);
    // South incoming (x in [0, +6], z = +9.0)
    this._addStopBar(LANE_WIDTH, STOP_LINE_OFFSET, stopBarLen, stopBarThick, whiteMat);
    // East incoming (x = +9.0, z in [-6, 0])
    this._addStopBar(STOP_LINE_OFFSET, -LANE_WIDTH, stopBarThick, stopBarLen, whiteMat);
    // West incoming (x = -9.0, z in [0, +6])
    this._addStopBar(-STOP_LINE_OFFSET, LANE_WIDTH, stopBarThick, stopBarLen, whiteMat);

    // Zebra Crosswalks spanning [6.4, 8.4] cleanly in front of stop lines
    this._createZebraCrosswalks(whiteMat);
  }

  _addStopBar(x, z, width, length, mat) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, length), mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(x, 0.017, z);
    this.group.add(mesh);
  }

  _createZebraCrosswalks(mat) {
    const stripeW = 0.55;
    const stripeL = 2.0;
    const gap = 0.55;

    for (const zSign of [-1, 1]) {
      const crosswalkZ = zSign * CROSSWALK_CENTER_OFFSET;
      for (let x = -(ROAD_HALF - 0.65); x <= ROAD_HALF - 0.65; x += stripeW + gap) {
        const stripe = new THREE.Mesh(new THREE.PlaneGeometry(stripeW, stripeL), mat);
        stripe.rotation.x = -Math.PI / 2;
        stripe.position.set(x, 0.013, crosswalkZ);
        this.group.add(stripe);
      }
    }

    for (const xSign of [-1, 1]) {
      const crosswalkX = xSign * CROSSWALK_CENTER_OFFSET;
      for (let z = -(ROAD_HALF - 0.65); z <= ROAD_HALF - 0.65; z += stripeW + gap) {
        const stripe = new THREE.Mesh(new THREE.PlaneGeometry(stripeL, stripeW), mat);
        stripe.rotation.x = -Math.PI / 2;
        stripe.position.set(crosswalkX, 0.013, z);
        this.group.add(stripe);
      }
    }
  }

  _createLaneArrows() {
    // Painted pavement arrows showing straight vs left-turn direction on each RHT incoming lane
    const arrowMat = new THREE.MeshBasicMaterial({
      color: 0xe2e8f0,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
    });

    const straightShape = new THREE.Shape();
    straightShape.moveTo(-0.14, -0.95);
    straightShape.lineTo(0.14, -0.95);
    straightShape.lineTo(0.14, 0.35);
    straightShape.lineTo(0.4, 0.35);
    straightShape.lineTo(0, 1.0);
    straightShape.lineTo(-0.4, 0.35);
    straightShape.lineTo(-0.14, 0.35);
    straightShape.closePath();
    const straightGeo = new THREE.ShapeGeometry(straightShape);

    // With mesh.rotation.x = -PI/2, shape +Y maps to local -Z (forward) and -X maps to driver's LEFT.
    const turnShape = new THREE.Shape();
    turnShape.moveTo(0.14, -0.95);
    turnShape.lineTo(-0.14, -0.95);
    turnShape.lineTo(-0.14, 0.05);
    turnShape.lineTo(-0.45, 0.05);
    turnShape.lineTo(-0.45, -0.2);
    turnShape.lineTo(-0.95, 0.18);
    turnShape.lineTo(-0.45, 0.56);
    turnShape.lineTo(-0.45, 0.31);
    turnShape.lineTo(0.14, 0.31);
    turnShape.closePath();
    const turnGeo = new THREE.ShapeGeometry(turnShape);

    // holder.rotation.y rotates local -Z (forward) to each lane's travel direction
    const arrowSpecs = [
      { x: -4.5, z: -13.5, rotY: Math.PI, isTurn: false },      // 0: N_STR (+z)
      { x: -1.5, z: -13.5, rotY: Math.PI, isTurn: true },       // 1: N_TRN (+z -> +x)
      { x: 4.5, z: 13.5, rotY: 0, isTurn: false },              // 2: S_STR (-z)
      { x: 1.5, z: 13.5, rotY: 0, isTurn: true },               // 3: S_TRN (-z -> -x)
      { x: 13.5, z: -4.5, rotY: Math.PI / 2, isTurn: false },   // 4: E_STR (-x)
      { x: 13.5, z: -1.5, rotY: Math.PI / 2, isTurn: true },    // 5: E_TRN (-x -> +z)
      { x: -13.5, z: 4.5, rotY: -Math.PI / 2, isTurn: false },  // 6: W_STR (+x)
      { x: -13.5, z: 1.5, rotY: -Math.PI / 2, isTurn: true },   // 7: W_TRN (+x -> -z)
    ];

    for (const spec of arrowSpecs) {
      const holder = new THREE.Group();
      const mesh = new THREE.Mesh(spec.isTurn ? turnGeo : straightGeo, arrowMat);
      mesh.rotation.x = -Math.PI / 2; // normal points +Y (upward)
      holder.add(mesh);
      holder.position.set(spec.x, 0.015, spec.z);
      holder.rotation.y = spec.rotY;
      this.group.add(holder);
    }
  }

  _createSidewalksAndCurbs() {
    const sidewalkMat = new THREE.MeshStandardMaterial({
      color: COLOR_SIDEWALK,
      roughness: 0.88,
      metalness: 0.05,
    });

    for (const xSign of [-1, 1]) {
      for (const zSign of [-1, 1]) {
        const cx = xSign * (ROAD_HALF + SIDEWALK_WIDTH / 2);
        const cz = zSign * (ROAD_HALF + SIDEWALK_WIDTH / 2);

        const corner = new THREE.Mesh(
          new THREE.BoxGeometry(SIDEWALK_WIDTH, SIDEWALK_HEIGHT, SIDEWALK_WIDTH),
          sidewalkMat
        );
        corner.position.set(cx, SIDEWALK_HEIGHT / 2, cz);
        corner.receiveShadow = true;
        this.group.add(corner);

        const armNS = new THREE.Mesh(
          new THREE.BoxGeometry(SIDEWALK_WIDTH, SIDEWALK_HEIGHT, ROAD_LENGTH),
          sidewalkMat
        );
        armNS.position.set(
          cx,
          SIDEWALK_HEIGHT / 2,
          zSign * (ROAD_HALF + SIDEWALK_WIDTH + ROAD_LENGTH / 2)
        );
        armNS.receiveShadow = true;
        this.group.add(armNS);

        const armEW = new THREE.Mesh(
          new THREE.BoxGeometry(ROAD_LENGTH, SIDEWALK_HEIGHT, SIDEWALK_WIDTH),
          sidewalkMat
        );
        armEW.position.set(
          xSign * (ROAD_HALF + SIDEWALK_WIDTH + ROAD_LENGTH / 2),
          SIDEWALK_HEIGHT / 2,
          cz
        );
        armEW.receiveShadow = true;
        this.group.add(armEW);
      }
    }
  }

  _createBrightBuildings() {
    const glassMat = new THREE.MeshStandardMaterial({
      color: 0x88b7d5,
      roughness: 0.2,
      metalness: 0.85,
      transparent: true,
      opacity: 0.85,
    });

    const rng = this._seededRng(77);

    for (const xSign of [-1, 1]) {
      for (const zSign of [-1, 1]) {
        const blockBaseX = xSign * (ROAD_HALF + SIDEWALK_WIDTH + 3.0);
        const blockBaseZ = zSign * (ROAD_HALF + SIDEWALK_WIDTH + 3.0);

        for (let row = 0; row < 2; row++) {
          for (let col = 0; col < 2; col++) {
            const w = 9.0 + rng() * 6.0;
            const d = 9.0 + rng() * 6.0;
            const h = 12.0 + rng() * 22.0;

            const color = BUILDING_COLORS[Math.floor(rng() * BUILDING_COLORS.length)];
            const facadeMat = new THREE.MeshStandardMaterial({
              color,
              roughness: 0.75,
              metalness: 0.1,
            });

            const bldMesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), facadeMat);
            const posX = blockBaseX + xSign * (col * 14.0 + w / 2);
            const posZ = blockBaseZ + zSign * (row * 14.0 + d / 2);
            bldMesh.position.set(posX, h / 2, posZ);
            bldMesh.castShadow = true;
            bldMesh.receiveShadow = true;
            this.group.add(bldMesh);

            const numFloors = Math.floor(h / 3.2);
            for (let f = 1; f < numFloors; f++) {
              const floorY = f * 3.2;

              const winX = new THREE.Mesh(
                new THREE.PlaneGeometry(w * 0.85, 1.4),
                glassMat
              );
              winX.position.set(posX, floorY, posZ - zSign * (d / 2 + 0.02));
              if (zSign > 0) winX.rotation.y = Math.PI;
              this.group.add(winX);

              const winZ = new THREE.Mesh(
                new THREE.PlaneGeometry(d * 0.85, 1.4),
                glassMat
              );
              winZ.position.set(posX - xSign * (w / 2 + 0.02), floorY, posZ);
              winZ.rotation.y = xSign > 0 ? -Math.PI / 2 : Math.PI / 2;
              this.group.add(winZ);
            }

            const roofBox = new THREE.Mesh(
              new THREE.BoxGeometry(w * 0.4, 2.5, d * 0.4),
              facadeMat
            );
            roofBox.position.set(posX, h + 1.25, posZ);
            roofBox.castShadow = true;
            this.group.add(roofBox);
          }
        }
      }
    }
  }

  _createTreesAndFoliage() {
    const trunkMat = new THREE.MeshStandardMaterial({
      color: 0x5c4033,
      roughness: 0.85,
    });

    const foliageColors = [0x38a169, 0x48bb78, 0x2f855a, 0x529b2b];
    const rng = this._seededRng(199);

    for (const xSign of [-1, 1]) {
      for (const zSign of [-1, 1]) {
        for (let offset = 15; offset < ROAD_LENGTH - 4; offset += 9.5) {
          const treeX1 = xSign * (ROAD_HALF + SIDEWALK_WIDTH / 2);
          const treeZ1 = zSign * offset;
          this._addSingleTree(treeX1, treeZ1, trunkMat, foliageColors, rng);

          const treeX2 = xSign * offset;
          const treeZ2 = zSign * (ROAD_HALF + SIDEWALK_WIDTH / 2);
          this._addSingleTree(treeX2, treeZ2, trunkMat, foliageColors, rng);
        }
      }
    }
  }

  _addSingleTree(x, z, trunkMat, foliageColors, rng) {
    const trunkH = 2.4 + rng() * 1.0;
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.25, trunkH, 6),
      trunkMat
    );
    trunk.position.set(x, trunkH / 2, z);
    trunk.castShadow = true;
    this.group.add(trunk);

    const leafColor = foliageColors[Math.floor(rng() * foliageColors.length)];
    const leafMat = new THREE.MeshStandardMaterial({
      color: leafColor,
      roughness: 0.8,
      metalness: 0.05,
    });
    const crownR = 1.2 + rng() * 0.5;
    const crown = new THREE.Mesh(new THREE.SphereGeometry(crownR, 7, 6), leafMat);
    crown.position.set(x, trunkH + crownR * 0.7, z);
    crown.castShadow = true;
    this.group.add(crown);
  }

  _createLaneHitSurfaces() {
    const baseHitMat = new THREE.MeshBasicMaterial({
      color: 0x3b82f6,
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
    });

    // RHT incoming lane hitboxes matching LaneId 0..7
    const hitMid = STOP_LINE_OFFSET + ROAD_LENGTH / 2;
    const laneHitSpecs = [
      { id: 0, x: -4.5, z: -hitMid, w: LANE_WIDTH, l: ROAD_LENGTH }, // N_STRAIGHT
      { id: 1, x: -1.5, z: -hitMid, w: LANE_WIDTH, l: ROAD_LENGTH }, // N_TURN
      { id: 2, x: 4.5, z: hitMid, w: LANE_WIDTH, l: ROAD_LENGTH },   // S_STRAIGHT
      { id: 3, x: 1.5, z: hitMid, w: LANE_WIDTH, l: ROAD_LENGTH },   // S_TURN
      { id: 4, x: hitMid, z: -4.5, w: ROAD_LENGTH, l: LANE_WIDTH },  // E_STRAIGHT
      { id: 5, x: hitMid, z: -1.5, w: ROAD_LENGTH, l: LANE_WIDTH },  // E_TURN
      { id: 6, x: -hitMid, z: 4.5, w: ROAD_LENGTH, l: LANE_WIDTH },  // W_STRAIGHT
      { id: 7, x: -hitMid, z: 1.5, w: ROAD_LENGTH, l: LANE_WIDTH },  // W_TURN
    ];

    for (const spec of laneHitSpecs) {
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(spec.w, spec.l),
        baseHitMat.clone()
      );
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(spec.x, 0.02, spec.z);
      mesh.userData = { laneId: spec.id, isLaneHitbox: true };
      this.group.add(mesh);
      this.laneHitMeshes.push(mesh);
    }
  }

  highlightLane(laneId) {
    for (const mesh of this.laneHitMeshes) {
      if (mesh.userData.laneId === laneId) {
        mesh.material.opacity = 0.28;
        mesh.material.color.setHex(laneId % 2 === 1 ? 0xf59e0b : 0x3b82f6);
      } else {
        mesh.material.opacity = 0.0;
      }
    }
  }

  clearLaneHighlight() {
    for (const mesh of this.laneHitMeshes) {
      mesh.material.opacity = 0.0;
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
