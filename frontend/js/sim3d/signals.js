/**
 * signals.js — 3D cantilever traffic signal gantries, illuminated 3-lens signal heads,
 * and per-lane stop-line LED indicator bars aligned with RHT incoming lanes 0..7.
 */
import * as THREE from 'three';
import { ROAD_HALF, STOP_LINE_OFFSET } from './world.js';

const POLE_HEIGHT = 5.4;
const POLE_RADIUS = 0.12;
const HOUSING_W = 0.65;
const HOUSING_H = 1.9;
const HOUSING_D = 0.45;
const LENS_RADIUS = 0.22;

const COLOR_OFF_RED = 0x2b1010;
const COLOR_OFF_YELLOW = 0x2b2510;
const COLOR_OFF_GREEN = 0x102b15;

const COLOR_ON_RED = 0xef4444;
const COLOR_ON_YELLOW = 0xeab308;
const COLOR_ON_GREEN = 0x22c55e;

/**
 * Each pole stands on the incoming approach's right-side sidewalk near the stop line,
 * with a horizontal mast arm carrying two signal heads directly over the two incoming lanes
 * and lenses facing oncoming traffic.
 */
const SIGNAL_POST_CONFIGS = [
  {
    id: 'N',
    polePos: [-(ROAD_HALF + 0.9), -STOP_LINE_OFFSET],
    armAxis: 'x',
    armTarget: -1.5,
    rotY: Math.PI, // lenses (+Z local) face North (-Z) toward Southbound drivers
    lanes: [
      { laneId: 0, coord: -4.5 }, // N_STR
      { laneId: 1, coord: -1.5 }, // N_TRN
    ],
    barDir: 'NS',
    barOffset: -STOP_LINE_OFFSET,
  },
  {
    id: 'S',
    polePos: [ROAD_HALF + 0.9, STOP_LINE_OFFSET],
    armAxis: 'x',
    armTarget: 1.5,
    rotY: 0, // lenses (+Z local) face South (+Z) toward Northbound drivers
    lanes: [
      { laneId: 2, coord: 4.5 }, // S_STR
      { laneId: 3, coord: 1.5 }, // S_TRN
    ],
    barDir: 'NS',
    barOffset: STOP_LINE_OFFSET,
  },
  {
    id: 'E',
    polePos: [STOP_LINE_OFFSET, -(ROAD_HALF + 0.9)],
    armAxis: 'z',
    armTarget: -1.5,
    rotY: Math.PI / 2, // lenses (+Z local) face East (+X) toward Westbound drivers
    lanes: [
      { laneId: 4, coord: -4.5 }, // E_STR
      { laneId: 5, coord: -1.5 }, // E_TRN
    ],
    barDir: 'EW',
    barOffset: STOP_LINE_OFFSET,
  },
  {
    id: 'W',
    polePos: [-STOP_LINE_OFFSET, ROAD_HALF + 0.9],
    armAxis: 'z',
    armTarget: 1.5,
    rotY: -Math.PI / 2, // lenses (+Z local) face West (-X) toward Eastbound drivers
    lanes: [
      { laneId: 6, coord: 4.5 }, // W_STR
      { laneId: 7, coord: 1.5 }, // W_TRN
    ],
    barDir: 'EW',
    barOffset: -STOP_LINE_OFFSET,
  },
];

export class TrafficSignals3D {
  constructor(scene) {
    this.group = new THREE.Group();
    this.group.name = 'signals';

    this.laneLenses = {};
    this.laneStopBars = {};
    this._buildSignals();

    scene.add(this.group);
  }

  _buildSignals() {
    const poleMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.45,
      metalness: 0.45,
    });
    const housingMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.6,
      metalness: 0.2,
    });

    const lensSpecs = [
      { name: 'red', y: 0.55, off: COLOR_OFF_RED, on: COLOR_ON_RED },
      { name: 'yellow', y: 0.0, off: COLOR_OFF_YELLOW, on: COLOR_ON_YELLOW },
      { name: 'green', y: -0.55, off: COLOR_OFF_GREEN, on: COLOR_ON_GREEN },
    ];

    for (const post of SIGNAL_POST_CONFIGS) {
      const [px, pz] = post.polePos;

      // Vertical post
      const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(POLE_RADIUS, POLE_RADIUS * 1.15, POLE_HEIGHT, 10),
        poleMat
      );
      pole.position.set(px, POLE_HEIGHT / 2, pz);
      pole.castShadow = true;
      this.group.add(pole);

      // Horizontal cantilever mast arm reaching over both incoming lanes
      const armStart = post.armAxis === 'x' ? px : pz;
      const armLen = Math.abs(post.armTarget - armStart) + 0.45;
      const armCenter = (armStart + post.armTarget) / 2;
      const armGeo =
        post.armAxis === 'x'
          ? new THREE.BoxGeometry(armLen, 0.14, 0.14)
          : new THREE.BoxGeometry(0.14, 0.14, armLen);
      const arm = new THREE.Mesh(armGeo, poleMat);
      arm.castShadow = true;
      if (post.armAxis === 'x') {
        arm.position.set(armCenter, POLE_HEIGHT - 0.12, pz);
      } else {
        arm.position.set(px, POLE_HEIGHT - 0.12, armCenter);
      }
      this.group.add(arm);

      // Signal heads + per-lane stop-line LED strip
      for (const laneSpec of post.lanes) {
        const { laneId, coord } = laneSpec;
        const headGroup = new THREE.Group();
        headGroup.rotation.y = post.rotY;

        const housing = new THREE.Mesh(
          new THREE.BoxGeometry(HOUSING_W, HOUSING_H, HOUSING_D),
          housingMat
        );
        housing.castShadow = true;
        headGroup.add(housing);

        const lenses = {};
        for (const spec of lensSpecs) {
          const mat = new THREE.MeshStandardMaterial({
            color: spec.off,
            emissive: 0x000000,
            roughness: 0.2,
            metalness: 0.1,
          });
          const lensMesh = new THREE.Mesh(
            new THREE.SphereGeometry(LENS_RADIUS, 12, 8),
            mat
          );
          lensMesh.position.set(0, spec.y, HOUSING_D / 2 + 0.02);
          headGroup.add(lensMesh);
          lenses[spec.name] = { mesh: lensMesh, onColor: spec.on, offColor: spec.off };
        }
        this.laneLenses[laneId] = lenses;

        if (post.armAxis === 'x') {
          headGroup.position.set(coord, POLE_HEIGHT - 0.55, pz);
        } else {
          headGroup.position.set(px, POLE_HEIGHT - 0.55, coord);
        }
        this.group.add(headGroup);

        // Glowing stop-line indicator bar right at the lane stop line (clearly visible from Top View)
        const barGeo =
          post.barDir === 'NS'
            ? new THREE.PlaneGeometry(2.55, 0.24)
            : new THREE.PlaneGeometry(0.24, 2.55);
        const barMat = new THREE.MeshBasicMaterial({ color: COLOR_ON_RED });
        const barMesh = new THREE.Mesh(barGeo, barMat);
        barMesh.rotation.x = -Math.PI / 2;
        if (post.barDir === 'NS') {
          barMesh.position.set(coord, 0.022, post.barOffset + (post.barOffset < 0 ? 0.28 : -0.28));
        } else {
          barMesh.position.set(post.barOffset + (post.barOffset < 0 ? 0.28 : -0.28), 0.022, coord);
        }
        this.group.add(barMesh);
        this.laneStopBars[laneId] = barMesh;
      }
    }
  }

  update(snapshot) {
    if (!snapshot) return;

    const stage = snapshot.signal_stage || 'GREEN';
    const greenLanes = new Set(snapshot.phase_green_lanes || []);

    for (let lane = 0; lane < 8; lane++) {
      const lenses = this.laneLenses[lane];
      const stopBar = this.laneStopBars[lane];
      if (!lenses) continue;

      const isActive = greenLanes.has(lane);

      for (const light of Object.values(lenses)) {
        light.mesh.material.color.setHex(light.offColor);
        light.mesh.material.emissive.setHex(0x000000);
        light.mesh.material.emissiveIntensity = 0;
      }

      let barColor = COLOR_ON_RED;
      if (isActive && stage === 'GREEN') {
        lenses.green.mesh.material.color.setHex(lenses.green.onColor);
        lenses.green.mesh.material.emissive.setHex(lenses.green.onColor);
        lenses.green.mesh.material.emissiveIntensity = 1.25;
        barColor = COLOR_ON_GREEN;
      } else if (isActive && stage === 'YELLOW') {
        lenses.yellow.mesh.material.color.setHex(lenses.yellow.onColor);
        lenses.yellow.mesh.material.emissive.setHex(lenses.yellow.onColor);
        lenses.yellow.mesh.material.emissiveIntensity = 1.25;
        barColor = COLOR_ON_YELLOW;
      } else {
        lenses.red.mesh.material.color.setHex(lenses.red.onColor);
        lenses.red.mesh.material.emissive.setHex(lenses.red.onColor);
        lenses.red.mesh.material.emissiveIntensity = 0.85;
        barColor = COLOR_ON_RED;
      }

      if (stopBar) {
        stopBar.material.color.setHex(barColor);
      }
    }
  }
}
