import * as THREE from "three";

/**
 * The race line: a closed loop over open water (checked against the heightmap: no land within
 * ±70 m of the line). West of the Golden Gate → under it eastbound → around Alcatraz → back
 * under it westbound. 24 km; a song is one lap.
 */
const WAYPOINTS: [number, number][] = [
  [-3600, 450], [-1800, 480], [0, 450], [1800, 300], [3300, -100], [4200, -1500], [5200, -1900],
  [6100, -1300], [6000, -300], [5000, 300], [3200, 800], [1500, 700], [0, 300], [-1500, 120],
  [-3000, 200], [-4300, 700], [-4200, 1150],
];

export const ALTITUDE = 42; // meters: low over the water, clear under the bridge deck (~60 m)
export const LANE_WIDTH = 45; // meters between lanes
export const LANES = [-1, 0, 1] as const;

export class Course {
  private curve: THREE.CatmullRomCurve3;
  readonly length: number;

  constructor() {
    this.curve = new THREE.CatmullRomCurve3(
      WAYPOINTS.map(([x, z]) => new THREE.Vector3(x, ALTITUDE, z)),
      true,
      "centripetal",
    );
    this.curve.arcLengthDivisions = 4000;
    this.length = this.curve.getLength();
  }

  /** Point and frame at distance `s` along the loop (wraps around). */
  frame(s: number) {
    const u = (((s / this.length) % 1) + 1) % 1;
    const pos = this.curve.getPointAt(u);
    const tangent = this.curve.getTangentAt(u).setY(0).normalize();
    const right = new THREE.Vector3().crossVectors(tangent, new THREE.Vector3(0, 1, 0)).normalize();
    return { pos, tangent, right };
  }

  /** World position of lane `lane` (-1, 0, 1, or anything between) at distance `s`. */
  at(s: number, lane: number) {
    const f = this.frame(s);
    return f.pos.addScaledVector(f.right, lane * LANE_WIDTH);
  }

  /** Small markers along each lane, so the player can see the lanes ahead. */
  laneMarkers(spacing = 30) {
    const n = Math.floor(this.length / spacing);
    const geo = new THREE.CircleGeometry(2.2, 10);
    geo.rotateX(-Math.PI / 2);
    const mesh = new THREE.InstancedMesh(
      geo,
      new THREE.MeshBasicMaterial({ color: "#d8f6ff", transparent: true, opacity: 0.45, depthWrite: false }),
      n * LANES.length,
    );
    const m = new THREE.Matrix4();
    let k = 0;
    for (let i = 0; i < n; i++) {
      for (const lane of LANES) {
        const p = this.at(i * spacing, lane);
        p.y = 0.6;
        m.makeTranslation(p.x, p.y, p.z);
        mesh.setMatrixAt(k++, m);
      }
    }
    return mesh;
  }
}
