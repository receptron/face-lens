import * as THREE from "three";
import { along } from "./geo";

/**
 * The Golden Gate Bridge at its real place and size (meters): towers 227 m above the water,
 * main span 1,280 m, side spans 343 m, deck 27 m wide about 67 m above the water, suspender
 * cables every 15.2 m. The bridge runs on a bearing of 354° (checked against the strait's
 * shorelines in the heightmap); the north tower stands at the Marin shore.
 */
export const BRIDGE = {
  bearing: 354,
  mainSpan: 1280,
  sideSpan: 343,
  towerHeight: 227,
  deckY: 67,
  deckWidth: 27,
  deckDepth: 7.6,
  cableSpacing: 27.4,
};

const ORANGE = new THREE.Color("#c0362c");

/** Mid-span centre in world coordinates: 15 m south of the published bridge coordinates. */
export const BRIDGE_CENTER = along(0, 0, BRIDGE.bearing, -15);

export class Bridge {
  readonly group = new THREE.Group();
  /** World → bridge-local (z along the bridge, north negative; x across; y up). */
  private toLocal = new THREE.Matrix4();

  constructor() {
    const mat = new THREE.MeshStandardMaterial({ color: ORANGE, roughness: 0.6, metalness: 0.2 });
    this.addTowers(mat);
    this.addDeck(mat);
    this.addCables(mat);
    this.group.position.set(BRIDGE_CENTER.x, 0, BRIDGE_CENTER.z);
    this.group.rotation.y = THREE.MathUtils.degToRad(360 - BRIDGE.bearing);
    this.group.updateMatrixWorld(true);
    this.toLocal.copy(this.group.matrixWorld).invert();
  }

  private local(p: THREE.Vector3) {
    return p.clone().applyMatrix4(this.toLocal);
  }

  /** True if a world point is inside a tower, a pier or the deck. */
  hits(p: THREE.Vector3): boolean {
    const l = this.local(p);
    const half = BRIDGE.mainSpan / 2;
    for (const tz of [-half, half]) {
      if (Math.abs(l.z - tz) < 9 && Math.abs(Math.abs(l.x) - BRIDGE.cableSpacing / 2) < 7 && l.y < BRIDGE.towerHeight + 2) {
        return true;
      }
    }
    const end = half + BRIDGE.sideSpan;
    return Math.abs(l.z) < end && Math.abs(l.x) < BRIDGE.deckWidth / 2 + 1 && Math.abs(l.y - this.deckTop(l.z) + BRIDGE.deckDepth / 2) < BRIDGE.deckDepth / 2 + 1;
  }

  /** True if a world point is under the main span: between the towers, below the deck, above the water. */
  underMainSpan(p: THREE.Vector3): boolean {
    const l = this.local(p);
    return Math.abs(l.z) < BRIDGE.mainSpan / 2 - 10 && Math.abs(l.x) < BRIDGE.deckWidth / 2 + 2 && l.y < this.deckTop(l.z) - BRIDGE.deckDepth;
  }

  /** Deck surface height: a slight arch, highest at mid-span. */
  private deckTop(z: number) {
    const half = BRIDGE.mainSpan / 2;
    return BRIDGE.deckY + 3 * (1 - Math.min(1, (z / half) ** 2));
  }

  private addTowers(mat: THREE.Material) {
    const half = BRIDGE.mainSpan / 2;
    const pierMat = new THREE.MeshStandardMaterial({ color: "#9a958c", roughness: 0.9 });
    const strutHeights = [45, 92, 132, 170, 206];
    for (const tz of [-half, half]) {
      // Concrete pier from the sea floor up to the base of the steel legs (13 m above the water).
      const pier = new THREE.Mesh(new THREE.BoxGeometry(48, 33, 30), pierMat);
      pier.position.set(0, 13 - 33 / 2, tz);
      this.group.add(pier);
      // Two tapering legs.
      for (const side of [-1, 1]) {
        const leg = new THREE.Mesh(taperedBox(10, 16, 6.5, 10, BRIDGE.towerHeight - 13), mat);
        leg.position.set((side * BRIDGE.cableSpacing) / 2, 13, tz);
        this.group.add(leg);
        // Stepped setbacks at each strut: a slightly wider collar.
        for (const h of strutHeights.slice(1)) {
          const collar = new THREE.Mesh(new THREE.BoxGeometry(9.5, 3, 14), mat);
          collar.position.set((side * BRIDGE.cableSpacing) / 2, h + 6, tz);
          this.group.add(collar);
        }
        // Aviation light.
        const light = new THREE.Mesh(
          new THREE.SphereGeometry(1.2, 10, 8),
          new THREE.MeshBasicMaterial({ color: new THREE.Color("#ff2a1a").multiplyScalar(4), toneMapped: false }),
        );
        light.position.set((side * BRIDGE.cableSpacing) / 2, BRIDGE.towerHeight + 1.5, tz);
        this.group.add(light);
      }
      // Portal struts between the legs (the deck passes between the lowest two).
      strutHeights.forEach((h, i) => {
        const tall = i === 0 ? 7 : 9 - i;
        const strut = new THREE.Mesh(new THREE.BoxGeometry(BRIDGE.cableSpacing - 6, tall, 5), mat);
        strut.position.set(0, h, tz);
        this.group.add(strut);
      });
      // Cable saddle housing on top.
      const cap = new THREE.Mesh(new THREE.BoxGeometry(BRIDGE.cableSpacing + 10, 4, 12), mat);
      cap.position.set(0, BRIDGE.towerHeight - 2, tz);
      this.group.add(cap);
    }
  }

  private addDeck(mat: THREE.Material) {
    const end = BRIDGE.mainSpan / 2 + BRIDGE.sideSpan;
    const segments = 120;
    // Roadway slab following the arch.
    const slab = new THREE.BoxGeometry(BRIDGE.deckWidth, 1.2, (2 * end) / segments);
    const truss = trussTexture();
    const trussMat = new THREE.MeshStandardMaterial({
      color: ORANGE, map: truss, alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.6, metalness: 0.2,
    });
    const roadMat = new THREE.MeshStandardMaterial({ color: "#3a3a3c", roughness: 0.9 });
    for (let i = 0; i < segments; i++) {
      const z = -end + ((i + 0.5) * 2 * end) / segments;
      const y = this.deckTop(z);
      const road = new THREE.Mesh(slab, roadMat);
      road.position.set(0, y, z);
      this.group.add(road);
      // Underside slab, in orange.
      const under = new THREE.Mesh(slab, mat);
      under.position.set(0, y - BRIDGE.deckDepth, z);
      this.group.add(under);
      // The two stiffening trusses: see-through X-bracing.
      for (const side of [-1, 1]) {
        const panel = new THREE.Mesh(new THREE.PlaneGeometry((2 * end) / segments, BRIDGE.deckDepth), trussMat);
        panel.rotation.y = Math.PI / 2;
        panel.position.set((side * BRIDGE.deckWidth) / 2, y - BRIDGE.deckDepth / 2, z);
        this.group.add(panel);
      }
    }
  }

  private addCables(mat: THREE.Material) {
    const half = BRIDGE.mainSpan / 2;
    const end = half + BRIDGE.sideSpan;
    const top = BRIDGE.towerHeight - 3;
    const low = BRIDGE.deckY + 5;
    // Main span: parabola from tower top to just above the deck. Side spans: down to the anchorages.
    const cableY = (z: number) => {
      const a = Math.abs(z);
      if (a <= half) return low + (top - low) * (a / half) ** 2;
      const t = (a - half) / BRIDGE.sideSpan;
      return top + (BRIDGE.deckY - top) * t + 18 * Math.sin(Math.PI * t) * -0.5;
    };
    const cableMat = new THREE.MeshStandardMaterial({ color: ORANGE, roughness: 0.5, metalness: 0.3 });
    const suspender = new THREE.CylinderGeometry(0.22, 0.22, 1, 5);
    suspender.translate(0, 0.5, 0);
    const spacing = 15.2;
    const count = Math.floor((2 * end) / spacing) * 2;
    const hangers = new THREE.InstancedMesh(suspender, mat, count);
    const m = new THREE.Matrix4();
    let n = 0;
    for (const side of [-1, 1]) {
      const x = (side * BRIDGE.cableSpacing) / 2;
      const pts: THREE.Vector3[] = [];
      for (let z = -end; z <= end; z += 10) pts.push(new THREE.Vector3(x, cableY(z), z));
      const tube = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 400, 0.9, 8);
      this.group.add(new THREE.Mesh(tube, cableMat));
      for (let z = -end + spacing; z < end; z += spacing) {
        if (Math.abs(Math.abs(z) - half) < 6) continue; // no hanger inside the tower
        const deck = this.deckTop(z);
        const len = cableY(z) - deck;
        if (len < 1.5) continue;
        m.compose(new THREE.Vector3(x, deck, z), new THREE.Quaternion(), new THREE.Vector3(1, len, 1));
        hangers.setMatrixAt(n++, m);
      }
    }
    hangers.count = n;
    this.group.add(hangers);
  }
}

/** Box whose cross-section shrinks linearly from bottom (w0×d0) to top (w1×d1). */
function taperedBox(w0: number, d0: number, w1: number, d1: number, h: number) {
  const g = new THREE.BoxGeometry(1, h, 1, 1, 1, 1);
  const p = g.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < p.count; i++) {
    const t = (p.getY(i) + h / 2) / h;
    p.setX(i, p.getX(i) * (w0 + (w1 - w0) * t));
    p.setZ(i, p.getZ(i) * (d0 + (d1 - d0) * t));
    p.setY(i, p.getY(i) + h / 2);
  }
  g.computeVertexNormals();
  return g;
}

/** X-bracing for the deck's stiffening truss, drawn once into a canvas. */
function trussTexture() {
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 64;
  const g = c.getContext("2d")!;
  g.strokeStyle = "#fff";
  g.lineWidth = 7;
  g.strokeRect(3, 3, 122, 58);
  g.beginPath();
  g.moveTo(0, 0);
  g.lineTo(64, 64);
  g.lineTo(128, 0);
  g.moveTo(64, 64);
  g.lineTo(64, 0);
  g.stroke();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
