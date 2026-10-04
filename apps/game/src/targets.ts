import * as THREE from "three";

interface Orb {
  mesh: THREE.Mesh;
  base: THREE.Vector3;
  alive: boolean;
  /** Seconds until it reappears elsewhere. */
  respawn: number;
  /** A strike is on its way. */
  claimed: boolean;
}

const COLORS = ["#ff5ad1", "#ffd23f", "#5affb4", "#7aa7ff"];

/** Glowing floating orbs over the Bay: the targets. Destroyed ones come back somewhere else. */
export class Targets {
  readonly group = new THREE.Group();
  private orbs: Orb[] = [];

  constructor(
    count: number,
    private place: () => THREE.Vector3,
  ) {
    const geo = new THREE.IcosahedronGeometry(9, 2);
    for (let i = 0; i < count; i++) {
      const color = new THREE.Color(COLORS[i % COLORS.length]);
      // Brighter than white, so the bloom pass makes it glow.
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: color.clone().multiplyScalar(5), toneMapped: false }));
      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(16, 16, 12),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18, depthWrite: false, toneMapped: false }),
      );
      mesh.add(halo);
      const base = place();
      mesh.position.copy(base);
      this.group.add(mesh);
      this.orbs.push({ mesh, base, alive: true, respawn: 0, claimed: false });
    }
  }

  update(dt: number, t: number) {
    this.orbs.forEach((o, i) => {
      if (!o.alive) {
        o.respawn -= dt;
        if (o.respawn <= 0) {
          o.base.copy(this.place());
          o.alive = true;
          o.claimed = false;
          o.mesh.visible = true;
        }
        return;
      }
      o.mesh.position.copy(o.base).add(new THREE.Vector3(0, Math.sin(t * 0.9 + i) * 6, 0));
      o.mesh.rotation.y = t * 0.5 + i;
      const pulse = 1 + 0.08 * Math.sin(t * 4 + i);
      o.mesh.scale.setScalar(pulse);
    });
  }

  /**
   * Best target for a strike: alive, not already claimed, ahead of the swarm within `range`,
   * on the requested side (`side` -1 = left, 1 = right, 0 = either), nearest to straight ahead.
   */
  pick(from: THREE.Vector3, heading: THREE.Vector3, range: number, side: -1 | 0 | 1): Orb | null {
    const right = new THREE.Vector3().crossVectors(heading, new THREE.Vector3(0, 1, 0)).normalize();
    let best: Orb | null = null;
    let bestScore = Infinity;
    for (const o of this.orbs) {
      if (!o.alive || o.claimed) continue;
      const to = o.mesh.position.clone().sub(from);
      const dist = to.length();
      if (dist > range) continue;
      const ahead = to.dot(heading) / dist;
      if (ahead < 0.2) continue;
      const lateral = to.dot(right);
      if (side !== 0 && Math.sign(lateral) !== side && Math.abs(lateral) > 15) continue;
      const score = dist * (2 - ahead);
      if (score < bestScore) [best, bestScore] = [o, score];
    }
    return best;
  }

  claim(o: Orb) {
    o.claimed = true;
  }

  destroy(o: Orb) {
    o.alive = false;
    o.mesh.visible = false;
    o.respawn = 4 + Math.random() * 3;
  }

  /** Orb currently highlighted as the next strike target (for the HUD). */
  lockMarker(o: Orb | null, marker: THREE.Object3D) {
    marker.visible = !!o;
    if (o) marker.position.copy(o.mesh.position);
  }
}
export type { Orb };

/** Additive particle bursts for drone impacts and crashes. */
export class Explosions {
  readonly points: THREE.Points;
  private pos: Float32Array;
  private vel: Float32Array;
  private life: Float32Array;
  private col: Float32Array;
  private next = 0;

  constructor(private max = 4000) {
    this.pos = new Float32Array(max * 3);
    this.vel = new Float32Array(max * 3);
    this.life = new Float32Array(max);
    this.col = new Float32Array(max * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(this.col, 3));
    // Round, soft-edged sparks (untextured points render as squares, which look like blocks
    // when a burst happens near the camera).
    this.points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: 2.2, map: sparkTexture(), vertexColors: true, transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
      }),
    );
    this.points.frustumCulled = false;
  }

  burst(at: THREE.Vector3, color: THREE.Color, n = 120, speed = 60) {
    for (let k = 0; k < n; k++) {
      const i = this.next;
      this.next = (this.next + 1) % this.max;
      const dir = new THREE.Vector3().randomDirection().multiplyScalar(speed * (0.3 + Math.random()));
      this.pos.set([at.x, at.y, at.z], i * 3);
      this.vel.set([dir.x, dir.y, dir.z], i * 3);
      this.life[i] = 0.8 + Math.random() * 0.8;
      this.col.set([color.r, color.g, color.b], i * 3);
    }
  }

  update(dt: number) {
    for (let i = 0; i < this.max; i++) {
      if (this.life[i] <= 0) continue;
      this.life[i] -= dt;
      const k = i * 3;
      this.vel[k + 1] -= 25 * dt;
      for (let a = 0; a < 3; a++) {
        this.vel[k + a] *= 1 - 1.5 * dt;
        this.pos[k + a] += this.vel[k + a] * dt;
      }
      const fade = Math.max(0, this.life[i]);
      if (fade <= 0) this.pos[k + 1] = -1e6;
      this.col[k] *= 0.985;
      this.col[k + 1] *= 0.97;
      this.col[k + 2] *= 0.95;
    }
    this.points.geometry.attributes.position.needsUpdate = true;
    this.points.geometry.attributes.color.needsUpdate = true;
  }
}

function sparkTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.6)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}
