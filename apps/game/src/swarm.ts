import * as THREE from "three";

/** One drone. Index 0 of the alive list is not special; `leader` points at the steered one. */
interface Drone {
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  alive: boolean;
  /** On its way to a target (a one-way trip). */
  strike: THREE.Vector3 | null;
  onHit: (() => void) | null;
  /** Per-drone phases for the wander noise, so each one drifts on its own. */
  phase: [number, number, number];
  /** How strongly this drone wants to keep up (a little personality). */
  eagerness: number;
}

const COUNT = 48;
const NEIGHBOR_R = 45; // meters: who counts as a flockmate
const SEPARATE_R = 9; // meters: personal space
const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _forward = new THREE.Vector3(0, 0, -1);
const LEADER_COLOR = new THREE.Color("#ffb84d");
const FOLLOWER_COLOR = new THREE.Color("#8fe9ff");
const STRIKE_COLOR = new THREE.Color("#ff3b2f").multiplyScalar(6);
const STRIKE_LIGHT_SCALE = new THREE.Vector3(5, 5, 5);

/**
 * A flock that follows one steered drone, like birds behind a leader.
 *
 * The player flies the leader directly. Every other drone runs Reynolds' boids rules —
 * separation from close neighbours, alignment with their heading, cohesion toward their
 * centre — plus a pull toward the leader and a slow, per-drone wander, so the flock stretches,
 * bunches and ripples instead of holding a shape. Lost drones are not replaced.
 */
export class Swarm {
  readonly mesh: THREE.InstancedMesh;
  readonly lights: THREE.InstancedMesh;
  /** Unit direction the leader is flying. */
  readonly heading = new THREE.Vector3(1, 0, 0);
  private drones: Drone[] = [];
  private leaderIndex = 0;
  /** Strike drones that crashed before reaching their target (for tuning). */
  strikeCrashes = 0;

  constructor(start: THREE.Vector3, headingDeg: number) {
    this.setHeading(headingDeg, 0);
    this.mesh = new THREE.InstancedMesh(
      droneGeometry(),
      new THREE.MeshStandardMaterial({ color: "#30343b", roughness: 0.45, metalness: 0.5 }),
      COUNT,
    );
    // Small status LEDs: visible, but kept below the bloom threshold so they do not glow.
    this.lights = new THREE.InstancedMesh(new THREE.SphereGeometry(0.3, 6, 4), new THREE.MeshBasicMaterial({ color: "#ffffff" }), COUNT);
    this.mesh.frustumCulled = false;
    this.lights.frustumCulled = false;
    for (let i = 0; i < COUNT; i++) {
      const offset = i === 0 ? new THREE.Vector3() : new THREE.Vector3().randomDirection().multiplyScalar(15 + Math.random() * 35);
      this.drones.push({
        pos: start.clone().add(offset).addScaledVector(this.heading, i === 0 ? 0 : -30),
        vel: this.heading.clone().multiplyScalar(80),
        alive: true,
        strike: null,
        onHit: null,
        phase: [Math.random() * 100, Math.random() * 100, Math.random() * 100],
        eagerness: 0.8 + Math.random() * 0.4,
      });
    }
    this.colorLights();
  }

  /** The drone the player steers. */
  get leader(): THREE.Vector3 {
    return this.drones[this.leaderIndex].pos;
  }

  /** Drones still flying with the flock (the leader included, strikers excluded). */
  get count() {
    return this.drones.filter((d) => d.alive && !d.strike).length;
  }

  setHeading(yawDeg: number, pitchDeg: number) {
    const y = THREE.MathUtils.degToRad(yawDeg);
    const p = THREE.MathUtils.degToRad(pitchDeg);
    // Compass yaw: 0 = north (-z), 90 = east (+x).
    this.heading.set(Math.sin(y) * Math.cos(p), Math.sin(p), -Math.cos(y) * Math.cos(p)).normalize();
  }

  /** Puts the leader (and the flock around it) at `p`, e.g. when a round starts. */
  reset(p: THREE.Vector3) {
    for (const [i, d] of this.drones.entries()) {
      d.alive = true;
      d.strike = null;
      d.onHit = null;
      const offset = i === this.leaderIndex ? new THREE.Vector3() : new THREE.Vector3().randomDirection().multiplyScalar(15 + Math.random() * 35);
      d.pos.copy(p).add(offset).addScaledVector(this.heading, i === this.leaderIndex ? 0 : -30);
      d.vel.copy(this.heading).multiplyScalar(80);
    }
    this.colorLights();
  }

  /** Sends the follower nearest to `target` on a one-way strike. False if none is left. */
  strike(target: THREE.Vector3, onHit: () => void): boolean {
    let best: Drone | null = null;
    let bestD = Infinity;
    this.drones.forEach((d, i) => {
      if (!d.alive || d.strike || i === this.leaderIndex) return;
      const dist = d.pos.distanceToSquared(target);
      if (dist < bestD) [best, bestD] = [d, dist];
    });
    if (!best) return false;
    const d = best as Drone;
    d.strike = target;
    d.onHit = onHit;
    return true;
  }

  /**
   * Advances the flock. `collides(p)` tells whether a point is inside terrain or a structure;
   * drones that hit something are lost for good. Returns where drones died this frame.
   */
  update(dt: number, t: number, speed: number, collides: (p: THREE.Vector3) => boolean, floorAt: (p: THREE.Vector3) => number) {
    const deaths: THREE.Vector3[] = [];
    const leader = this.drones[this.leaderIndex];
    // The leader flies exactly where the player points it.
    leader.vel.copy(this.heading).multiplyScalar(speed);
    leader.pos.addScaledVector(leader.vel, dt);

    const flock = this.drones.filter((d) => d.alive && !d.strike);
    for (const d of this.drones) {
      if (!d.alive || d === leader) continue;
      const acc = new THREE.Vector3();
      if (d.strike) {
        // One-way dive at the target, faster than the flock.
        const to = d.strike.clone().sub(d.pos);
        if (to.length() < 8) {
          d.onHit?.();
          deaths.push(d.pos.clone());
          d.alive = false;
          d.strike = null;
          continue;
        }
        acc.copy(to.normalize().multiplyScalar(speed * 2.6)).sub(d.vel).multiplyScalar(3);
      } else {
        const sep = new THREE.Vector3();
        const ali = new THREE.Vector3();
        const coh = new THREE.Vector3();
        let n = 0;
        for (const o of flock) {
          if (o === d) continue;
          const away = d.pos.clone().sub(o.pos);
          const dist = away.length();
          if (dist > NEIGHBOR_R) continue;
          if (dist < SEPARATE_R && dist > 0.01) sep.addScaledVector(away, (SEPARATE_R - dist) / (dist * SEPARATE_R));
          ali.add(o.vel);
          coh.add(o.pos);
          n++;
        }
        if (n > 0) {
          ali.divideScalar(n).sub(d.vel).multiplyScalar(0.9);
          coh.divideScalar(n).sub(d.pos).multiplyScalar(0.25);
        }
        // Follow the leader by matching its velocity, plus a correction toward a point a little
        // behind it. Steering toward a desired velocity (not springing toward a position) is
        // what damps the swing: without it the flock overshoots the leader after every turn.
        const toSlot = leader.pos.clone().addScaledVector(this.heading, -25).sub(d.pos).multiplyScalar(0.8);
        if (toSlot.length() > speed * 0.9) toSlot.setLength(speed * 0.9);
        const follow = leader.vel.clone().add(toSlot).sub(d.vel).multiplyScalar(1.6 * d.eagerness);
        // Slow wander: each drone drifts on its own, like birds jostling.
        const [a, b, c] = d.phase;
        const wander = new THREE.Vector3(Math.sin(t * 0.7 + a), Math.sin(t * 0.9 + b) * 0.6, Math.sin(t * 0.8 + c)).multiplyScalar(14);
        acc.addScaledVector(sep, 220).add(ali).add(coh).add(follow).add(wander);
        // Pull up before the ground or the water.
        const floor = floorAt(d.pos) + 12;
        if (d.pos.y < floor) acc.y += (floor - d.pos.y) * 6;
      }
      // Limit acceleration and keep speed in a bird-like band around the leader's.
      const maxAcc = d.strike ? 400 : 120;
      if (acc.length() > maxAcc) acc.setLength(maxAcc);
      d.vel.addScaledVector(acc, dt);
      const v = d.vel.length();
      const lo = d.strike ? speed : speed * 0.4;
      const hi = d.strike ? speed * 3 : speed * 1.45;
      if (v < lo) d.vel.setLength(lo);
      else if (v > hi) d.vel.setLength(hi);
      d.pos.addScaledVector(d.vel, dt);
    }

    for (const [i, d] of this.drones.entries()) {
      if (d.alive && d.strike && collides(d.pos)) {
        this.strikeCrashes++;
        deaths.push(d.pos.clone());
        d.alive = false;
        d.strike = null;
        continue;
      }
      if (d.alive && !d.strike && collides(d.pos)) {
        deaths.push(d.pos.clone());
        d.alive = false;
        if (i === this.leaderIndex) this.promote();
      }
    }
    this.writeInstances();
    return deaths;
  }

  /** The leader is gone: the nearest surviving flock member takes over. */
  private promote() {
    const old = this.drones[this.leaderIndex].pos;
    let best = -1;
    let bestD = Infinity;
    this.drones.forEach((d, i) => {
      if (!d.alive || d.strike) return;
      const dist = d.pos.distanceToSquared(old);
      if (dist < bestD) [best, bestD] = [i, dist];
    });
    if (best >= 0) this.leaderIndex = best;
    this.colorLights();
  }

  /** Furthest any follower is ahead of the leader along its heading, in meters (overshoot). */
  maxAhead() {
    const leader = this.drones[this.leaderIndex];
    let max = -Infinity;
    for (const d of this.drones) {
      if (!d.alive || d.strike || d === leader) continue;
      max = Math.max(max, d.pos.clone().sub(leader.pos).dot(this.heading));
    }
    return max;
  }

  /** Flock centre relative to the leader: lateral (+ = right) and behind, in meters. */
  offset() {
    const c = this.centroid(new THREE.Vector3()).sub(this.leader);
    const right = new THREE.Vector3().crossVectors(this.heading, new THREE.Vector3(0, 1, 0)).normalize();
    return { lateral: c.dot(right), behind: -c.dot(this.heading) };
  }

  /** True once every drone is lost. */
  get wiped() {
    return !this.drones[this.leaderIndex].alive;
  }

  centroid(out: THREE.Vector3) {
    out.set(0, 0, 0);
    let n = 0;
    for (const d of this.drones) {
      if (d.alive && !d.strike) {
        out.add(d.pos);
        n++;
      }
    }
    return n ? out.divideScalar(n) : out.copy(this.leader);
  }

  private colorLights() {
    this.drones.forEach((_, i) => this.lights.setColorAt(i, i === this.leaderIndex ? LEADER_COLOR : FOLLOWER_COLOR));
    if (this.lights.instanceColor) this.lights.instanceColor.needsUpdate = true;
  }

  private writeInstances() {
    let k = 0;
    const scale = new THREE.Vector3(1, 1, 1);
    for (const [i, d] of this.drones.entries()) {
      if (!d.alive) continue;
      const dir = d.vel.lengthSq() > 1 ? d.vel.clone().normalize() : this.heading;
      _q.setFromUnitVectors(_forward, dir);
      _m.compose(d.pos, _q, scale);
      this.mesh.setMatrixAt(k, _m);
      // A drone on a strike lights up: a big red light, brighter than white so the bloom pass
      // makes it glow, easy to follow on its way to the orb. Others keep a small, plain LED.
      const striking = !!d.strike;
      _m.compose(d.pos.clone().add(new THREE.Vector3(0, 0.45, 0)), _q, striking ? STRIKE_LIGHT_SCALE : scale);
      this.lights.setMatrixAt(k, _m);
      // Instance colours follow the drone, not the slot.
      this.lights.setColorAt(k, striking ? STRIKE_COLOR : i === this.leaderIndex ? LEADER_COLOR : FOLLOWER_COLOR);
      k++;
    }
    this.mesh.count = k;
    this.lights.count = k;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.lights.instanceMatrix.needsUpdate = true;
    if (this.lights.instanceColor) this.lights.instanceColor.needsUpdate = true;
  }
}

/** A small quadcopter: body, four arms, four rotor discs. Scaled up (~4 m) to read from the chase camera. */
function droneGeometry() {
  const parts: THREE.BufferGeometry[] = [];
  parts.push(new THREE.BoxGeometry(1.6, 0.6, 2.2));
  for (const [x, z] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const arm = new THREE.BoxGeometry(2.2, 0.2, 0.25);
    arm.rotateY(Math.atan2(z, x));
    arm.translate(x * 0.8, 0, z * 0.8);
    parts.push(arm);
    const rotor = new THREE.CylinderGeometry(0.9, 0.9, 0.08, 12);
    rotor.translate(x * 1.6, 0.25, z * 1.6);
    parts.push(rotor);
  }
  const pos: number[] = [];
  const nor: number[] = [];
  for (const g of parts) {
    const ng = g.index ? g.toNonIndexed() : g;
    pos.push(...(ng.attributes.position.array as Float32Array));
    nor.push(...(ng.attributes.normal.array as Float32Array));
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
  return out;
}
