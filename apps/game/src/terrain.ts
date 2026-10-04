import * as THREE from "three";
import { latToTileY, lonToTileX, tileXToLon, tileYToLat, toLatLon, toWorld } from "./geo";

interface Meta {
  zoom: number;
  tileX0: number;
  tileY0: number;
  width: number;
  height: number;
  attribution: string;
}

/** Real elevation of the Bay entrance (meters, below 0 = sea floor) and its mesh. */
export class Terrain {
  readonly mesh: THREE.Mesh;

  private constructor(
    private heights: Float32Array,
    private meta: Meta,
  ) {
    this.mesh = this.buildMesh(2);
  }

  static async load(base: string): Promise<Terrain> {
    const meta = (await (await fetch(`${base}terrain/bay.json`)).json()) as Meta;
    const img = await createImageBitmap(await (await fetch(`${base}terrain/bay.png`)).blob());
    const canvas = new OffscreenCanvas(meta.width, meta.height);
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const { data } = ctx.getImageData(0, 0, meta.width, meta.height);
    const heights = new Float32Array(meta.width * meta.height);
    // Terrarium encoding.
    for (let i = 0; i < heights.length; i++) {
      heights[i] = data[i * 4] * 256 + data[i * 4 + 1] + data[i * 4 + 2] / 256 - 32768;
    }
    return new Terrain(heights, meta);
  }

  get attribution() {
    return this.meta.attribution;
  }

  /** Ground (or sea floor) height in meters at world (x, z); bilinear. */
  heightAt(x: number, z: number): number {
    const { lat, lon } = toLatLon(x, z);
    const px = (lonToTileX(lon, this.meta.zoom) - this.meta.tileX0) * 256;
    const py = (latToTileY(lat, this.meta.zoom) - this.meta.tileY0) * 256;
    return this.sample(px, py);
  }

  private sample(px: number, py: number) {
    const { width: w, height: h } = this.meta;
    const x = Math.min(Math.max(px, 0), w - 1.001);
    const y = Math.min(Math.max(py, 0), h - 1.001);
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0;
    const i = y0 * w + x0;
    const a = this.heights[i], b = this.heights[i + 1], c = this.heights[i + w], d = this.heights[i + w + 1];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
  }

  /** World-space bounds of the heightmap. */
  bounds() {
    const { zoom, tileX0, tileY0, width, height } = this.meta;
    const nw = toWorld(tileYToLat(tileY0, zoom), tileXToLon(tileX0, zoom));
    const se = toWorld(tileYToLat(tileY0 + height / 256, zoom), tileXToLon(tileX0 + width / 256, zoom));
    return { minX: nw.x, maxX: se.x, minZ: nw.z, maxZ: se.z };
  }

  private buildMesh(step: number): THREE.Mesh {
    const { zoom, tileX0, tileY0, width, height } = this.meta;
    const cols = Math.floor((width - 1) / step) + 1;
    const rows = Math.floor((height - 1) / step) + 1;
    const pos = new Float32Array(cols * rows * 3);
    const col = new Float32Array(cols * rows * 3);
    const c = new THREE.Color();
    for (let r = 0; r < rows; r++) {
      const py = r * step;
      const lat = tileYToLat(tileY0 + py / 256, zoom);
      for (let q = 0; q < cols; q++) {
        const px = q * step;
        const lon = tileXToLon(tileX0 + px / 256, zoom);
        const { x, z } = toWorld(lat, lon);
        const y = this.heights[py * width + px];
        const k = (r * cols + q) * 3;
        pos[k] = x;
        pos[k + 1] = y;
        pos[k + 2] = z;
        groundColor(c, y, this.slope(px, py), lat, lon);
        col[k] = c.r;
        col[k + 1] = c.g;
        col[k + 2] = c.b;
      }
    }
    const index: number[] = [];
    for (let r = 0; r < rows - 1; r++) {
      for (let q = 0; q < cols - 1; q++) {
        const a = r * cols + q, b = a + 1, d = a + cols, e = d + 1;
        index.push(a, d, b, b, d, e);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.setIndex(index);
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 });
    return new THREE.Mesh(geo, mat);
  }

  private slope(px: number, py: number) {
    const dx = this.sample(px + 1, py) - this.sample(px - 1, py);
    const dy = this.sample(px, py + 1) - this.sample(px, py - 1);
    return Math.hypot(dx, dy) / 30; // ~15 m per pixel at zoom 13
  }
}

// Rough urban footprints (lat/lon polygons): San Francisco north and east of the Presidio
// and Golden Gate Park, and Oakland / Treasure Island. Buildings come later; for now this
// only tints the ground.
const URBAN: [number, number][][] = [
  [[37.808, -122.448], [37.808, -122.385], [37.75, -122.385], [37.75, -122.47], [37.772, -122.47], [37.79, -122.448]],
  [[37.835, -122.333], [37.835, -122.3], [37.79, -122.3], [37.79, -122.333]],
];

function inPolygon(lat: number, lon: number, poly: [number, number][]) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [ai, oi] = poly[i];
    const [aj, oj] = poly[j];
    if (oi > lon !== oj > lon && lat < ((aj - ai) * (lon - oi)) / (oj - oi) + ai) inside = !inside;
  }
  return inside;
}

/** Golden summer grass on the hills, dark scrub on steep slopes, sand at the shore, gray city. */
function groundColor(c: THREE.Color, y: number, slope: number, lat: number, lon: number) {
  if (y < -2) return c.setRGB(0.12, 0.16, 0.18, THREE.SRGBColorSpace); // sea floor, seen through the water
  if (y < 4) return c.setRGB(0.62, 0.56, 0.44, THREE.SRGBColorSpace); // beach
  if (URBAN.some((p) => inPolygon(lat, lon, p))) {
    const v = 0.42 + 0.06 * Math.sin(lat * 9000) * Math.cos(lon * 9000);
    return c.setRGB(v, v * 0.98, v * 0.95, THREE.SRGBColorSpace);
  }
  const grass = new THREE.Color().setRGB(0.62, 0.52, 0.3, THREE.SRGBColorSpace);
  const scrub = new THREE.Color().setRGB(0.24, 0.29, 0.18, THREE.SRGBColorSpace);
  return c.copy(grass).lerp(scrub, Math.min(1, slope * 1.6 + Math.max(0, (y - 150) / 300)));
}
