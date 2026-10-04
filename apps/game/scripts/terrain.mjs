// Builds public/terrain/bay.png: a stitched heightmap of the San Francisco Bay entrance from
// AWS Terrain Tiles (terrarium encoding: height = R*256 + G + B/256 - 32768 meters), plus
// bay.json with its bounds. Data: Mapzen / AWS Terrain Tiles, which combine USGS 3DEP, SRTM,
// GEBCO, ETOPO1 and others (see https://github.com/tilezen/joerd/blob/master/docs/attribution.md).
import { mkdirSync, writeFileSync } from "node:fs";
import { createCanvas, loadImage } from "@napi-rs/canvas";

const Z = 13;
// Golden Gate, the Presidio and downtown SF, the Marin Headlands, Alcatraz, Angel Island,
// Treasure Island and the Bay Bridge.
const BOUNDS = { west: -122.56, east: -122.33, south: 37.76, north: 37.88 };

const lon2x = (lon) => ((lon + 180) / 360) * 2 ** Z;
const lat2y = (lat) => {
  const r = (lat * Math.PI) / 180;
  return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** Z;
};
const x2lon = (x) => (x / 2 ** Z) * 360 - 180;
const y2lat = (y) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / 2 ** Z))) * 180) / Math.PI;

const x0 = Math.floor(lon2x(BOUNDS.west));
const x1 = Math.floor(lon2x(BOUNDS.east));
const y0 = Math.floor(lat2y(BOUNDS.north));
const y1 = Math.floor(lat2y(BOUNDS.south));
const W = (x1 - x0 + 1) * 256;
const H = (y1 - y0 + 1) * 256;
const canvas = createCanvas(W, H);
const ctx = canvas.getContext("2d");
for (let x = x0; x <= x1; x++) {
  for (let y = y0; y <= y1; y++) {
    const url = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${Z}/${x}/${y}.png`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    ctx.drawImage(await loadImage(Buffer.from(await res.arrayBuffer())), (x - x0) * 256, (y - y0) * 256);
  }
}
const out = new URL("../public/terrain/", import.meta.url).pathname;
mkdirSync(out, { recursive: true });
writeFileSync(`${out}bay.png`, canvas.toBuffer("image/png"));
const meta = {
  // The image covers whole tiles; these are its exact edges (Web Mercator, so rows are not
  // evenly spaced in latitude — the game maps through the same formulas).
  zoom: Z, tileX0: x0, tileY0: y0, width: W, height: H,
  west: x2lon(x0), east: x2lon(x1 + 1), north: y2lat(y0), south: y2lat(y1 + 1),
  encoding: "terrarium",
  attribution: "Terrain: Mapzen / AWS Terrain Tiles (USGS 3DEP, SRTM, GEBCO, ETOPO1 and others)",
};
writeFileSync(`${out}bay.json`, JSON.stringify(meta, null, 2) + "\n");
console.log(`${W}x${H} px, ${(x1 - x0 + 1) * (y1 - y0 + 1)} tiles`, meta);
