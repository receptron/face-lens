/**
 * World coordinates: meters in a local tangent plane centred on the Golden Gate Bridge.
 * x = east, y = up, z = south (so north is -z, as Three.js cameras look down -z).
 */

export const ORIGIN = { lat: 37.8199, lon: -122.4786 };
const M_PER_DEG_LAT = 110574;
const M_PER_DEG_LON = 111320 * Math.cos((ORIGIN.lat * Math.PI) / 180);

export function toWorld(lat: number, lon: number): { x: number; z: number } {
  return { x: (lon - ORIGIN.lon) * M_PER_DEG_LON, z: -(lat - ORIGIN.lat) * M_PER_DEG_LAT };
}

export function toLatLon(x: number, z: number): { lat: number; lon: number } {
  return { lat: ORIGIN.lat - z / M_PER_DEG_LAT, lon: ORIGIN.lon + x / M_PER_DEG_LON };
}

/** Point `d` meters from (x, z) along a compass bearing in degrees (0 = north, 90 = east). */
export function along(x: number, z: number, bearingDeg: number, d: number) {
  const b = (bearingDeg * Math.PI) / 180;
  return { x: x + Math.sin(b) * d, z: z - Math.cos(b) * d };
}

// Web Mercator tile coordinates (the heightmap's own grid).
export function lonToTileX(lon: number, zoom: number) {
  return ((lon + 180) / 360) * 2 ** zoom;
}
export function latToTileY(lat: number, zoom: number) {
  const r = (lat * Math.PI) / 180;
  return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** zoom;
}
export function tileXToLon(x: number, zoom: number) {
  return (x / 2 ** zoom) * 360 - 180;
}
export function tileYToLat(y: number, zoom: number) {
  return (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / 2 ** zoom))) * 180) / Math.PI;
}
