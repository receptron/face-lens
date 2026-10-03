/** Clothing color from MediaPipe's selfie multiclass segmentation (category 4 = clothes). */

export const CLOTHES_CATEGORY = 4;

export interface ClothingColor {
  /** Color name, e.g. "yellow", "navy". */
  name: ColorName;
  /** Share of the visible clothing, 0..1. */
  share: number;
  /** Average sRGB of the pixels with this name. */
  rgb: [number, number, number];
}

export const COLOR_NAMES = [
  "black", "white", "gray", "red", "orange", "yellow", "green", "teal",
  "blue", "navy", "purple", "pink", "brown", "beige",
] as const;
export type ColorName = (typeof COLOR_NAMES)[number];

/** sRGB 0..255 → CIELAB (D65). */
export function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  const lin = (c: number) => {
    c /= 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const x = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047;
  const y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
  const z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const [fx, fy, fz] = [f(x), f(y), f(z)];
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** Names a color by hue angle in CIE LCh, with lightness and chroma deciding the shade words. */
export function nameColor(r: number, g: number, b: number): ColorName {
  const [L, A, B] = rgbToLab(r, g, b);
  const C = Math.hypot(A, B);
  if (C < 10 || (L < 22 && C < 18)) return L < 28 ? "black" : L > 82 ? "white" : "gray";
  const h = ((Math.atan2(B, A) * 180) / Math.PI + 360) % 360;
  if (h >= 330 || h < 42) {
    // Reds: light or bluish-red is pink; dark stays red (maroon), not brown.
    if (L > 68 || (h >= 330 && L > 45)) return "pink";
    return "red";
  }
  if (h < 75) {
    if (C < 40 && L >= 58) return "beige"; // tan, camel, khaki
    if (L < 50) return "brown";
    return "orange";
  }
  if (h < 105) {
    if (C < 32 && L >= 58) return "beige";
    if (L < 55) return "green"; // olive
    return "yellow";
  }
  if (h < 185) return "green";
  if (h < 235) return L < 40 ? "navy" : "teal";
  // Saturated blues sit around h≈295 in Lab, so blue runs up to 308.
  if (h < 308) return L < 38 ? "navy" : "blue";
  return L > 70 ? "pink" : "purple";
}

/**
 * Names the clothing colors in a frame.
 * @param mask category per pixel at maskW × maskH (the segmenter's output)
 * @param pixels RGBA of the same frame, downscaled to pixW × pixH
 * @param minYFrac ignore everything above this fraction of the height (e.g. the chin)
 */
export function clothingColors(
  mask: Uint8Array,
  maskW: number,
  maskH: number,
  pixels: Uint8ClampedArray,
  pixW: number,
  pixH: number,
  minYFrac: number,
): ClothingColor[] {
  const sums = new Map<ColorName, [number, number, number, number]>();
  let total = 0;
  const sx = maskW / pixW;
  const sy = maskH / pixH;
  for (let y = Math.max(0, Math.floor(minYFrac * pixH)); y < pixH; y++) {
    const my = Math.min(maskH - 1, Math.floor((y + 0.5) * sy));
    for (let x = 0; x < pixW; x++) {
      const mx = Math.min(maskW - 1, Math.floor((x + 0.5) * sx));
      if (mask[my * maskW + mx] !== CLOTHES_CATEGORY) continue;
      const i = (y * pixW + x) * 4;
      const [r, g, b] = [pixels[i], pixels[i + 1], pixels[i + 2]];
      const name = nameColor(r, g, b);
      const s = sums.get(name) ?? [0, 0, 0, 0];
      s[0] += r;
      s[1] += g;
      s[2] += b;
      s[3] += 1;
      sums.set(name, s);
      total++;
    }
  }
  // Too few clothing pixels (face-only framing): report nothing rather than noise.
  if (total < 0.02 * pixW * pixH) return [];
  return [...sums.entries()]
    .map(([name, [r, g, b, n]]) => ({
      name,
      share: n / total,
      rgb: [Math.round(r / n), Math.round(g / n), Math.round(b / n)] as [number, number, number],
    }))
    .filter((c) => c.share >= 0.12)
    .sort((a, b) => b.share - a.share)
    .slice(0, 3);
}
