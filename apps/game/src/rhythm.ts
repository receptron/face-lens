/** Songs, the notes generated from their beat grid, and judging. */

export interface Song {
  id: string;
  file: string;
  title: string;
  creator: string;
  page: string;
  license: string;
  level: "Normal" | "Hard";
  bpm: number;
  /** Time of the first beat, seconds. */
  offset: number;
  duration: number;
  /** Loudness per beat, 0..1 (quiet intros and breakdowns are low). */
  energy: number[];
}

export interface Note {
  /** Song time it must be hit, seconds. */
  time: number;
  lane: -1 | 0 | 1;
  /** Fingers to show. */
  count: number;
  judged: boolean;
  result: Judgement | null;
}

export type Judgement = "perfect" | "good" | "miss";

/** Timing windows, seconds. Wider than a button game: cameras and hand tracking add delay. */
export const PERFECT = 0.15;
export const GOOD = 0.35;

/** Deterministic PRNG, so a song always has the same chart. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Notes on the beat grid. Density follows the music: loud sections every 2 beats (every beat
 * on Hard), mid every 2–4, quiet every 4 or none. Finger counts go up to 3 (Normal) or 5 (Hard)
 * and, like lanes, change only by small steps when notes come quickly.
 */
export function chart(song: Song): Note[] {
  const rand = mulberry32([...song.id].reduce((a, c) => a * 31 + c.charCodeAt(0), 7));
  const beat = 60 / song.bpm;
  const hard = song.level === "Hard";
  const maxCount = hard ? 5 : 3;
  const notes: Note[] = [];
  let lane: -1 | 0 | 1 = 0;
  let count = 1;
  let lastTime = -10;
  for (let i = 8; i < song.energy.length - 4; i++) {
    const e = song.energy[i];
    const step = e > 0.7 ? (hard ? 1 : 2) : e > 0.4 ? (hard ? 2 : 4) : e > 0.15 ? 4 : 0;
    if (step === 0 || i % step !== 0) continue;
    const time = song.offset + i * beat;
    const gap = time - lastTime;
    // Lane: random, but at most one lane over when the previous note was under a second ago.
    const lanes: (-1 | 0 | 1)[] = gap < 1 ? ([lane - 1, lane, lane + 1].filter((l) => l >= -1 && l <= 1) as (-1 | 0 | 1)[]) : [-1, 0, 1];
    lane = lanes[Math.floor(rand() * lanes.length)];
    // Count: 1..max, a step of at most 2 when quick.
    const lo = gap < 1 ? Math.max(1, count - 2) : 1;
    const hi = gap < 1 ? Math.min(maxCount, count + 2) : maxCount;
    count = lo + Math.floor(rand() * (hi - lo + 1));
    notes.push({ time, lane, count, judged: false, result: null });
    lastTime = time;
  }
  return notes;
}

/** Perfect when the right count appeared within PERFECT of the beat, else Good. */
export function grade(note: Note, changedAt: number): Judgement {
  return Math.abs(changedAt - note.time) <= PERFECT ? "perfect" : "good";
}

export function rank(accuracy: number) {
  return accuracy >= 0.95 ? "S" : accuracy >= 0.85 ? "A" : accuracy >= 0.7 ? "B" : accuracy >= 0.5 ? "C" : "D";
}
