/**
 * features.ts - TypeScript port of features.py (keep the two in sync!).
 *
 * Per frame: 132 values = 2 hand slots x 66
 *   63 hand shape (wrist at origin, divided by hand size) + wrist x,y + hand scale
 * A sign = 30 frames sampled by TIME from the last 1.5 s, plus frame-to-frame velocity
 * -> model input of 30 x 264 floats.
 *
 * frameFeatures()/handVector() are worklets: they run on the camera thread.
 */
export const SEQ_LEN = 30;
export const WINDOW_SECONDS = 1.5;
export const NUM_HANDS = 2;
export const PER_HAND = 66;
export const FRAME_DIM = NUM_HANDS * PER_HAND; // 132
export const MODEL_DIM = FRAME_DIM * 2; // 264

export type Landmark = { x: number; y: number; z: number };

/** 21 landmarks (normalised image coords) -> 66-value vector, or null if degenerate. */
export function handVector(lms: Landmark[], mirrorX: boolean): Float32Array | null {
  'worklet';
  if (lms.length < 21) return null;
  const x0 = mirrorX ? 1 - lms[0].x : lms[0].x;
  const y0 = lms[0].y;
  const z0 = lms[0].z;
  let scale = 0;
  for (let i = 0; i < 21; i++) {
    const rx = (mirrorX ? 1 - lms[i].x : lms[i].x) - x0;
    const ry = lms[i].y - y0;
    const d = Math.sqrt(rx * rx + ry * ry);
    if (d > scale) scale = d;
  }
  if (scale < 1e-4) return null;
  const out = new Float32Array(66);
  for (let i = 0; i < 21; i++) {
    out[i * 3] = ((mirrorX ? 1 - lms[i].x : lms[i].x) - x0) / scale;
    out[i * 3 + 1] = (lms[i].y - y0) / scale;
    out[i * 3 + 2] = (lms[i].z - z0) / scale;
  }
  out[63] = x0;
  out[64] = y0;
  out[65] = scale;
  return out;
}

/** All hands in one frame -> fixed 132-value vector (hands ordered left -> right). */
export function frameFeatures(hands: Landmark[][], mirrorX: boolean): Float32Array {
  'worklet';
  const vecs: Float32Array[] = [];
  for (let i = 0; i < hands.length; i++) {
    const v = handVector(hands[i], mirrorX);
    if (v !== null) vecs.push(v);
  }
  vecs.sort((a, b) => a[63] - b[63]);
  const out = new Float32Array(FRAME_DIM);
  for (let i = 0; i < vecs.length && i < NUM_HANDS; i++) out.set(vecs[i], i * PER_HAND);
  return out;
}

/** Pick the SEQ_LEN frames nearest to evenly spaced times in [end-seconds, end]. */
export function resampleWindow(
  times: number[],
  feats: Float32Array[],
  endTime: number,
  seconds: number = WINDOW_SECONDS,
  n: number = SEQ_LEN,
): Float32Array[] {
  const out: Float32Array[] = [];
  for (let k = 0; k < n; k++) {
    const target = endTime - seconds + (seconds * k) / (n - 1);
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < times.length; i++) {
      const d = Math.abs(times[i] - target);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    out.push(feats[best]);
  }
  return out;
}

/** Fraction of frames in the window that contain at least one hand. */
export function handPresence(window: Float32Array[]): number {
  let c = 0;
  for (const f of window) if (f[PER_HAND - 1] > 0 || f[FRAME_DIM - 1] > 0) c++;
  return window.length ? c / window.length : 0;
}

/** window (30 x 132) -> flat model input (30 x 264): features + velocity. */
export function toModelInput(window: Float32Array[]): Float32Array {
  const out = new Float32Array(SEQ_LEN * MODEL_DIM);
  for (let t = 0; t < SEQ_LEN; t++) {
    const cur = window[t];
    const prev = t === 0 ? window[0] : window[t - 1];
    for (let j = 0; j < FRAME_DIM; j++) {
      out[t * MODEL_DIM + j] = cur[j];
      out[t * MODEL_DIM + FRAME_DIM + j] = cur[j] - prev[j];
    }
  }
  return out;
}
