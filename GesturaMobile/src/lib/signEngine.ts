/**
 * signEngine.ts - port of the smoothing logic in test_model.py.
 * Pure TypeScript, no React Native imports, so it is unit-testable in Node.
 * Time is in SECONDS (any monotonic clock).
 */
import { WINDOW_SECONDS, handPresence, resampleWindow, toModelInput } from './features';

export const ENGINE_DEFAULTS = {
  confAuto: 0.85, // smoothed confidence needed in AUTO mode
  confManual: 0.6,
  stableCount: 3, // consecutive agreeing predictions (AUTO)
  predInterval: 0.1, // seconds between predictions
  emaAlpha: 0.5,
  cooldown: 1.5, // seconds of silence after accepting a word
  minPresence: 0.4, // skip when a hand is in < 40% of the window
  idleLabel: 'none',
};
export type EngineConfig = typeof ENGINE_DEFAULTS;

export type Accepted = { label: string; confidence: number };
export type EngineOutput = { probs: Float32Array | null; accepted: Accepted | null };

export class SignEngine {
  private times: number[] = [];
  private feats: Float32Array[] = [];
  private ema: Float32Array | null = null;
  private stableLabel: string | null = null;
  private stableN = 0;
  private lastPred = 0;
  private cooldownUntil = 0;
  private cfg: EngineConfig;

  constructor(
    private predict: (modelInput: Float32Array) => Float32Array,
    private labels: string[],
    cfg: Partial<EngineConfig> = {},
  ) {
    this.cfg = { ...ENGINE_DEFAULTS, ...cfg };
  }

  clear(): void {
    this.times = [];
    this.feats = [];
    this.ema = null;
    this.stableLabel = null;
    this.stableN = 0;
  }

  private window(now: number) {
    return resampleWindow(this.times, this.feats, now);
  }

  private argmax(p: Float32Array): number {
    let b = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
    return b;
  }

  /** AUTO mode: call once per camera frame with that frame's 132-d features. */
  push(now: number, feat: Float32Array): EngineOutput {
    this.times.push(now);
    this.feats.push(feat);
    while (this.times.length && now - this.times[0] > WINDOW_SECONDS + 0.7) {
      this.times.shift();
      this.feats.shift();
    }
    const out: EngineOutput = { probs: null, accepted: null };
    const enough = now - this.times[0] >= WINDOW_SECONDS * 0.9;
    if (!enough || now - this.lastPred < this.cfg.predInterval || now < this.cooldownUntil) {
      return { probs: this.ema, accepted: null };
    }
    this.lastPred = now;
    const w = this.window(now);
    if (handPresence(w) < this.cfg.minPresence) {
      this.ema = null;
      this.stableLabel = null;
      this.stableN = 0;
      return out;
    }
    const p = this.predict(toModelInput(w));
    if (this.ema === null) this.ema = Float32Array.from(p);
    else
      for (let i = 0; i < p.length; i++)
        this.ema[i] = this.cfg.emaAlpha * p[i] + (1 - this.cfg.emaAlpha) * this.ema[i];
    out.probs = this.ema;
    const best = this.argmax(this.ema);
    const label = this.labels[best];
    if (label !== this.cfg.idleLabel && this.ema[best] >= this.cfg.confAuto) {
      this.stableN = label === this.stableLabel ? this.stableN + 1 : 1;
      this.stableLabel = label;
      if (this.stableN >= this.cfg.stableCount) {
        out.accepted = { label, confidence: this.ema[best] };
        this.cooldownUntil = now + this.cfg.cooldown;
        this.clear();
      }
    } else {
      this.stableLabel = null;
      this.stableN = 0;
    }
    return out;
  }

  /** MANUAL (push-to-sign) mode: classify the last 1.5 s right now. */
  classifyNow(now: number): EngineOutput {
    if (!this.times.length) return { probs: null, accepted: null };
    const w = this.window(now);
    const p = this.predict(toModelInput(w));
    const best = this.argmax(p);
    const ok =
      this.labels[best] !== this.cfg.idleLabel &&
      p[best] >= this.cfg.confManual &&
      handPresence(w) >= this.cfg.minPresence;
    const accepted = ok ? { label: this.labels[best], confidence: p[best] } : null;
    if (accepted) this.cooldownUntil = now + this.cfg.cooldown;
    return { probs: p, accepted };
  }
}
