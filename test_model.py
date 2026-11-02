"""
test_model.py - live sign -> text -> speech demo.

    python test_model.py

Keys
    m      switch AUTO (continuous) <-> MANUAL (push-to-sign) mode
    SPACE  MANUAL mode: capture the next 1.5 s and classify it
    t      toggle speech on/off
    c      clear the transcript
    q/ESC  quit

AUTO mode keeps a 1.5 s rolling window, smooths predictions and only "speaks" a word
once it is confident AND stable, then waits a short cooldown. MANUAL mode uses
exactly the same model but only classifies when you press SPACE - the safest choice
if lighting or the room is unpredictable during the presentation.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from camera_utils import HandTracker, draw_hands, open_camera, put_text
from features import (BASE_DIR, WINDOW_SECONDS, add_velocity, hand_presence, resample_window)

# ---- tunables (adjust at the venue if needed) ----
CONF_AUTO = 0.85          # smoothed confidence needed to accept a word in AUTO mode
CONF_MANUAL = 0.60
STABLE_COUNT = 3          # consecutive agreeing predictions required (AUTO)
PRED_INTERVAL = 0.10      # seconds between predictions
EMA_ALPHA = 0.5           # weight of the newest prediction
COOLDOWN = 1.5            # seconds of silence after a word is accepted
MIN_PRESENCE = 0.4        # skip prediction unless a hand is in >= 40% of the window
IDLE_LABEL = "none"


class Classifier:
    def __init__(self):
        self.labels = json.load(open(os.path.join(BASE_DIR, "labels.json")))
        self.interp = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "sign_language_model.tflite"))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def predict(self, window):
        x = add_velocity(window)[None].astype(np.float32)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])[0].copy()


class Speaker:
    """Non-blocking text-to-speech. macOS: `say`. Elsewhere: pyttsx3 (optional)."""

    def __init__(self):
        self.enabled = True
        self.q = queue.Queue()
        self.mac = sys.platform == "darwin"
        if not self.mac:
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as e:
            print(f"[speech disabled] pyttsx3 unavailable: {e}")
            return
        while True:
            engine.say(self.q.get())
            engine.runAndWait()

    def say(self, text):
        if not self.enabled:
            return
        text = text.replace("_", " ")
        if self.mac:
            subprocess.Popen(["say", text])
        else:
            self.q.put(text)


def top3(probs, labels):
    order = np.argsort(probs)[::-1][:3]
    return [(labels[i], float(probs[i])) for i in order]


def main():
    clf = Classifier()
    tracker = HandTracker()
    cap = open_camera()
    speaker = Speaker()
    labels = clf.labels

    buf = deque()                       # (time, feature) rolling buffer
    mode = "AUTO"
    ema, stable_label, stable_n = None, None, 0
    last_pred, cooldown_until = 0.0, 0.0
    shown, shown_until = None, 0.0      # last accepted word banner
    manual_start = None
    live_probs = None
    transcript = []

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)                       # same mirroring as tracker.py
        now = time.monotonic()
        result, feat = tracker.process(frame)
        draw_hands(frame, result)
        buf.append((now, feat))
        while buf and now - buf[0][0] > WINDOW_SECONDS + 0.7:
            buf.popleft()

        def window_now():
            return resample_window([t for t, _ in buf], [f for _, f in buf], now)

        enough = bool(buf) and (now - buf[0][0]) >= WINDOW_SECONDS * 0.9

        def accept(word, conf):
            nonlocal shown, shown_until, cooldown_until, ema, stable_label, stable_n
            shown, shown_until = f"{word.replace('_', ' ').upper()}  ({conf:.0%})", now + 2.0
            transcript.append(word.replace("_", " "))
            speaker.say(word)
            cooldown_until = now + COOLDOWN
            ema, stable_label, stable_n = None, None, 0
            buf.clear()

        # ---------------- AUTO mode ----------------
        if mode == "AUTO" and enough and now - last_pred >= PRED_INTERVAL and now >= cooldown_until:
            last_pred = now
            w = window_now()
            if hand_presence(w) < MIN_PRESENCE:
                ema, stable_label, stable_n, live_probs = None, None, 0, None
            else:
                p = clf.predict(w)
                ema = p if ema is None else EMA_ALPHA * p + (1 - EMA_ALPHA) * ema
                live_probs = ema
                best = int(np.argmax(ema))
                word = labels[best]
                if word != IDLE_LABEL and ema[best] >= CONF_AUTO:
                    stable_n = stable_n + 1 if word == stable_label else 1
                    stable_label = word
                    if stable_n >= STABLE_COUNT:
                        accept(word, float(ema[best]))
                else:
                    stable_label, stable_n = None, 0

        # ---------------- MANUAL mode ----------------
        if mode == "MANUAL" and manual_start is not None and now - manual_start >= WINDOW_SECONDS:
            manual_start = None
            w = window_now()
            p = clf.predict(w)
            live_probs = p
            best = int(np.argmax(p))
            if labels[best] != IDLE_LABEL and p[best] >= CONF_MANUAL and hand_presence(w) >= MIN_PRESENCE:
                accept(labels[best], float(p[best]))
            else:
                shown, shown_until = "Not recognised - try again", now + 2.0

        # ---------------- overlay ----------------
        h, w_ = frame.shape[:2]
        put_text(frame, f"Mode: {mode}   Speech: {'ON' if speaker.enabled else 'OFF'}", (20, 35), 0.7, (200, 200, 200), 1)
        if mode == "MANUAL":
            if manual_start is not None:
                put_text(frame, "SIGN NOW...", (w_ // 2 - 120, 80), 1.3, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, h - 12), (int(w_ * (now - manual_start) / WINDOW_SECONDS), h), (0, 0, 255), -1)
            else:
                put_text(frame, "Press SPACE, then sign", (20, 70), 0.7, (0, 255, 255), 2)
        if shown and now < shown_until:
            put_text(frame, shown, (20, 120), 1.4, (0, 255, 0), 3)
        if live_probs is not None and len(buf):
            for i, (name, pr) in enumerate(top3(live_probs, labels)):
                y0 = 160 + i * 28
                cv2.rectangle(frame, (20, y0 - 16), (20 + int(200 * pr), y0 + 4), (255, 160, 0), -1)
                put_text(frame, f"{name} {pr:.0%}", (230, y0), 0.6, (255, 255, 255), 1)
        put_text(frame, "Transcript: " + " ".join(transcript[-8:]), (20, h - 25), 0.8, (255, 255, 255), 2)
        cv2.imshow("Sign Language AI - Live", frame)

        # ---------------- keys ----------------
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        elif k in (ord("m"), ord("M")):
            mode = "MANUAL" if mode == "AUTO" else "AUTO"
            ema, stable_label, stable_n, live_probs, manual_start = None, None, 0, None, None
        elif k == 32 and mode == "MANUAL" and manual_start is None:
            buf.clear()
            manual_start = now
        elif k in (ord("t"), ord("T")):
            speaker.enabled = not speaker.enabled
        elif k in (ord("c"), ord("C")):
            transcript.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
