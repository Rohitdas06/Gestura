"""
tracker.py - record sign CLIPS (1.5 s each) for the sequence model.

    python tracker.py --person alice
    python tracker.py --person bob --labels hello thank_you welcome none --target 60

Keys
    SPACE  record one clip (2 s countdown, then 1.5 s recording)
    a      auto mode: keep recording clips back-to-back until --target is reached
    n / b  next / previous label
    u      undo (delete) the last saved clip
    q/ESC  quit

Tips for a model that works on stage
    * Record each sign >= 60 times, and a 'none' class >= 80 times
      ('none' = hand resting, hand moving randomly, adjusting hair, waving at nothing...).
    * Start moving the moment recording begins; the clip is 1.5 s long.
    * Vary distance, angle, speed, lighting and background between runs.
    * Get at least 3 different people. Each run gets its own session id, so the
      trainer can test on sessions/people it has never seen (honest accuracy).
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

from camera_utils import HandTracker, draw_hands, open_camera, put_text
from features import (DATA_DIR, SEQ_LEN, WINDOW_SECONDS, count_clips,
                      hand_presence, resample_window)

IDLE, COUNTDOWN, RECORDING = 0, 1, 2
MIN_PRESENCE = 0.7          # a real sign must have a hand in >= 70% of frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="p1", help="who is signing (used to group data)")
    ap.add_argument("--labels", nargs="+", default=["hello", "thank_you", "welcome", "none"])
    ap.add_argument("--target", type=int, default=60, help="clips per label for auto mode")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()

    person = args.person.replace("__", "_")
    session = time.strftime("%Y%m%d-%H%M%S")
    labels = args.labels
    li = 0
    for lab in labels:
        os.makedirs(os.path.join(args.data_dir, lab), exist_ok=True)
    counts = count_clips(args.data_dir)

    tracker = HandTracker()
    cap = open_camera(args.camera)

    state, state_end = IDLE, 0.0
    auto, next_auto = False, 0.0
    rec_start, rec_buf = 0.0, []
    msg, msg_until = "", 0.0
    last_saved = None
    prev, fps = time.monotonic(), 0.0

    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)                     # mirror BEFORE detection (matches test_model.py)
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-3))
        prev = now
        result, feat = tracker.process(frame)
        draw_hands(frame, result)
        label = labels[li]

        # ---- state machine ----
        if auto and state == IDLE and now >= next_auto:
            if counts.get(label, 0) >= args.target:
                auto = False
                msg, msg_until = f"'{label}' reached {args.target} clips", now + 3
            else:
                state, state_end = COUNTDOWN, now + 1.0
        if state == COUNTDOWN and now >= state_end:
            state, rec_start, rec_buf = RECORDING, now, []
        if state == RECORDING:
            rec_buf.append((now, feat))
            if now - rec_start >= WINDOW_SECONDS:
                times = [t for t, _ in rec_buf]
                window = resample_window(times, [f for _, f in rec_buf], rec_start + WINDOW_SECONDS)
                presence = hand_presence(window)
                if label != "none" and presence < MIN_PRESENCE:
                    msg, msg_until = f"Rejected: hand visible only {presence:.0%} of the time - redo", now + 2.5
                else:
                    counts[label] = counts.get(label, 0) + 1
                    path = os.path.join(args.data_dir, label,
                                        f"{person}__{session}__{label}_{counts[label]:04d}.npy")
                    np.save(path, window.astype(np.float32))
                    last_saved = (path, label)
                    msg, msg_until = f"Saved {label} #{counts[label]}", now + 1.2
                state = IDLE
                next_auto = now + 1.0

        # ---- overlay ----
        put_text(frame, f"Label: {label}   saved: {counts.get(label, 0)}/{args.target}   FPS {fps:.0f}",
                 (20, 35), 0.75)
        put_text(frame, "  ".join(f"{k}:{v}" for k, v in counts.items()), (20, 68), 0.55, (200, 200, 200), 1)
        hands = len(result.hand_landmarks or [])
        put_text(frame, f"hands: {hands}", (20, 98), 0.6, (0, 255, 0) if hands else (0, 0, 255), 1)
        h, w = frame.shape[:2]
        if state == COUNTDOWN:
            put_text(frame, str(max(1, math.ceil(state_end - now))), (w // 2 - 30, h // 2 + 30), 3.0, (0, 255, 255), 6)
        elif state == RECORDING:
            cv2.circle(frame, (w - 40, 40), 14, (0, 0, 255), -1)
            cv2.rectangle(frame, (0, h - 12), (int(w * (now - rec_start) / WINDOW_SECONDS), h), (0, 0, 255), -1)
            put_text(frame, "SIGN NOW", (w // 2 - 110, 60), 1.2, (0, 0, 255), 3)
        else:
            put_text(frame, "SPACE record | a auto | n/b label | u undo | q quit", (20, h - 20), 0.55, (200, 200, 200), 1)
        if auto:
            put_text(frame, "AUTO", (w - 120, 100), 0.8, (0, 200, 255))
        if now < msg_until:
            put_text(frame, msg, (20, 140), 0.8, (0, 255, 255))
        cv2.imshow("Record sign clips", frame)

        # ---- keys ----
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        elif k == 32:                                   # SPACE
            if state == IDLE:
                auto = False
                state, state_end = COUNTDOWN, now + 2.0
            else:
                state, auto = IDLE, False
        elif k in (ord("a"), ord("A")):
            auto = not auto
            next_auto = now + 2.0
        elif k in (ord("n"), ord("N")) and state == IDLE:
            li, auto = (li + 1) % len(labels), False
        elif k in (ord("b"), ord("B")) and state == IDLE:
            li, auto = (li - 1) % len(labels), False
        elif k in (ord("u"), ord("U")) and last_saved and state == IDLE:
            path, lab = last_saved
            if os.path.exists(path):
                os.remove(path)
                counts[lab] -= 1
                msg, msg_until = f"Deleted last '{lab}' clip", now + 1.5
            last_saved = None

    cap.release()
    cv2.destroyAllWindows()
    print("Clips per label:", count_clips(args.data_dir))


if __name__ == "__main__":
    main()
