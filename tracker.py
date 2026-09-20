import os
import sys
import time
import cv2
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
SIGN_LABEL = "hello"  # Change label name when recording different signs
CSV_FILENAME = "sign_language_dataset.csv"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILENAME)

# 21 Hand Landmark Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 5), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)
]

def initialize_csv(file_path):
    """Ensure CSV file exists and contains valid header row."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            headers = ['label', 'hand']
            for i in range(21):
                headers.extend([f'x{i}', f'y{i}', f'z{i}'])
            writer.writerow(headers)
        print(f"Created new dataset CSV with headers at: {file_path}")
    else:
        print(f"Using existing dataset CSV at: {file_path}")

def count_recorded_samples(file_path):
    """Count number of data rows in CSV (excluding header)."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0
    try:
        with open(file_path, mode='r') as f:
            lines = [line for line in f if line.strip()]
            return max(0, len(lines) - 1)
    except Exception:
        return 0

def main():
    initialize_csv(CSV_PATH)
    recorded_count = count_recorded_samples(CSV_PATH)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    print("Initializing MediaPipe HandLandmarker...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    print("Opening camera stream...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open any camera. Please check camera permissions in System Settings.")
        sys.exit(1)

    print("Camera opened successfully!")
    print("-------------------------------------------------------")
    print(f"Current Target Sign Label: '{SIGN_LABEL}'")
    print(f"Dataset File Path: {CSV_PATH}")
    print(f"Total Samples Recorded So Far: {recorded_count}")
    print("-------------------------------------------------------")
    print("Controls:")
    print("  'r' or Spacebar : Start / Pause Recording")
    print("  's'             : Stop / Pause Recording")
    print("  'q' or ESC      : Quit Application")
    print("-------------------------------------------------------")

    prev_time = time.time()
    is_recording = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        h, w, _ = frame.shape

        # 1. MediaPipe detection on original UNFLIPPED frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        # 2. Flip frame horizontally for natural mirror display
        display_frame = cv2.flip(frame, 1)

        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Map landmark X coordinate to mirrored display_frame
                coords = []
                for lm in hand_landmarks:
                    cx = int((1.0 - lm.x) * w)
                    cy = int(lm.y * h)
                    coords.append((cx, cy))

                # Determine handedness label
                hand_label = "Unknown"
                if detection_result.handedness and idx < len(detection_result.handedness):
                    hand_label = detection_result.handedness[idx][0].category_name

                # Save raw landmark coordinates if recording dataset
                if is_recording:
                    row = [SIGN_LABEL, hand_label]
                    for lm in hand_landmarks:
                        row.extend([round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)])
                    with open(CSV_PATH, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                        f.flush()
                    recorded_count += 1

                # Draw skeleton lines
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(display_frame, coords[start_idx], coords[end_idx], (255, 200, 0), 2, cv2.LINE_AA)

                # Draw landmark keypoints (highlight fingertips)
                for i, (cx, cy) in enumerate(coords):
                    if i in [4, 8, 12, 16, 20]:
                        cv2.circle(display_frame, (cx, cy), 8, (0, 255, 255), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(display_frame, (cx, cy), 5, (255, 0, 128), -1, cv2.LINE_AA)

                # Display handedness label on screen
                if detection_result.handedness and idx < len(detection_result.handedness):
                    score = detection_result.handedness[idx][0].score
                    label_str = f"{hand_label} ({score:.0%})"
                    wrist_x, wrist_y = coords[0]
                    cv2.putText(display_frame, label_str, (wrist_x - 40, wrist_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # UI Overlay - Banner & Status
        cv2.putText(display_frame, f"FPS: {int(fps)} | Label: '{SIGN_LABEL}' | Total Saved: {recorded_count}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if is_recording:
            # Flashing red dot indicator
            dot_color = (0, 0, 255) if int(time.time() * 4) % 2 == 0 else (0, 0, 180)
            cv2.circle(display_frame, (30, 75), 10, dot_color, -1, cv2.LINE_AA)
            cv2.putText(display_frame, f"RECORDING SAMPLES... ({recorded_count} saved)", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(display_frame, (30, 75), 10, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(display_frame, "PAUSED (Press 'r' or SPACE to record)", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Hand Tracking - Sign Language Project", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # 'q', 'Q', or ESC
            print(f"Exiting tracker. Total recorded samples in CSV: {recorded_count}")
            break
        elif key in [ord('r'), ord('R'), 32]:  # 'r', 'R', or Spacebar
            is_recording = not is_recording
            status = "STARTED" if is_recording else "PAUSED"
            print(f"[{status}] Recording dataset for sign '{SIGN_LABEL}'. Total saved so far: {recorded_count}")
        elif key in [ord('s'), ord('S')]:
            is_recording = False
            print(f"[PAUSED] Recording dataset. Total saved so far: {recorded_count}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()