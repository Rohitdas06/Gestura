import os
import urllib.request
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "sign_language_dataset.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

# 1. Download MediaPipe Hand Landmarker model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model file...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded successfully!")

# 2. Initialize MediaPipe Tasks Hand Detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

records = []
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory not found at: {DATASET_DIR}")

categories = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print(f"Found categories ({len(categories)}): {categories}")

# 3. Process Images
for category in categories:
    cat_path = os.path.join(DATASET_DIR, category)
    images = os.listdir(cat_path)
    print(f"Processing '{category}' ({len(images)} images)...")

    for img_name in images:
        img_path = os.path.join(cat_path, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]

            # Wrist reference point (Landmark 0)
            wrist_x = landmarks[0].x
            wrist_y = landmarks[0].y
            wrist_z = landmarks[0].z

            row = [category]
            for lm in landmarks:
                # Wrist-centered relative coordinates (matches GesturaMobile normalization)
                row.append(lm.x - wrist_x)
                row.append(lm.y - wrist_y)
                row.append(lm.z - wrist_z)

            records.append(row)

detector.close()

# 4. Save to CSV
columns = ['label'] + [f'f_{i}' for i in range(63)]
df = pd.DataFrame(records, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nExtraction complete! Created '{OUTPUT_CSV}' with {len(df)} samples.")