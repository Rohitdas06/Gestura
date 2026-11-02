import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "sign_language_dataset.csv")
TFLITE_PATH = os.path.join(SCRIPT_DIR, "sign_language_model.tflite")
LABELS_PATH = os.path.join(SCRIPT_DIR, "labels.json")

# 1. Load Dataset
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset CSV not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
X = df.drop("label", axis=1).values.astype(np.float32)
y_raw = df["label"].values

# 2. Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
classes = list(label_encoder.classes_)

with open(LABELS_PATH, "w") as f:
    json.dump(classes, f, indent=2)

print(f"Loaded {len(df)} samples across {len(classes)} classes: {classes}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3. Build Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train Model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32
)

# 5. Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"\nTraining & conversion complete!")
print(f"Saved: '{TFLITE_PATH}' and '{LABELS_PATH}'")