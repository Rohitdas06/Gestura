import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train():
    print("Loading dataset from sign_language_dataset.csv...")
    if not pd.io.common.file_exists('sign_language_dataset.csv'):
        print("Error: sign_language_dataset.csv not found.")
        return

    df = pd.read_csv('sign_language_dataset.csv')
    print(f"Dataset loaded: {len(df)} total samples.")

    # 1. Separate features (X) and target label (y)
    X = df.drop(['label', 'hand'], axis=1, errors='ignore').values
    y = df['label'].values

    # 2. Encode categorical text labels into numerical class indices
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    classes = list(encoder.classes_)
    num_classes = len(classes)
    print(f"Classes found ({num_classes}): {classes}")

    # Save class labels to labels.json for live inference matching
    with open('labels.json', 'w') as f:
        json.dump(classes, f)
    print("Saved label mappings to 'labels.json'")

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 4. Neural Network Architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting Neural Network Training...")
    history = model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate Final Accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

    # 5. Convert to TensorFlow Lite (.tflite) for Mobile / Edge AI
    print("Exporting model to 'sign_language_model.tflite'...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('sign_language_model.tflite', 'wb') as f:
        f.write(tflite_model)

    # 6. Save as Keras format for React Native/TensorFlow.js conversion
    model.save('sign_language_model.h5')

    print("SUCCESS: Model updated and exported to both .tflite and .h5 formats!")

if __name__ == '__main__':
    train()