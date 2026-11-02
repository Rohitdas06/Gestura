# 🤟 Gestura — Real-Time Sign Language Interpreter & Speech Translator

[![React Native](https://img.shields.io/badge/React%20Native-0.86.3-blue.svg)](https://reactnative.dev/)
[![Expo SDK](https://img.shields.io/badge/Expo%20SDK-57-black.svg)](https://expo.dev/)
[![MediaPipe](https://img.shields.io/badge/Google%20MediaPipe-0.10.18-orange.svg)](https://developers.google.com/mediapipe)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-Fast--TFLite%20v3-FF6F00.svg)](https://github.com/mrousavy/react-native-fast-tflite)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.12-yellow.svg)](https://www.python.org/)

**Gestura** is an intelligent, accessible on-device sign language translation platform. It uses computer vision and deep learning to bridge communication between the deaf/hard-of-hearing community and non-signers in real time—converting camera-captured sign gestures into **live visual HUD predictions, text transcripts, and natural spoken speech**.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Supported Models & Vocabulary](#-supported-models--vocabulary)
- [End-to-End Data Pipeline](#-end-to-end-data-pipeline)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the Mobile App (GesturaMobile)](#running-the-mobile-app-gesturamobile)
  - [Running the ML Training Pipeline](#running-the-ml-training-pipeline)
  - [Running the Desktop Prototype](#running-the-desktop-prototype)
- [Engineering & Optimization Highlights](#-engineering--optimization-highlights)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Key Features

### 1. Multi-Model Sign Language Recognition
* **Standard Model (`STD`)**: Classifies 8 universal gesture classes (`openPalm`, `closedFist`, `fingerCircle`, `fingerSymbols`, `singleFingerBend`, `multiFingerBend`, `semiOpenPalm`, `semiOpenFist`).
* **Indian Sign Language (`ISL`)**: Translates 35 classes including the alphabet (**A–Z**) and numbers (**1–9**).
* **Gesture Model (`GESTURE`)**: Fast general-purpose symbolic gesture engine.
* **Instant Cycling**: Switch models on-the-fly with the top control bar.

### 2. Live Hand Skeleton & Framing Reticle
* Real-time vector rendering of all **21 MediaPipe hand joints** and connecting **bone segments**.
* Color-coded joints: Wrist (gold), knuckles/joints (cyan), fingertips (neon magenta).
* Centered **targeting reticle** with corner accents that glow neon green and announce **`HAND LOCKED`** upon detection.
* Automated front-camera coordinate mirroring.

### 3. Dual Operation Modes
* **MANUAL (Push-to-Sign)**: Zero background photo captures. Viewfinder runs at a silky 60 FPS. Hold your sign, press **`SIGN NOW`**, and receive an instant translation locked on screen.
* **AUTO Mode**: Periodic background inference (~1 s cadence) with debounced text-to-speech output.

### 4. Interactive Sign Dictionary & Practice Guide (Explore Tab)
* Searchable encyclopedia of supported gestures and letters.
* Filter pills for **Standard**, **ISL Alphabet**, and **Numbers (1–9)**.
* Expandable flashcards featuring **exact finger placement guides** and **camera positioning pro-tips**.

### 5. Sentence Transcript & Audio Synthesis
* **Text-to-Speech (TTS)**: Translates recognized signs into spoken voice via `expo-speech`.
* **Transcript Actions**:
  * **`⌫ Undo`**: Erase the last recognized word if misclassified.
  * **`▶ Speak All`**: Re-read the entire composed sentence aloud.
  * **`📤 Share / Copy`**: Copy to clipboard or export to WhatsApp, SMS, or Notes.
  * **`✕ Clear`**: Reset sentence history.

### 6. Hardware Controls
* **Front / Back Camera Flip**: Toggle between self-signing (front) and translating a conversation partner (rear).
* **Flashlight / Torch**: Conditional rear-camera illumination for low-light environments (hardware-safe).

---

## 🏛 System Architecture

```
                                 GESTURA SYSTEM ARCHITECTURE
                                 
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   FRONTEND (React Native / Expo)                           │
 │                                                                                             │
 │   ┌───────────────────────────────┐                  ┌──────────────────────────────────┐   │
 │   │      src/app/index.tsx        │                  │       src/app/explore.tsx        │   │
 │   │   (Live Viewfinder & HUD)     │                  │    (Interactive Dictionary)      │   │
 │   └──────────────┬────────────────┘                  └──────────────────────────────────┘   │
 │                  │                                                                          │
 │                  ▼                                                                          │
 │   ┌───────────────────────────────┐                  ┌──────────────────────────────────┐   │
 │   │    react-native-vision-camera │                  │     HandSkeletonOverlay.tsx      │   │
 │   │ (CameraX / PhotoOutput API)   │                  │ (21 Joints + 21 Bone Vectors)    │   │
 │   └──────────────┬────────────────┘                  └─────────────────▲────────────────┘   │
 └──────────────────┼─────────────────────────────────────────────────────┼────────────────────┘
                    │ File path (JPEG)                                    │ 63 Landmarks
                    ▼                                                     │
 ┌────────────────────────────────────────────────────────────────────────┴────────────────────┐
 │                     NATIVE ANDROID MODULE (expo-mediapipe-hands)                            │
 │                                                                                             │
 │   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
 │   │                          ExpoMediapipeHandsModule.kt                                │   │
 │   │                                                                                     │   │
 │   │  1. EXIF Orientation Matrix     ──► Rotates portrait front camera images upright    │   │
 │   │  2. In-Memory Downsampling      ──► inSampleSize scales multi-MP photo to ~384–512px│   │
 │   │  3. MediaPipe HandLandmarker    ──► Evaluates hand_landmarker.task on CPU/GPU       │   │
 │   │  4. Native Bitmap Disposal      ──► Bitmap.recycle() prevents GC lag & OOMs         │   │
 │   └─────────────────────────────────────────────────────────────────────────────────────┘   │
 └──────────────────┬──────────────────────────────────────────────────────────────────────────┘
                    │ 63 Raw doubles [x0, y0, z0, ..., x20, y20, z20]
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                         INFERENCE & SPEECH (JavaScript Engine)                              │
 │                                                                                             │
 │   ┌───────────────────────────────┐                  ┌──────────────────────────────────┐   │
 │   │  src/lib/handDetector.ts      │                  │     react-native-fast-tflite     │   │
 │   │  normalizeLandmarks():        ├─────────────────►│     activeModel.runSync():       │   │
 │   │  Δx = xi - x0 (wrist-relative)│ 63 Floats        │     Inference in <1 ms           │   │
 │   └───────────────────────────────┘                  └─────────────────┬────────────────┘   │
 │                                                                        │ Softmax Probs      │
 │                                                                        ▼                    │
 │                                                      ┌──────────────────────────────────┐   │
 │                                                      │      expo-speech (Audio TTS)     │   │
 │                                                      │  "HELLO" • "THANK YOU" • "A"     │   │
 │                                                      └──────────────────────────────────┘   │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Supported Models & Vocabulary

| Model ID | Asset File | Classes | Input Shape | Supported Vocabulary |
|---|---|---|---|---|
| **`STD`** | `sign_language_model.tflite` | 8 | `[1, 63]` | `openPalm`, `closedFist`, `fingerCircle`, `fingerSymbols`, `singleFingerBend`, `multiFingerBend`, `semiOpenPalm`, `semiOpenFist` |
| **`ISL`** | `gesture_model_indian.tflite` | 35 | `[1, 63]` | Letters **A–Z** and Numbers **1–9** (Indian Sign Language single-hand set) |
| **`GESTURE`** | `gesture_model.tflite` | 8 | `[1, 63]` | Universal symbolic interaction gestures |

---

## 🔄 End-to-End Data Pipeline

1. **Optical Capture**:
   VisionCamera captures a frame via `usePhotoOutput` with balanced quality prioritization to ensure universal CameraX hardware stream compatibility.
2. **Native Preprocessing (`ExpoMediapipeHandsModule.kt`)**:
   - `ExifInterface` detects front camera sensor rotation (90° or 270°) and applies a rotation matrix so MediaPipe processes an upright portrait image.
   - `inJustDecodeBounds` calculates an optimal `inSampleSize` to downsample large camera photos directly into a ~384–512 px bitmap, cutting memory consumption and decoding time by over 90%.
3. **MediaPipe Landmark Extraction**:
   Google MediaPipe's `HandLandmarker` extracts **21 3D landmarks** ($x, y, z$) in normalized $[0.0, 1.0]$ coordinate space.
4. **Wrist-Relative Normalization (`handDetector.ts`)**:
   To make the model position-invariant (so a sign is recognized anywhere on screen), Landmark 0 (wrist) is subtracted from all 21 landmarks:
   $$\Delta x_i = x_i - x_0, \quad \Delta y_i = y_i - y_0, \quad \Delta z_i = z_i - z_0$$
   This yields a 63-element `Float32Array` where the wrist is centered at $(0, 0, 0)$.
5. **On-Device Neural Inference (`react-native-fast-tflite`)**:
   The vector is passed to `activeModel.runSync([inputBuffer])`, which evaluates the deep Multi-Layer Perceptron (MLP) in `<1 ms` on the device CPU/GPU.
6. **HUD & Speech Output**:
   - If confidence $\ge 60\%$, the HUD displays the sign and confidence percentage.
   - `expo-speech` synthesizes the word into audible audio.
   - `HandSkeletonOverlay.tsx` projects the coordinates into screen pixels to illuminate joints, bones, and the `HAND LOCKED` target reticle.

---

## 📁 Repository Structure

```
Gestura/
├── GesturaMobile/                 # Primary Expo / React Native Mobile Application
│   ├── assets/
│   │   ├── images/                # Icons, logos, and UI artwork
│   │   └── model/                 # Bundled .tflite models & label mappings
│   │       ├── sign_language_model.tflite
│   │       ├── gesture_model_indian.tflite
│   │       ├── gesture_model.tflite
│   │       ├── labels.json
│   │       ├── labelsIndian.json
│   │       └── gesture_model_labels.json
│   ├── modules/
│   │   └── expo-mediapipe-hands/  # Local Native Android Expo Module
│   │       ├── android/           # Kotlin source, build.gradle, & MediaPipe dependencies
│   │       │   └── src/main/java/expo/modules/mediapipehands/
│   │       │       └── ExpoMediapipeHandsModule.kt
│   │       └── src/index.ts       # TypeScript module definitions
│   ├── src/
│   │   ├── app/                   # Expo Router file-based screens
│   │   │   ├── _layout.tsx        # Bottom tab navigation container
│   │   │   ├── index.tsx          # Home: Live Camera Translator Viewfinder
│   │   │   └── explore.tsx        # Explore: Interactive Sign Language Dictionary
│   │   ├── components/
│   │   │   └── HandSkeletonOverlay.tsx # 21-Joint skeleton & target reticle canvas
│   │   ├── lib/
│   │   │   ├── handDetector.ts    # MediaPipe bridge & wrist-relative normalizer
│   │   │   ├── features.ts        # 132-d spatial feature calculation utilities
│   │   │   └── signEngine.ts      # Multi-frame temporal consensus smoothing engine
│   │   └── screens/
│   │       └── LiveTranslationScreen.tsx # Architectural reference (JNI isolation pattern)
│   ├── android/                   # Native Android project (Gradle, Manifest, Assets)
│   ├── app.json                   # Expo application configuration & plugins
│   └── package.json               # JavaScript dependencies and scripts
│
├── ml_pipeline/                   # Machine Learning Training Pipeline
│   ├── extract_landmarks.py       # Extracts 21 wrist-relative landmarks from image folders
│   ├── train_model.py             # Trains Keras sequential MLP & exports quantized .tflite
│   ├── sign_language_dataset.csv  # Extracted landmark dataset (63 features + label)
│   └── hand_landmarker.task       # MediaPipe HandLandmarker task bundle
│
├── camera_utils.py                # Desktop OpenCV camera & skeleton drawing helpers
├── tracker.py                     # Desktop training clip recorder
├── test_model.py                  # Desktop live model evaluation harness
├── requirements.txt               # Python ML & desktop dependencies
└── Gestura_Technical_Architecture.pdf # Comprehensive technical specification document
```

---

## 🚀 Getting Started

### Prerequisites
* **Node.js**: v18+ (Node v20 or v22 recommended)
* **Java Development Kit (JDK)**: JDK 17 or JDK 20
* **Android SDK**: Build-Tools 35.0.0, Platform SDK 35
* **Python**: 3.10+ (for ML pipeline and desktop testing)
* Physical Android phone with USB debugging enabled (recommended for camera testing)

---

### Running the Mobile App (GesturaMobile)

1. **Navigate to the mobile directory**:
   ```bash
   cd GesturaMobile
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Verify the MediaPipe model asset**:
   Ensure `hand_landmarker.task` is located in `GesturaMobile/android/app/src/main/assets/hand_landmarker.task`. *(Already configured)*

4. **Build and install the development build on your connected Android device**:
   ```bash
   npx expo run:android
   ```

5. **Start the Metro development server**:
   ```bash
   npx expo start
   ```

---

### Running the ML Training Pipeline

To retrain the gesture model on your own custom sign dataset:

1. **Activate Python environment & install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Extract landmarks from image folders**:
   Place labeled image folders under `ml_pipeline/dataset/<class_name>/` and run:
   ```bash
   python ml_pipeline/extract_landmarks.py
   ```

3. **Train the neural network and export TFLite**:
   ```bash
   python ml_pipeline/train_model.py
   ```
   This generates `sign_language_model.tflite` and `labels.json`, ready to be copied into `GesturaMobile/assets/model/`.

---

### Running the Desktop Prototype

To test camera tracking and models directly on your PC with a webcam:

```bash
# Test live sign recognition via webcam
python test_model.py

# Record sign training clips
python tracker.py --person alice
```

---

## ⚡ Engineering & Optimization Highlights

* **95% Latency Reduction via Adaptive Downsampling**:
  By default, VisionCamera captures photos at 4K UHD (`UHD_4_3` ≈ 12 Megapixels), creating massive 5 MB files and multi-megabyte bitmap allocations. In `ExpoMediapipeHandsModule.kt`, `inJustDecodeBounds` inspects dimensions and dynamically downsamples the image to **~384–512 px**, reducing end-to-end processing latency from 800 ms to **~25 ms**.
* **Front Camera EXIF Alignment**:
  Android front-camera sensors record portrait images rotated 270° or 90°. `ExifInterface` detects and rotates bitmaps upright before feeding them to MediaPipe, ensuring consistent landmark detection.
* **CameraX Stream Configuration Protection**:
  Removed forced portrait aspect ratios (`CommonResolutions.VGA_4_3`) in favor of balanced quality prioritization, eliminating `CameraState.ERROR_STREAM_CONFIG` exceptions across diverse Android camera hardware levels.
* **Hardware-Safe Torch**:
  Prevents `CameraAccessException: no flash unit` on front cameras by conditioning `torchMode` strictly on `device.hasTorch`.
* **Zero-Dependency Vector Overlay**:
  `HandSkeletonOverlay.tsx` renders 21 joints and 21 bones with pure React Native `<View>` elements and CSS rotational transforms, avoiding heavyweight graphic engine bloat.

---

## ❓ Troubleshooting & FAQs

#### 1. Live feed shows "No hand detected" even when my hand is in view
* Make sure `hand_landmarker.task` is bundled inside `GesturaMobile/android/app/src/main/assets/`. If you just added it, run `npx expo run:android` to rebuild the APK with the asset included.
* Ensure adequate lighting and hold your hand inside the dashed reticle box.

#### 2. The camera preview feels laggy
* Verify that you are running in **`MANUAL` mode** for smooth 60 FPS previews, or ensure the AUTO interval is set to $\ge 1000\text{ ms}$.
* Ensure the adaptive downsampling code in `ExpoMediapipeHandsModule.kt` is active.

#### 3. Error: `Failed to apply the stream configuration for the given outputs`
* Do not pass custom forced resolutions to `usePhotoOutput`. Use `{ quality: 0.5, qualityPrioritization: 'balanced' }`, which CameraX can safely pair with the preview stream on all Android devices.

---

## 📄 License & Credits
Developed as part of the **Gestura Minor Project**. Built with Google MediaPipe, React Native, and TensorFlow Lite.
