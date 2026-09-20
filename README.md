# 🤟 Gestura: Two-Way Real-Time Sign Language Interpreter

> An accessible, cross-platform mobile application enabling seamless bidirectional communication between the Deaf/Hard-of-Hearing (DHH) community and hearing non-signers.

---

## 📌 Overview

**BridgeSign** bridges communication barriers by turning a standard smartphone into an accessible, real-time two-way interpreter. It operates via two distinct processing pipelines:

1. **Sign-to-Speech (S2S):** Uses device cameras and **Google MediaPipe** to extract 3D hand/pose keypoints, classifies gestures using an on-device **TensorFlow Lite** sequence model, and outputs spoken audio via native TTS.
2. **Speech-to-Sign (S2S):** Captures spoken voice, converts it into text, reorders English grammar into Sign Language Gloss via NLP, and animates a rigged **3D humanoid avatar** using **Three.js / React Three Fiber**.

---

## ✨ Key Features

- ⚡ **On-Device Inference:** Local ML execution ensures ultra-low latency (< 400 ms) and zero reliance on cloud servers.
- 🔒 **Privacy-First:** Video and audio streams are processed in-memory directly on the device with no external transmission.
- 🤖 **Dynamic 3D Avatar:** Smooth animation blending via Three.js `AnimationMixer` for natural sign transitions.
- 📱 **Cross-Platform:** Built on React Native for seamless deployment across iOS and Android.

---

## 🛠️ Tech Stack

- **Framework:** React Native (New Architecture / JSI Bridge)
- **Computer Vision:** Google MediaPipe (Hands & Pose Tracking)
- **Machine Learning:** TensorFlow Lite (`react-native-fast-tflite`)
- **3D Graphics:** Three.js, React Three Fiber, Expo-GL
- **Audio & Speech:** `@react-native-voice/voice`, `react-native-tts`