# 🤟 Gestura — Real-Time Sign Language Interpreter

Webcam → MediaPipe hand tracking → on-device TensorFlow Lite sequence model → text + speech.

**Supported signs:** hello · thank you · welcome (+ a `none` class so random movement is ignored)

## How it works
1. **MediaPipe** extracts 21 3D landmarks per hand (video mode, mirrored view).
2. **features.py** converts each frame to a 132-value vector: hand shape (position/scale-invariant), wrist position and hand size for up to two hands. A sign is a 1.5 s clip resampled to 30 frames, plus frame-to-frame velocity.
3. **1D-CNN** (TFLite built-ins only, <1 MB) classifies the clip.
4. **Smoothing**: rolling window, EMA over probabilities, confidence threshold, N consistent frames, cooldown → speech via TTS.

## Run
```bash
pip install -r requirements.txt
python tracker.py --person alice        # record clips (SPACE / a / n / u / q)
python train_model.py                   # trains, evaluates on unseen sessions, exports .tflite
python test_model.py                    # live demo (m = manual mode, SPACE = capture)
```
`hand_landmarker.task` (MediaPipe model) must sit next to the scripts.

## Evaluation
`train_model.py` splits by recording session/person so test clips are never near-duplicates of training clips. It writes `reports/confusion_matrix.png`, `reports/training_curves.png` and `reports/metrics.json`.

## Roadmap
- Port the exported `.tflite` to React Native (`react-native-fast-tflite`)
- Speech → Sign (gloss + 3D avatar)
- More vocabulary and signers
