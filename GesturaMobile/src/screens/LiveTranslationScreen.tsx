/**
 * LiveTranslationScreen.tsx
 *
 * Real-time sign-language gesture recognition screen for Gestura.
 *
 * Pipeline (runs at ~3 FPS via snapshot timer):
 *   1. A timer captures a photo via VisionCamera v5's `usePhotoOutput`
 *      + `capturePhotoToFile()` every ~350ms.
 *   2. The file path is passed to the Expo Module `expo-mediapipe-hands`
 *      which decodes the image and runs MediaPipe HandLandmarker to
 *      extract 21 normalised 3-D landmarks (63 floats).
 *   3. The landmarks are fed into the on-device TFLite gesture classifier
 *      (`sign_language_model.tflite`) via `model.run()`.
 *   4. The argmax of the softmax output is mapped to a label and
 *      displayed in the UI overlay.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  Camera,
  type CameraRef,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
  type CameraPhotoOutput,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';

// Local Expo Module — wraps MediaPipe HandLandmarker (Android).
import ExpoMediapipeHands from 'expo-mediapipe-hands';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LABELS: string[] = ['hello', 'thank_you', 'welcome'];
const MODEL_ASSET = require('@/assets/model/sign_language_model.tflite');
const CONFIDENCE_THRESHOLD = 0.60;
const SNAPSHOT_INTERVAL_MS = 350;

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

function argmax(arr: ArrayLike<number>): { index: number; value: number } {
  let maxIdx = 0;
  let maxVal = arr[0]!;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i]! > maxVal) {
      maxVal = arr[i]!;
      maxIdx = i;
    }
  }
  return { index: maxIdx, value: maxVal };
}

// ---------------------------------------------------------------------------
// HUD Overlay — separated to prevent Camera re-renders from state changes
// ---------------------------------------------------------------------------

interface HUDProps {
  gesture: string | null;
  confidence: number;
  modelState: 'loading' | 'loaded' | 'error';
  modelError?: string;
}

const HUDOverlay = React.memo(function HUDOverlay({
  gesture,
  confidence,
  modelState,
  modelError,
}: HUDProps) {
  return (
    <View style={styles.overlay} pointerEvents="none">
      {/* Model loading indicator */}
      {modelState === 'loading' && (
        <View style={styles.modelChip}>
          <ActivityIndicator size="small" color="#ffffff" style={{ marginRight: 8 }} />
          <Text style={styles.modelChipText}>Loading model…</Text>
        </View>
      )}
      {modelState === 'error' && (
        <View style={styles.modelChip}>
          <Text style={styles.modelChipText}>Model error: {modelError ?? 'unknown'}</Text>
        </View>
      )}
      {modelState === 'loaded' && <View />}

      {/* Detected gesture badge */}
      <View style={styles.gestureContainer}>
        {gesture != null ? (
          <View style={styles.gestureBadge}>
            <Text style={styles.gestureText}>
              {gesture.replace('_', ' ')}
            </Text>
            <Text style={styles.confidenceText}>{confidence}%</Text>
          </View>
        ) : (
          <View style={styles.gestureBadgeIdle}>
            <Text style={styles.gestureTextIdle}>Show a sign…</Text>
          </View>
        )}
      </View>
    </View>
  );
});

// ---------------------------------------------------------------------------
// Camera View — stable component that doesn't re-render with gesture state
// ---------------------------------------------------------------------------

interface CameraLayerProps {
  cameraRef: React.RefObject<CameraRef | null>;
  outputs: CameraPhotoOutput[];
}

const CameraLayer = React.memo(function CameraLayer({
  cameraRef,
  outputs,
}: CameraLayerProps) {
  const device = useCameraDevice('front');

  if (device == null) {
    return (
      <View style={styles.center}>
        <Text style={styles.statusText}>No front camera found</Text>
      </View>
    );
  }

  return (
    <Camera
      ref={cameraRef}
      style={StyleSheet.absoluteFill}
      device={device}
      isActive={true}
      outputs={outputs}
    />
  );
});

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function LiveTranslationScreen() {
  // ---- Camera Permission --------------------------------------------------
  const { hasPermission, requestPermission } = useCameraPermission();

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  // ---- Camera Ref ---------------------------------------------------------
  const cameraRef = useRef<CameraRef>(null);

  // ---- Photo Output (VisionCamera v5) -------------------------------------
  const photoOutput = usePhotoOutput({
    quality: 0.5,
    qualityPrioritization: 'speed',
  });
  const outputs = useMemo(() => [photoOutput], [photoOutput]);

  // ---- TFLite Gesture Classifier ------------------------------------------
  const tfModel = useTensorflowModel(MODEL_ASSET, []);

  // ---- Recognised Gesture State -------------------------------------------
  const [gesture, setGesture] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);

  // ---- Refs for the processing loop (avoid re-renders) --------------------
  const busyRef = useRef(false);
  const activeRef = useRef(true);
  const tfModelRef = useRef(tfModel);
  tfModelRef.current = tfModel;

  // ---- Snapshot → MediaPipe → TFLite pipeline -----------------------------
  const processSnapshot = useCallback(async () => {
    if (busyRef.current) return;

    const model = tfModelRef.current;
    if (model.state !== 'loaded') {
      console.log('[Gestura] Model not loaded yet, state:', model.state);
      return;
    }

    busyRef.current = true;

    try {
      console.log('[Gestura] Capturing photo...');
      const photoFile = await photoOutput.capturePhotoToFile(
        { flashMode: 'off', enableShutterSound: false },
        {},
      );
      console.log('[Gestura] Photo captured:', photoFile.filePath);

      const landmarks: number[] =
        await ExpoMediapipeHands.detectLandmarksFromFile(photoFile.filePath);
      console.log('[Gestura] Landmarks:', landmarks.length, 'values');

      if (!landmarks || landmarks.length !== 63) {
        setGesture(null);
        setConfidence(0);
        return;
      }

      const inputBuffer = new Float32Array(63);
      for (let i = 0; i < 63; i++) inputBuffer[i] = landmarks[i]!;

      const result = await model.model.run([inputBuffer.buffer]);
      const probabilities = new Float32Array(result[0]!);
      const { index, value: maxConf } = argmax(probabilities);
      console.log('[Gestura] Prediction:', LABELS[index], 'conf:', maxConf);

      if (maxConf >= CONFIDENCE_THRESHOLD) {
        setGesture(LABELS[index] ?? 'Unknown');
        setConfidence(Math.round(maxConf * 100));
      } else {
        setGesture(null);
        setConfidence(Math.round(maxConf * 100));
      }
    } catch (e) {
      console.error('[Gestura] Pipeline error:', e);
    } finally {
      busyRef.current = false;
    }
  }, [photoOutput]);

  // ---- Snapshot loop ------------------------------------------------------
  useEffect(() => {
    activeRef.current = true;
    let timeoutId: ReturnType<typeof setTimeout>;

    const loop = () => {
      if (!activeRef.current) return;
      processSnapshot().finally(() => {
        if (activeRef.current) {
          timeoutId = setTimeout(loop, SNAPSHOT_INTERVAL_MS);
        }
      });
    };

    timeoutId = setTimeout(loop, 2000);
    return () => {
      activeRef.current = false;
      clearTimeout(timeoutId);
    };
  }, [processSnapshot]);

  // ---- Render Guards ------------------------------------------------------
  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#6C63FF" />
        <Text style={styles.statusText}>Requesting camera permission…</Text>
      </View>
    );
  }

  // ---- Main Render --------------------------------------------------------
  return (
    <View style={styles.container}>
      <CameraLayer cameraRef={cameraRef} outputs={outputs} />
      <HUDOverlay
        gesture={gesture}
        confidence={confidence}
        modelState={tfModel.state}
        modelError={tfModel.state === 'error' ? tfModel.error?.message : undefined}
      />
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    gap: 16,
  },
  statusText: {
    color: '#aaa',
    fontSize: 16,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  overlay: {
    ...(StyleSheet.absoluteFill as object),
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 50,
  },
  modelChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  modelChipText: { color: '#fff', fontSize: 13, fontWeight: '500' },
  gestureContainer: { alignItems: 'center' },
  gestureBadge: {
    alignItems: 'center',
    backgroundColor: 'rgba(108, 99, 255, 0.85)',
    paddingHorizontal: 28,
    paddingVertical: 14,
    borderRadius: 18,
    gap: 4,
  },
  gestureText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    textTransform: 'capitalize',
  },
  confidenceText: {
    fontSize: 14,
    fontWeight: '600',
    color: 'rgba(255,255,255,0.8)',
  },
  gestureBadgeIdle: {
    backgroundColor: 'rgba(0,0,0,0.45)',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 16,
  },
  gestureTextIdle: {
    fontSize: 22,
    fontWeight: '600',
    color: 'rgba(255,255,255,0.6)',
  },
});
