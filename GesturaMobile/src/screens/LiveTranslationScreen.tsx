/**
 * LiveTranslationScreen.tsx
 *
 * Real-time sign-language gesture recognition screen for Gestura.
 *
 * Architecture (to prevent JNI global reference leaks):
 *   The screen is split into two components:
 *   - CameraShell: Holds VisionCamera's `usePhotoOutput` and the Camera
 *     view. Gesture state lives in GestureProcessor so HUD updates do
 *     not recreate camera outputs.
 *   - GestureProcessor: Holds gesture recognition state (gesture,
 *     confidence). Its re-renders are isolated from camera hooks.
 *
 * Pipeline (runs at ~2 FPS via snapshot timer):
 *   1. A timer captures a photo via VisionCamera v5's `usePhotoOutput`
 *      + `capturePhotoToFile()` every ~500ms.
 *   2. The file path is passed to the Expo Module `expo-mediapipe-hands`
 *      which decodes the image and runs MediaPipe HandLandmarker to
 *      extract 21 normalised 3-D landmarks (63 floats).
 *   3. The landmarks are fed into the on-device TFLite gesture classifier
 *      (`sign_language_model.tflite`) via `model.run()`.
 *   4. The argmax of the softmax output is mapped to a label and
 *      displayed in the UI overlay.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  Camera,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';

// Local Expo Module — wraps MediaPipe HandLandmarker (Android).
import ExpoMediapipeHands from 'expo-mediapipe-hands';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LABELS: string[] = ['hello', 'thank_you', 'welcome'];
// eslint-disable-next-line @typescript-eslint/no-require-imports
const MODEL_ASSET: number = require('@/assets/model/sign_language_model.tflite');
const CONFIDENCE_THRESHOLD = 0.60;
const SNAPSHOT_INTERVAL_MS = 500;

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
// HUD Overlay
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
// GestureProcessor — holds all mutable gesture state.
// Re-renders from state changes are isolated here, away from camera hooks.
// Receives photoOutput.capturePhotoToFile as a stable ref callback.
// ---------------------------------------------------------------------------

interface GestureProcessorProps {
  capturePhotoRef: React.RefObject<(() => Promise<{ filePath: string }>) | null>;
}

function GestureProcessor({ capturePhotoRef }: GestureProcessorProps) {
  const tfModel = useTensorflowModel(MODEL_ASSET, []);

  const [gesture, setGesture] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);

  const busyRef = useRef(false);
  const activeRef = useRef(true);

  const processSnapshot = useCallback(async () => {
    if (busyRef.current) return;

    const model = tfModel;
    if (model.state !== 'loaded') {
      console.log('[Gestura] Model not loaded yet, state:', model.state);
      return;
    }

    const capturePhoto = capturePhotoRef.current;
    if (!capturePhoto) {
      console.log('[Gestura] capturePhoto not ready yet');
      return;
    }

    busyRef.current = true;

    try {
      console.log('[Gestura] Capturing photo...');
      const photoFile = await capturePhoto();
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
  }, [tfModel, capturePhotoRef]);

  // Snapshot loop
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

    // Give the camera 3s to initialize before starting the loop.
    timeoutId = setTimeout(loop, 3000);
    return () => {
      activeRef.current = false;
      clearTimeout(timeoutId);
    };
  }, [processSnapshot]);

  return (
    <HUDOverlay
      gesture={gesture}
      confidence={confidence}
      modelState={tfModel.state}
      modelError={tfModel.state === 'error' ? tfModel.error?.message : undefined}
    />
  );
}

// ---------------------------------------------------------------------------
// CameraShell — holds camera hooks, NEVER re-renders from gesture state.
// This prevents the JNI global reference leak in useCameraDevice().
// ---------------------------------------------------------------------------

function CameraShell() {
  const [cameraError, setCameraError] = useState<string | null>(null);

  const photoOutput = usePhotoOutput({
    quality: 0.5,
    qualityPrioritization: 'balanced',
  });

  // Expose capturePhotoToFile via a stable ref so GestureProcessor
  // can call it without causing CameraShell to re-render.
  const capturePhotoRef = useRef<(() => Promise<{ filePath: string }>) | null>(null);
  useEffect(() => {
    capturePhotoRef.current = () =>
      photoOutput.capturePhotoToFile(
        { flashMode: 'off', enableShutterSound: false },
        {},
      );
  }, [photoOutput]);

  if (cameraError != null) {
    return (
      <View style={styles.center}>
        <Text style={styles.statusText}>{cameraError}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device="front"
        isActive={true}
        outputs={[photoOutput]}
        onError={(error) => {
          console.error('[Gestura] Camera error:', error);
          setCameraError(error.message || 'Camera failed to start');
        }}
      />
      <GestureProcessor capturePhotoRef={capturePhotoRef} />
    </View>
  );
}

// ---------------------------------------------------------------------------
// Main Component — only handles permission, delegates to CameraShell
// ---------------------------------------------------------------------------

class ScreenErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error != null) {
      return (
        <View style={styles.center}>
          <Text style={styles.statusText}>
            {this.state.error.message || 'Something went wrong'}
          </Text>
        </View>
      );
    }
    return this.props.children;
  }
}

export default function LiveTranslationScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#6C63FF" />
        <Text style={styles.statusText}>Requesting camera permission…</Text>
      </View>
    );
  }

  return (
    <ScreenErrorBoundary>
      <CameraShell />
    </ScreenErrorBoundary>
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
