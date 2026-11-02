import * as Speech from 'expo-speech';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  LayoutChangeEvent,
  Pressable,
  Share,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';

import { HandSkeletonOverlay } from '@/components/HandSkeletonOverlay';
import {
  detectLandmarksFromPhoto,
  normalizeLandmarks,
  HAND_DETECTOR_CONNECTED,
} from '@/lib/handDetector';

// Static model asset imports
// eslint-disable-next-line @typescript-eslint/no-require-imports
const MODEL_STD_ASSET = require('../../assets/model/sign_language_model.tflite');
// eslint-disable-next-line @typescript-eslint/no-require-imports
const MODEL_ISL_ASSET = require('../../assets/model/gesture_model_indian.tflite');
// eslint-disable-next-line @typescript-eslint/no-require-imports
const MODEL_GESTURE_ASSET = require('../../assets/model/gesture_model.tflite');

// Labels
const LABELS_STD: string[] = require('../../assets/model/labels.json');
const LABELS_ISL: string[] = require('../../assets/model/labelsIndian.json');
const LABELS_GESTURE: string[] = require('../../assets/model/gesture_model_labels.json');

type ActiveModelType = 'STD' | 'ISL' | 'GESTURE';
type CameraPosition = 'front' | 'back';

const SNAPSHOT_INTERVAL_MS = 1000;
const CONFIDENCE_THRESHOLD = 0.60;

export default function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();

  // Camera Device State (Front / Back flip)
  const [cameraPosition, setCameraPosition] = useState<CameraPosition>('front');
  const [torchOn, setTorchOn] = useState(false);
  const device = useCameraDevice(cameraPosition);

  // Layout dimensions of the camera viewfinder for skeleton overlay
  const [viewDim, setViewDim] = useState({ width: 0, height: 0 });

  // Raw 63 landmarks for Skeleton Overlay
  const [rawLandmarks, setRawLandmarks] = useState<number[] | null>(null);

  // Active Model State
  const [modelType, setModelType] = useState<ActiveModelType>('STD');

  // TFLite Models
  const stdPlugin = useTensorflowModel(MODEL_STD_ASSET, []);
  const islPlugin = useTensorflowModel(MODEL_ISL_ASSET, []);
  const gesturePlugin = useTensorflowModel(MODEL_GESTURE_ASSET, []);

  const activeModel = useMemo(() => {
    if (modelType === 'STD' && stdPlugin.state === 'loaded') return stdPlugin.model;
    if (modelType === 'ISL' && islPlugin.state === 'loaded') return islPlugin.model;
    if (modelType === 'GESTURE' && gesturePlugin.state === 'loaded') return gesturePlugin.model;
    return null;
  }, [modelType, stdPlugin, islPlugin, gesturePlugin]);

  const activePluginState = useMemo(() => {
    if (modelType === 'STD') return stdPlugin.state;
    if (modelType === 'ISL') return islPlugin.state;
    return gesturePlugin.state;
  }, [modelType, stdPlugin, islPlugin, gesturePlugin]);

  const activeLabels = useMemo(() => {
    if (modelType === 'STD') return LABELS_STD;
    if (modelType === 'ISL') return LABELS_ISL;
    return LABELS_GESTURE;
  }, [modelType]);

  const [word, setWord] = useState('No hand detected');
  const [transcript, setTranscript] = useState<string[]>([]);
  const [top, setTop] = useState<{ label: string; p: number }[]>([]);
  const [mode, setMode] = useState<'AUTO' | 'MANUAL'>('MANUAL');
  const [capturing, setCapturing] = useState(false);
  const [speechOn, setSpeechOn] = useState(true);
  const [copiedNotice, setCopiedNotice] = useState(false);

  const modeRef = useRef(mode);
  const speechRef = useRef(speechOn);
  const busyRef = useRef(false);
  const lastSpokenRef = useRef<{ text: string; time: number }>({ text: '', time: 0 });

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    speechRef.current = speechOn;
  }, [speechOn]);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  // Handle Viewport Layout Dimensions
  const onLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setViewDim({ width, height });
  }, []);

  // Flip Camera Front <-> Back
  const flipCamera = useCallback(() => {
    setCameraPosition((pos) => (pos === 'front' ? 'back' : 'front'));
    setTorchOn(false);
    setRawLandmarks(null);
  }, []);

  // Cycle through models
  const cycleModel = useCallback(() => {
    setTop([]);
    setWord('Model switched');
    setRawLandmarks(null);
    setModelType((prev) => {
      if (prev === 'STD') return 'ISL';
      if (prev === 'ISL') return 'GESTURE';
      return 'STD';
    });
  }, []);

  // Safe, device-compatible photo output configuration
  // Uses balanced prioritization to ensure CameraX stream compatibility on all Android devices
  const photoOutput = usePhotoOutput({
    quality: 0.5,
    qualityPrioritization: 'balanced',
  });

  const capturePhotoRef = useRef<(() => Promise<{ filePath: string }>) | null>(null);
  useEffect(() => {
    capturePhotoRef.current = () =>
      photoOutput.capturePhotoToFile(
        { flashMode: 'off', enableShutterSound: false },
        {},
      );
  }, [photoOutput]);

  // Classification logic from photo file
  const classifyPhoto = useCallback(
    async (filePath: string) => {
      if (!activeModel) return;

      const landmarks = await detectLandmarksFromPhoto(filePath);

      if (!landmarks || landmarks.length !== 63) {
        setRawLandmarks(null);
        if (modeRef.current === 'AUTO') {
          setWord('No hand detected');
          setTop([]);
        }
        return;
      }

      // Update landmarks for skeleton visual overlay
      setRawLandmarks(landmarks);

      // Normalize coordinates relative to wrist (Landmark 0)
      const inputBuffer = normalizeLandmarks(landmarks);

      const rawBuffer =
        inputBuffer.byteOffset === 0 && inputBuffer.byteLength === inputBuffer.buffer.byteLength
          ? inputBuffer.buffer
          : inputBuffer.buffer.slice(
              inputBuffer.byteOffset,
              inputBuffer.byteOffset + inputBuffer.byteLength,
            );

      const output = activeModel.runSync([rawBuffer as ArrayBuffer])[0];
      const probs = new Float32Array(output);

      // Compute Top 3 Predictions
      const ranked = Array.from(probs, (p, i) => ({
        label: activeLabels[i] || `Class ${i}`,
        p,
      }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3);

      setTop(ranked);

      const best = ranked[0];
      if (best && best.p >= CONFIDENCE_THRESHOLD) {
        const text = best.label.replace(/_/g, ' ');
        setWord(`${text.toUpperCase()} (${Math.round(best.p * 100)}%)`);

        // Speak and append with debounce
        const now = Date.now();
        if (
          speechRef.current &&
          (lastSpokenRef.current.text !== text || now - lastSpokenRef.current.time > 2000)
        ) {
          lastSpokenRef.current = { text, time: now };
          Speech.speak(text);
          setTranscript((t) => [...t.slice(-9), text]);
        }
      } else if (best) {
        const text = best.label.replace(/_/g, ' ');
        setWord(`${text} (${Math.round(best.p * 100)}%)`);
      }
    },
    [activeModel, activeLabels],
  );

  // Snapshot processing loop for AUTO mode
  const processSnapshot = useCallback(async () => {
    if (busyRef.current || modeRef.current !== 'AUTO') return;
    if (!capturePhotoRef.current || !activeModel) return;

    busyRef.current = true;
    try {
      const photoFile = await capturePhotoRef.current();
      if (photoFile?.filePath) {
        await classifyPhoto(photoFile.filePath);
      }
    } catch (err) {
      console.log('[Gestura Snapshot Error]:', err);
    } finally {
      busyRef.current = false;
    }
  }, [activeModel, classifyPhoto]);

  useEffect(() => {
    if (mode !== 'AUTO') return;

    let isActive = true;
    let timer: ReturnType<typeof setTimeout>;

    const loop = () => {
      if (!isActive) return;
      processSnapshot().finally(() => {
        if (isActive) {
          timer = setTimeout(loop, SNAPSHOT_INTERVAL_MS);
        }
      });
    };

    timer = setTimeout(loop, 1200);

    return () => {
      isActive = false;
      clearTimeout(timer);
    };
  }, [mode, processSnapshot]);

  // MANUAL Mode: Single trigger capture
  const signNow = useCallback(async () => {
    if (capturing || !capturePhotoRef.current || !activeModel) return;

    setCapturing(true);
    setWord('CAPTURING SIGN...');

    try {
      const photoFile = await capturePhotoRef.current();
      if (photoFile?.filePath) {
        await classifyPhoto(photoFile.filePath);
      }
    } catch {
      setWord('Capture failed - try again');
    } finally {
      setCapturing(false);
    }
  }, [capturing, activeModel, classifyPhoto]);

  // Transcript Controls
  const undoLastWord = useCallback(() => {
    setTranscript((t) => t.slice(0, -1));
  }, []);

  const speakTranscript = useCallback(() => {
    if (transcript.length === 0) return;
    Speech.speak(transcript.join(' '));
  }, [transcript]);

  const copyOrShareTranscript = useCallback(async () => {
    if (transcript.length === 0) {
      Alert.alert('Transcript empty', 'Sign gestures first to build a sentence.');
      return;
    }
    const fullText = transcript.join(' ');
    try {
      await Share.share({
        message: fullText,
        title: 'Gestura Translation',
      });
      setCopiedNotice(true);
      setTimeout(() => setCopiedNotice(false), 2000);
    } catch {
      // Ignore dismiss
    }
  }, [transcript]);

  if (!hasPermission) return <Text style={styles.msg}>Camera permission needed</Text>;
  if (!device) return <Text style={styles.msg}>No camera device found</Text>;

  const hasTorch = Boolean(device.hasTorch);

  return (
    <View style={styles.root} onLayout={onLayout}>
      {/* VisionCamera Viewport */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        outputs={[photoOutput]}
        torchMode={device?.hasTorch && torchOn ? 'on' : undefined}
        orientationSource="device"
      />

      {/* Real-time Hand Skeleton & Framing Reticle Overlay */}
      {viewDim.width > 0 && viewDim.height > 0 && (
        <HandSkeletonOverlay
          landmarks={rawLandmarks}
          width={viewDim.width}
          height={viewDim.height}
          mirrorX={cameraPosition === 'front'}
          handDetected={rawLandmarks !== null && rawLandmarks.length === 63}
        />
      )}

      {/* Top Header Floating Controls */}
      <View style={styles.topBar}>
        <View style={styles.topRow}>
          {/* Flip Camera Button */}
          <Pressable style={styles.iconBtn} onPress={flipCamera}>
            <Text style={styles.iconBtnText}>🔄 {cameraPosition.toUpperCase()}</Text>
          </Pressable>

          {/* Torch Toggle (only when back camera has torch) */}
          {hasTorch && (
            <Pressable
              style={[styles.iconBtn, torchOn && styles.iconBtnActive]}
              onPress={() => setTorchOn((t) => !t)}
            >
              <Text style={styles.iconBtnText}>🔦 {torchOn ? 'ON' : 'OFF'}</Text>
            </Pressable>
          )}

          {/* Model Switcher Button */}
          <Pressable style={[styles.iconBtn, styles.modelBtn]} onPress={cycleModel}>
            <Text style={styles.iconBtnText}>🤖 {modelType}</Text>
          </Pressable>

          {/* Speech TTS Toggle */}
          <Pressable
            style={[styles.iconBtn, speechOn && styles.iconBtnActive]}
            onPress={() => setSpeechOn((s) => !s)}
          >
            <Text style={styles.iconBtnText}>🔊 {speechOn ? 'ON' : 'MUTE'}</Text>
          </Pressable>
        </View>

        {!HAND_DETECTOR_CONNECTED && (
          <Text style={styles.warn}>Hand detector not connected</Text>
        )}
        {activePluginState === 'loading' && (
          <Text style={styles.warn}>Loading {modelType} model...</Text>
        )}
      </View>

      {/* Main HUD Predictions */}
      <View style={styles.hudContainer}>
        <Text style={styles.hudStatus}>
          {mode} MODE | {modelType}
        </Text>
        {top.map((t) => (
          <Text key={t.label} style={styles.hudProb}>
            {t.label}: {Math.round(t.p * 100)}%
          </Text>
        ))}
        {word !== '' && <Text style={styles.hudWord}>{word}</Text>}
      </View>

      {/* Bottom Transcript & Interaction Panel */}
      <View style={styles.bottomPanel}>
        {/* Transcript Text Box */}
        <View style={styles.transcriptCard}>
          <View style={styles.transcriptHeader}>
            <Text style={styles.transcriptTitle}>TRANSCRIPT</Text>
            {copiedNotice && <Text style={styles.copiedBadge}>Copied!</Text>}
          </View>
          <Text style={styles.transcriptText} numberOfLines={2}>
            {transcript.length > 0 ? transcript.join(' ') : 'Signs will appear here as you translate...'}
          </Text>

          {/* Transcript Action Row */}
          {transcript.length > 0 && (
            <View style={styles.transcriptActions}>
              <Pressable style={styles.actionPill} onPress={speakTranscript}>
                <Text style={styles.actionPillText}>▶ Speak All</Text>
              </Pressable>
              <Pressable style={styles.actionPill} onPress={copyOrShareTranscript}>
                <Text style={styles.actionPillText}>📤 Share / Copy</Text>
              </Pressable>
              <Pressable style={styles.actionPill} onPress={undoLastWord}>
                <Text style={styles.actionPillText}>⌫ Undo</Text>
              </Pressable>
              <Pressable style={styles.actionPill} onPress={() => setTranscript([])}>
                <Text style={styles.actionPillText}>✕ Clear</Text>
              </Pressable>
            </View>
          )}
        </View>

        {/* Bottom Mode & Action Buttons */}
        <View style={styles.bottomControlsRow}>
          <Pressable
            style={styles.modeToggleBtn}
            onPress={() => setMode((m) => (m === 'AUTO' ? 'MANUAL' : 'AUTO'))}
          >
            <Text style={styles.modeToggleText}>
              {mode === 'AUTO' ? 'Switch to MANUAL' : 'Switch to AUTO'}
            </Text>
          </Pressable>

          {mode === 'MANUAL' && (
            <Pressable
              style={[styles.signNowBtn, capturing && styles.signNowCapturing]}
              onPress={signNow}
              disabled={capturing}
            >
              <Text style={styles.signNowText}>
                {capturing ? 'RECORDING...' : '✋ SIGN NOW'}
              </Text>
            </Pressable>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  msg: { color: '#fff', textAlign: 'center', marginTop: 100, fontSize: 16 },
  warn: {
    color: '#ffdd00',
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    fontSize: 12,
    alignSelf: 'flex-start',
    marginTop: 6,
  },
  topBar: {
    position: 'absolute',
    top: 50,
    left: 16,
    right: 16,
    zIndex: 10,
  },
  topRow: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },
  iconBtn: {
    backgroundColor: 'rgba(18, 20, 29, 0.85)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.15)',
  },
  iconBtnActive: {
    backgroundColor: '#0055FF',
    borderColor: '#3388FF',
  },
  modelBtn: {
    backgroundColor: 'rgba(0, 100, 255, 0.4)',
    borderColor: '#0066FF',
  },
  iconBtnText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 12,
  },
  hudContainer: {
    position: 'absolute',
    top: 115,
    left: 18,
    right: 18,
    zIndex: 5,
  },
  hudStatus: {
    color: '#8A8F9E',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1.2,
    textShadowColor: '#000',
    textShadowRadius: 3,
  },
  hudProb: {
    color: '#E0E0E0',
    fontSize: 13,
    fontWeight: '600',
    marginTop: 2,
    textShadowColor: '#000',
    textShadowRadius: 3,
  },
  hudWord: {
    color: '#00FF66',
    fontSize: 32,
    fontWeight: '800',
    marginTop: 6,
    letterSpacing: -0.5,
    textShadowColor: 'rgba(0, 0, 0, 0.85)',
    textShadowRadius: 6,
  },
  bottomPanel: {
    position: 'absolute',
    bottom: 24,
    left: 16,
    right: 16,
    zIndex: 10,
    gap: 10,
  },
  transcriptCard: {
    backgroundColor: 'rgba(15, 17, 26, 0.92)',
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  transcriptHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  transcriptTitle: {
    color: '#8A8F9E',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1,
  },
  copiedBadge: {
    color: '#00FFCC',
    fontSize: 11,
    fontWeight: '700',
  },
  transcriptText: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '600',
    lineHeight: 22,
    minHeight: 24,
  },
  transcriptActions: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 10,
    flexWrap: 'wrap',
  },
  actionPill: {
    backgroundColor: '#252837',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
  },
  actionPillText: {
    color: '#C0C5D6',
    fontSize: 11,
    fontWeight: '700',
  },
  bottomControlsRow: {
    flexDirection: 'row',
    gap: 10,
  },
  modeToggleBtn: {
    flex: 1,
    backgroundColor: 'rgba(24, 27, 39, 0.9)',
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.15)',
  },
  modeToggleText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 13,
  },
  signNowBtn: {
    flex: 1.3,
    backgroundColor: '#FF2255',
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#FF2255',
    shadowOpacity: 0.5,
    shadowRadius: 8,
  },
  signNowCapturing: {
    backgroundColor: '#991133',
  },
  signNowText: {
    color: '#FFFFFF',
    fontWeight: '800',
    fontSize: 14,
    letterSpacing: 0.8,
  },
});
