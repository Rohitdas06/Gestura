import ExpoMediapipeHands from '../../modules/expo-mediapipe-hands';

export const HAND_DETECTOR_CONNECTED = true;

let hasLoggedInit = false;

/**
 * Detects 21 3D hand landmarks from a captured photo file.
 * Returns 63 normalized coordinates [x0, y0, z0, ..., x20, y20, z20] or null if no hand detected.
 */
export async function detectLandmarksFromPhoto(filePath: string): Promise<number[] | null> {
  if (!filePath) return null;

  try {
    const mod: any = (ExpoMediapipeHands as any)?.default || ExpoMediapipeHands;

    if (!hasLoggedInit) {
      console.log('[Gestura MediaPipe Init]:', {
        type: typeof mod,
        hasDetectFromFile: typeof mod?.detectLandmarksFromFile === 'function',
      });
      hasLoggedInit = true;
    }

    if (typeof mod?.detectLandmarksFromFile !== 'function') {
      console.warn('[Gestura]: detectLandmarksFromFile is not available on native module');
      return null;
    }

    const landmarks: number[] = await mod.detectLandmarksFromFile(filePath);

    if (landmarks && landmarks.length === 63) {
      return landmarks;
    }
    return null;
  } catch (error) {
    console.error('[Gestura HandDetector Error]:', error);
    return null;
  }
}

/**
 * Normalizes 63 landmarks relative to wrist (Landmark 0) as expected by the TFLite models.
 * Output: Float32Array(63) where landmark 0 is [0, 0, 0] and others are relative offsets.
 */
export function normalizeLandmarks(raw63: number[]): Float32Array {
  const out = new Float32Array(63);
  if (!raw63 || raw63.length < 63) return out;

  const wristX = raw63[0];
  const wristY = raw63[1];
  const wristZ = raw63[2];

  for (let i = 0; i < 21; i++) {
    out[i * 3] = raw63[i * 3] - wristX;
    out[i * 3 + 1] = raw63[i * 3 + 1] - wristY;
    out[i * 3 + 2] = raw63[i * 3 + 2] - wristZ;
  }

  return out;
}
