/**
 * expo-mediapipe-hands — local Expo Module
 *
 * Wraps Google MediaPipe HandLandmarker for Android.
 * Exposes an async function that accepts an image file path
 * and returns the 21 normalised 3-D hand landmarks (63 doubles).
 */
import { requireNativeModule } from 'expo-modules-core';

interface ExpoMediapipeHandsModule {
  /**
   * Detect hand landmarks from an image file on-device.
   *
   * @param filePath  Absolute path to a JPEG/PNG image.
   * @returns A promise resolving to an array of 63 doubles
   *          (x0, y0, z0, …, x20, y20, z20), or an empty array if no hand found.
   */
  detectLandmarksFromFile(filePath: string): Promise<number[]>;
}

const mod = requireNativeModule<ExpoMediapipeHandsModule>('ExpoMediapipeHands');

export default mod;
