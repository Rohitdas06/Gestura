/**
 * expo-mediapipe-hands — local Expo Module
 *
 * Wraps Google MediaPipe HandLandmarker for Android.
 * Uses requireOptionalNativeModule so a missing or failed native
 * module cannot crash the JS bundle on launch.
 */
import { requireOptionalNativeModule } from 'expo-modules-core';

interface ExpoMediapipeHandsModule {
  detectLandmarksFromFile(filePath: string): Promise<number[]>;
}

const native = requireOptionalNativeModule<ExpoMediapipeHandsModule>('ExpoMediapipeHands');

const ExpoMediapipeHands: ExpoMediapipeHandsModule = native ?? {
  async detectLandmarksFromFile(_filePath: string): Promise<number[]> {
    return [];
  },
};

export default ExpoMediapipeHands;
