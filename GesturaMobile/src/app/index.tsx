import React, { useEffect } from 'react';
import { StyleSheet, Text, View, ActivityIndicator } from 'react-native';
// Make sure this is the ONLY camera import in the file!
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';

// @ts-ignore - This bypasses TypeScript's strict JSON rule
import labels from '../../assets/model/labels.json';

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  
  // @ts-ignore - Bypasses TypeScript's strict hardware delegate rule
  const model = useTensorflowModel(require('../../assets/model/gesture_model.tflite'));

  // Request camera permissions on startup
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission]);

  // Handle loading states
  if (!hasPermission) return <View style={styles.center}><Text>Grant Camera Permission...</Text></View>;
  if (device == null) return <View style={styles.center}><Text>No Camera Found</Text></View>;
  if (model.state === 'loading') return <View style={styles.center}><ActivityIndicator size="large" /></View>;
  if (model.state === 'error') return <View style={styles.center}><Text>Error loading AI Model</Text></View>;

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
      />
      <View style={styles.overlay}>
        <Text style={styles.gestureText}>
          {model.state === 'loaded' ? 'AI Model Loaded!' : 'Waiting...'}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  overlay: {
    position: 'absolute',
    bottom: 50,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  gestureText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 12,
    overflow: 'hidden',
  }
});