import { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, Button, TouchableOpacity } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as handpose from '@tensorflow-models/handpose';

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isAiReady, setIsAiReady] = useState(false);
  const [handModel, setHandModel] = useState<handpose.HandPose | null>(null);
  const [translation, setTranslation] = useState("Press Start to begin...");
  
  // States and refs to safely control the loop
  const [isTranslating, setIsTranslating] = useState(false);
  const isTranslatingRef = useRef(false); 
  
  const cameraRef = useRef<CameraView>(null);

  useEffect(() => {
    const wakeUpBrain = async () => {
      try {
        console.log("Starting TensorFlow Engine...");
        await tf.ready();
        console.log("Loading Hand Tracking Model...");
        const tracker = await handpose.load();
        setHandModel(tracker);
        console.log("Hand Tracker Loaded!");
        setIsAiReady(true);
      } catch (error) {
        console.error("CRITICAL AI ERROR: ", error);
      }
    };
    wakeUpBrain();
  }, []);

  const toggleTranslation = () => {
    const newState = !isTranslating;
    setIsTranslating(newState);
    isTranslatingRef.current = newState;
    
    if (newState) {
      setTranslation("Waiting for hand...");
      scanFrame(); 
    } else {
      setTranslation("Paused.");
    }
  };

  const scanFrame = async () => {
    // 1. Break out of the loop if paused
    if (!cameraRef.current || !handModel || !isTranslatingRef.current) return;

    try {
      console.log("1. Snapping photo...");
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.1,
        base64: true,
        skipProcessing: true,
      });

      // CRITICAL FIX: Pause for 50ms to unfreeze the UI so the Stop button works
      await new Promise(resolve => setTimeout(resolve, 50));

      if (photo && photo.base64 && isTranslatingRef.current) {
        console.log("2. Converting image text to AI Tensor...");
        
        const uint8Array = tf.util.encodeString(photo.base64, 'base64');
        const imageTensor = decodeJpeg(uint8Array);

        // Yield the thread again before the heavy AI math
        await new Promise(resolve => setTimeout(resolve, 50));

        if (isTranslatingRef.current) {
          console.log("3. Scanning for hands...");
          const predictions = await handModel.estimateHands(imageTensor);
          console.log("4. Hands found: ", predictions.length);

          if (predictions && predictions.length > 0) {
            setTranslation("Hand Detected! 🖐");
          } else {
            setTranslation("Waiting for hand...");
          }
        }

        // Memory cleanup
        tf.dispose([imageTensor]);
      }
    } catch (error) {
      console.log("SCANNING ERROR: ", error);
    }

    // Schedule the next frame only if still translating
    if (isTranslatingRef.current) {
      setTimeout(scanFrame, 500); 
    }
  };

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Gestura needs camera access to translate sign language.</Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  if (!isAiReady) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Waking up the AI Brain...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        style={styles.camera}
        facing="front"
        ref={cameraRef}
      />
      
      <View style={styles.overlay}>
        <Text style={styles.titleText}>Gestura AI</Text>
        <Text style={styles.translationText}>{translation}</Text>
        
        <TouchableOpacity 
          style={[styles.button, isTranslating ? styles.buttonStop : styles.buttonStart]} 
          onPress={toggleTranslation}
        >
          <Text style={styles.buttonText}>
            {isTranslating ? "Stop Translating" : "Start Translating"}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', backgroundColor: '#000' },
  message: { textAlign: 'center', paddingBottom: 20, color: 'white', fontSize: 18 },
  camera: { flex: 1 },
  overlay: { position: 'absolute', bottom: 40, width: '100%', alignItems: 'center', zIndex: 10 },
  titleText: { color: '#00FF00', fontSize: 20, fontWeight: 'bold', marginBottom: 5, textShadowColor: 'black', textShadowRadius: 5 },
  translationText: { color: 'white', fontSize: 24, fontWeight: 'bold', backgroundColor: 'rgba(0, 0, 0, 0.6)', paddingHorizontal: 20, paddingVertical: 10, borderRadius: 15, overflow: 'hidden', marginBottom: 20 },
  button: { paddingHorizontal: 30, paddingVertical: 15, borderRadius: 25, elevation: 5 },
  buttonStart: { backgroundColor: '#007AFF' },
  buttonStop: { backgroundColor: '#FF3B30' },
  buttonText: { color: 'white', fontSize: 18, fontWeight: 'bold' }
});