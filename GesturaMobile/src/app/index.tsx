import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';

// Import your custom model assets
const customModelJson = require('../../assets/model/model.json'); 
const customModelWeights = require('../../assets/model/group1-shard1of1.bin');
const customModelLabels = require('../../assets/model/labels.json');

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isTranslating, setIsTranslating] = useState(false);
  const [trackerModel, setTrackerModel] = useState<handpose.HandPose | null>(null);
  const [signModel, setSignModel] = useState<tf.LayersModel | null>(null);
  const [translation, setTranslation] = useState("Waiting for sign...");
  
  const cameraRef = useRef<any>(null);
  const isTranslatingRef = useRef(isTranslating); 
  const isProcessingFrame = useRef(false);

  useEffect(() => {
    isTranslatingRef.current = isTranslating;
  }, [isTranslating]);

  useEffect(() => {
    wakeUpBrain();
  }, []);

  const wakeUpBrain = async () => {
    try {
      await tf.ready();
      console.log("TensorFlow ready!");

      const tracker = await handpose.load();
      setTrackerModel(tracker);
      console.log("Hand Tracker Loaded!");

      const loadedSignModel = await tf.loadLayersModel(
        bundleResourceIO(customModelJson, customModelWeights)
      );
      setSignModel(loadedSignModel);
      console.log("Sign Language Model Loaded!");

    } catch (error) {
      console.error("Error waking up brains:", error);
    }
  };

  const startScanning = async () => {
    // FORCE YIELD: Give the UI thread a full 1-second breather to register clicks
    await new Promise(resolve => setTimeout(resolve, 1000));

    if (!cameraRef.current || !trackerModel || !signModel || !isTranslatingRef.current) return;
    if (isProcessingFrame.current) return; 

    isProcessingFrame.current = true;
    
    // Declare tensors out here so they can ALWAYS be cleaned up, even on failure
    let imageTensor: tf.Tensor3D | null = null;
    let inputTensor: tf.Tensor2D | null = null;
    let predictionTensor: tf.Tensor | null = null;

    try {
      console.log("📸 Snapping photo...");
      // Removed skipProcessing to prevent Android hardware crashes
      const photo = await cameraRef.current.takePictureAsync({ 
        base64: true, 
        quality: 0.1, 
        shutterSound: false 
      });
      
      if (!isTranslatingRef.current) return;

      console.log("🧠 Processing AI...");
      const imgBuffer = tf.util.encodeString(photo.base64, 'base64').buffer;
      const raw = new Uint8Array(imgBuffer);
      imageTensor = decodeJpeg(raw);

      const predictions = await trackerModel.estimateHands(imageTensor);
      
      if (predictions.length > 0) {
        const landmarks = predictions[0].landmarks;
        const flatLandmarks = landmarks.flat();
        inputTensor = tf.tensor2d([flatLandmarks]);
        
        predictionTensor = signModel.predict(inputTensor) as tf.Tensor;
        const predictionArray = predictionTensor.dataSync();
        
        const maxProbabilityIndex = predictionArray.indexOf(Math.max(...predictionArray));
        const predictedWord = customModelLabels[maxProbabilityIndex];
        
        setTranslation(predictedWord);
        console.log("✅ Sign detected:", predictedWord);
      } else {
        console.log("❌ No hands detected in this frame.");
      }

    } catch (error) {
      console.error("Scanning Error:", error);
    } finally {
      isProcessingFrame.current = false;
      
      // GUARANTEED MEMORY CLEANUP: Prevents the app from freezing
      if (imageTensor) imageTensor.dispose();
      if (inputTensor) inputTensor.dispose();
      if (predictionTensor) predictionTensor.dispose();

      if (isTranslatingRef.current) {
        startScanning(); 
      }
    }
  };

  const toggleTranslation = () => {
    if (isTranslating) {
      setIsTranslating(false);
      setTranslation("Paused");
    } else {
      setIsTranslating(true);
      setTranslation("Waiting for sign...");
      startScanning();
    }
  };

  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>We need your permission to show the camera</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView 
        style={styles.camera} 
        ref={cameraRef} 
        facing="front" 
      />

      <View style={styles.translationContainer}>
        <Text style={styles.translationText}>{translation}</Text>
      </View>

      <View style={styles.controlsContainer}>
        <TouchableOpacity 
          style={[styles.button, isTranslating ? styles.buttonStop : styles.buttonStart]} 
          onPress={toggleTranslation}
          disabled={!trackerModel || !signModel}
        >
          <Text style={styles.buttonText}>
            {!trackerModel || !signModel 
              ? "Loading Brains..." 
              : isTranslating 
                ? "Stop Translating" 
                : "Start Translating"}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  translationContainer: {
    position: 'absolute',
    top: 60,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 20,
    borderRadius: 15,
    alignItems: 'center',
    zIndex: 10,
  },
  translationText: {
    color: '#00FF00', 
    fontSize: 28,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
  controlsContainer: {
    position: 'absolute',
    bottom: 100, 
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  button: {
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    minWidth: 200,
    alignItems: 'center',
  },
  buttonStart: {
    backgroundColor: '#4CAF50',
  },
  buttonStop: {
    backgroundColor: '#F44336',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  text: {
    color: 'white',
    textAlign: 'center',
    marginBottom: 20,
  }
});