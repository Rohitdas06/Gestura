import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';

const customModelJson = require('../../assets/model/model.json'); 
const customModelWeights = require('../../assets/model/group1-shard1of1.bin');
const customModelLabels = require('../../assets/model/labels.json');

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isTranslating, setIsTranslating] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [trackerModel, setTrackerModel] = useState<handpose.HandPose | null>(null);
  const [signModel, setSignModel] = useState<tf.LayersModel | null>(null);
  const [translation, setTranslation] = useState("Waiting for sign...");
  const [statusInfo, setStatusInfo] = useState("Ready to start"); 
  
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
    if (!cameraRef.current || !trackerModel || !signModel || !isTranslatingRef.current) return;
    if (isProcessingFrame.current) return; 

    isProcessingFrame.current = true;
    
    let imageTensor: tf.Tensor3D | null = null;
    let resizedTensor: tf.Tensor3D | null = null;
    let inputTensor: tf.Tensor2D | null = null;
    let predictionTensor: tf.Tensor | null = null;

    try {
      // Changed to a 2-second countdown for faster testing
      setStatusInfo("📸 POSE NOW!");
      console.log("📸 POSE NOW! (Waiting 2 seconds for you to lift your hand...)");
      
      await new Promise(resolve => setTimeout(resolve, 2000));
      if (!isTranslatingRef.current) return;
      
      const photo = await cameraRef.current.takePictureAsync({ 
        base64: true, 
        quality: 0.1, 
        shutterSound: false 
      });
      
      if (!isTranslatingRef.current) return;

      setStatusInfo("🧠 ANALYZING... (Rest your hand)");
      console.log("🧠 ANALYZING AI...");
      
      await new Promise(resolve => setTimeout(resolve, 150)); 

      const imgBuffer = tf.util.encodeString(photo.base64, 'base64').buffer;
      const raw = new Uint8Array(imgBuffer);
      
      imageTensor = decodeJpeg(raw);
      resizedTensor = tf.image.resizeBilinear(imageTensor, [320, 240]).toInt();

      await new Promise(resolve => setTimeout(resolve, 150)); 
      if (!isTranslatingRef.current) return;

      const predictions = await trackerModel.estimateHands(resizedTensor, true);
      
      if (predictions.length > 0) {
        const landmarks = predictions[0].landmarks;
        const flatLandmarks = landmarks.flat();
        inputTensor = tf.tensor2d([flatLandmarks]);
        
        predictionTensor = signModel.predict(inputTensor) as tf.Tensor;
        const predictionArray = predictionTensor.dataSync();
        
        const maxProbabilityIndex = predictionArray.indexOf(Math.max(...predictionArray));
        const predictedWord = customModelLabels[maxProbabilityIndex];
        
        setTranslation(predictedWord);
        setStatusInfo("✅ Success!");
        console.log("✅ Sign detected:", predictedWord);
      } else {
        setStatusInfo("❌ No hand found.");
        console.log("❌ No hands detected in this frame. (Tip: Show your wrist and hold still!)");
      }

    } catch (error) {
      console.error("Scanning Error:", error);
      setStatusInfo("⚠️ Error processing");
    } finally {
      isProcessingFrame.current = false;
      
      if (imageTensor) imageTensor.dispose();
      if (resizedTensor) resizedTensor.dispose();
      if (inputTensor) inputTensor.dispose();
      if (predictionTensor) predictionTensor.dispose();

      if (!isTranslatingRef.current) {
        setTimeout(() => {
          setIsStopping(false);
          setStatusInfo("Paused");
        }, 500);
        console.log("🛑 Translation Paused.");
        return;
      }

      setTimeout(() => {
        if (isTranslatingRef.current) startScanning();
      }, 1500);
    }
  };

  const toggleTranslation = () => {
    if (isTranslating) {
      setIsTranslating(false);
      setIsStopping(true);
      setStatusInfo("Finishing current thought...");
      console.log("⚠️ Stop button clicked, waiting for AI to finish...");
    } else {
      setIsTranslating(true);
      setIsStopping(false);
      setTranslation("Waiting for sign...");
      console.log("▶️ Starting translation loop...");
      setTimeout(startScanning, 100);
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
        <Text style={styles.statusText}>{statusInfo}</Text>
      </View>

      <View style={styles.controlsContainer}>
        <TouchableOpacity 
          style={[
            styles.button, 
            !trackerModel ? styles.buttonLoading : 
            isStopping ? styles.buttonStopping :
            isTranslating ? styles.buttonStop : 
            styles.buttonStart
          ]} 
          onPress={toggleTranslation}
          disabled={!trackerModel || !signModel || isStopping}
        >
          <Text style={styles.buttonText}>
            {!trackerModel || !signModel 
              ? "Loading Brains..." 
              : isStopping
                ? "Stopping..."
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
  statusText: {
    color: '#FFA500', 
    fontSize: 16,
    fontWeight: '600',
    marginTop: 10,
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
    backgroundColor: '#4CAF50', // Green
  },
  buttonStop: {
    backgroundColor: '#F44336', // Red
  },
  buttonStopping: {
    backgroundColor: '#888888', // Grey
  },
  buttonLoading: {
    backgroundColor: '#333333', // Dark Grey
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