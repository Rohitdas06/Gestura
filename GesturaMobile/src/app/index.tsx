import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system';

// Import your custom model assets
const customModelJson = require('../assets/model/model.json'); // Adjusted to 1 directory up from 'app'
const customModelWeights = require('../assets/model/group1-shard1of1.bin');
const customModelLabels = require('../assets/model/labels.json');

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isTranslating, setIsTranslating] = useState(false);
  const [trackerModel, setTrackerModel] = useState<handpose.HandPose | null>(null);
  const [signModel, setSignModel] = useState<tf.LayersModel | null>(null);
  const [translation, setTranslation] = useState("Waiting for sign...");
  
  const cameraRef = useRef<any>(null);
  const isTranslatingRef = useRef(isTranslating); // Ref used to immediately stop the background loop

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

      // 1. Load Handpose
      const tracker = await handpose.load();
      setTrackerModel(tracker);
      console.log("Hand Tracker Loaded!");

      // 2. Load Custom Translation Model
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
    if (!cameraRef.current || !trackerModel || !signModel) return;

    try {
      // 1. Snapping photo
      const photo = await cameraRef.current.takePictureAsync({ base64: false, quality: 0.1 });
      
      // Stop immediately if user pressed stop during the photo snap
      if (!isTranslatingRef.current) return;

      // 2. Shrinking image size to prevent memory freeze
      const manipResult = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 300 } }],
        { format: ImageManipulator.SaveFormat.JPEG }
      );

      // 3. Converting tiny image to AI Tensor
      const imgB64 = await FileSystem.readAsStringAsync(manipResult.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
      const raw = new Uint8Array(imgBuffer);
      const imageTensor = decodeJpeg(raw);

      // 4. Scanning for hands
      const predictions = await trackerModel.estimateHands(imageTensor);
      
      // 5. Translating if hands are found
      if (predictions.length > 0) {
        console.log("Hands found: 1");
        
        // Extract 3D coordinates (21 joints)
        const landmarks = predictions[0].landmarks;
        
        // Flatten into a single array of 63 numbers
        const flatLandmarks = landmarks.flat();
        
        // Convert to a Tensor (Shape: [1, 63])
        const inputTensor = tf.tensor2d([flatLandmarks]);
        
        // Run prediction
        const predictionTensor = signModel.predict(inputTensor) as tf.Tensor;
        const predictionArray = predictionTensor.dataSync();
        
        // Find highest probability
        const maxProbabilityIndex = predictionArray.indexOf(Math.max(...predictionArray));
        const predictedWord = customModelLabels[maxProbabilityIndex];
        
        setTranslation(predictedWord);
        console.log("Translated:", predictedWord);

        // Clean up tensors inside the if-block
        inputTensor.dispose();
        predictionTensor.dispose();
      } else {
        console.log("Hands found: 0");
      }

      // Clean up the image tensor
      imageTensor.dispose();

      // Loop again if still translating
      if (isTranslatingRef.current) {
        // Small timeout to let the UI breathe
        setTimeout(startScanning, 100);
      }

    } catch (error) {
      console.error("Scanning Error:", error);
      if (isTranslatingRef.current) {
        setTimeout(startScanning, 500);
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
      // Start the loop asynchronously
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
      <CameraView style={styles.camera} ref={cameraRef} facing="front">
        {/* Top Overlay: Display the Translation */}
        <View style={styles.translationContainer}>
          <Text style={styles.translationText}>{translation}</Text>
        </View>

        {/* Bottom Overlay: Controls */}
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
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
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
  },
  translationText: {
    color: '#00FF00', // Hacker green for that CS vibe
    fontSize: 28,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
  controlsContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    paddingBottom: 40,
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