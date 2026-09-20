import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load the exported Edge AI Brain (.tflite)
# The Interpreter is a lightweight engine designed specifically to run on mobile CPUs
interpreter = tf.lite.Interpreter(model_path="sign_language_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# The exact alphabetical order scikit-learn used to encode your words
LABELS = ['hello', 'thank_you', 'welcome']

# 2. Setup MediaPipe Hand Tracking (Identical to your tracker script)
model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Start the Camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("Live Inference Started. Show your signs to the camera!")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Extract the skeleton
    detection_result = detector.detect(mp_image)

    current_sign = "Waiting for hand..."
    confidence = 0.0

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Step A: Extract the exact 63 numbers for this single frame
            frame_coordinates = []
            for lm in hand_landmarks:
                frame_coordinates.extend([lm.x, lm.y, lm.z])
                # Draw a quick dot on the hand for visual feedback
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Step B: Format the numbers for the Neural Network
            # TF Lite expects a 2D numpy array of type float32, shaped (1, 63)
            input_data = np.array([frame_coordinates], dtype=np.float32)
            
            # Step C: Feed the math to the Brain and get the prediction
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Step D: Find the word with the highest probability score
            predicted_index = np.argmax(output_data)
            confidence = output_data[predicted_index] * 100
            
            # Only display the word if the AI is more than 60% sure
            if confidence > 60.0:
                current_sign = LABELS[predicted_index]
            else:
                current_sign = "Unsure"

    # Display the AI's translation on the screen
    cv2.putText(frame, f"Translation: {current_sign}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    if confidence > 0:
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Sign Language AI - Live Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()