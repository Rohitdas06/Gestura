import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the Dataset
print("Loading dataset...")
df = pd.read_csv('sign_language_dataset.csv')

# 2. Extract Features (X) and Labels (y)
# Drop the text columns ('label' and 'hand') so X only contains the 63 math coordinates
X = df.drop(['label', 'hand'], axis=1).values
y = df['label'].values

# 3. Encode the Labels
# Neural networks only understand math, not words. 
# This converts ["hello", "thank_you", "welcome"] into [0, 1, 2]
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Training on {num_classes} classes: {encoder.classes_}")

# 4. Split the Data
# Keep 80% of data for training, and hide 20% to test the AI on unseen data later
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Build the Neural Network Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(63,)),         # Input: 63 coordinates
    tf.keras.layers.Dense(128, activation='relu'),         # Hidden Layer 1: Finds basic patterns
    tf.keras.layers.Dropout(0.2),                          # Regularization: Randomly turns off nodes to prevent overfitting
    tf.keras.layers.Dense(64, activation='relu'),          # Hidden Layer 2: Finds complex patterns
    tf.keras.layers.Dense(num_classes, activation='softmax') # Output: Gives a probability % for each sign
])

# 6. Compile the Model
# Adam is the standard optimization algorithm for updating network weights
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 7. Train the Model
print("Starting Neural Network Training...")
# 'epochs=50' means the AI will review the dataset 50 times to learn
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 8. Export for Mobile Edge Computing
# Convert the heavy TensorFlow model into a lightweight .tflite format for your mobile app
print("Converting model for mobile...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("SUCCESS: Model saved as 'sign_language_model.tflite'")