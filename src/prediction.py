import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("models/medical_ai_model.h5")

# Step 1: Load image
image_path = "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))

# Step 2: Normalize
image = image / 255.0
image = image.reshape(1, 256, 256, 1)

# Step 3: Predict
prediction = model.predict(image)[0][0]

# Step 4: Result
if prediction > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")