import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from data_preprocessing import test_data

# Load trained model
model = tf.keras.models.load_model("models/medical_ai_model.h5")

# Step 1: Predict
y_pred = model.predict(test_data)
y_pred = np.round(y_pred)

# Step 2: True labels
y_true = test_data.classes

# Step 3: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Step 4: Classification Report
report = classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'])
print("\nClassification Report:")
print(report)