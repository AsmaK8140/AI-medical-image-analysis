import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_preprocessing import train_data, test_data

# Step 1: Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

# Step 2: Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 3: Train Model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# Step 4: Save training history
import pickle
with open("outputs/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Step 5: Save Model
model.save("models/medical_ai_model.h5")

print("Model training completed and saved!")