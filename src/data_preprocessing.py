import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Define paths
train_path = "data/chest_xray/train"
test_path = "data/chest_xray/test"

# Step 2: Create Image Data Generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Step 3: Load training data
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)

# Step 4: Load testing data
test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)

print("Data preprocessing completed successfully!")
print("Classes:", train_data.class_indices)