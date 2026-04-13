import os
import cv2
import matplotlib.pyplot as plt

# Step 1: Define image path
image_path = "data/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg"

# Step 2: Check if file exists
if not os.path.exists(image_path):
    print("Image path not found. Please check dataset path.")
else:
    # Step 3: Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 4: Display image
    plt.imshow(image, cmap='gray')
    plt.title("Sample X-Ray Image")
    plt.axis('off')
    plt.show()