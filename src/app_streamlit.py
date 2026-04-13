import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Medical Image Analysis", layout="centered")

st.title("🩺 AI Medical Image Analysis")
st.write("Upload an X-ray image to detect Pneumonia")

# Load model
model = tf.keras.models.load_model("models/medical_ai_model.h5")

# Upload file
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = image.reshape(1, 256, 256, 1)

    # Predict
    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        st.error("⚠️ Pneumonia Detected")
    else:
        st.success("✅ Normal")