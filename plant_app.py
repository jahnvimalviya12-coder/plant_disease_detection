
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Load class names
with open("class_names.pkl", "rb") as f:
    classes = pickle.load(f)

# Title
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect the disease.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    # Load image
    img = image.load_img(uploaded_file, target_size=(128, 128))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Disease: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
