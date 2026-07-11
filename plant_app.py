import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Page settings
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿"
)

# Load trained model
model = tf.keras.models.load_model("plant_disease_model_new.h5")

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)


# Title
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect its disease.")


# Upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    # Display smaller image
    st.image(
        image,
        caption="Uploaded Leaf Image",
        width=300
    )


    # Preprocess image for model (128x128)
    img = image.resize((128, 128))

    img = np.array(img)

    # Normalize pixel values
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)


    # Prediction
    prediction = model.predict(img)

    predicted_index = np.argmax(prediction)


    # Display result
    st.success(
        f"🌿 Predicted Disease: {class_names[predicted_index]}"
    )
