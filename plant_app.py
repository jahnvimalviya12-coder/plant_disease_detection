import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load model
model = tf.keras.models.load_model("plant_disease_model_new.h5")

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect its disease.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_index = np.argmax(prediction)

    st.write("Prediction Shape:", prediction.shape)
    st.write("Predicted Index:", predicted_index)
    st.write("Total Class Names:", len(class_names))

    confidence = np.max(prediction) * 100

    if predicted_index < len(class_names):
        st.success(f"Predicted Disease: {class_names[predicted_index]}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.error("Prediction index is out of range.")


