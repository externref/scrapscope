import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('garbage_classifier_mobilenet.keras')

model = load_model()

# Dynamically get class names from dataset folder
DATASET_PATH = 'dataset'
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

st.title("Garbage Classification App")
st.write("Upload an image of garbage and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    img_size = (224, 224)
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]

    st.write(f"**Prediction:** {pred_class}")
