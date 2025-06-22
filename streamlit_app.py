import os
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Dynamically get class names from dataset folder
DATASET_PATH = 'dataset'
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('garbage_classifier.keras')

model = load_model()

st.sidebar.title("Garbage Classification App")
st.sidebar.write("Upload an image of garbage and the model will predict its class.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

result_placeholder = st.empty()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_size = (224, 224)
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    result_placeholder.markdown(f"**Prediction:** {pred_class}")
    st.image(image, caption='Uploaded Image', use_column_width=True)


