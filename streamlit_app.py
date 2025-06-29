import os
import warnings

warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import glob
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"
class_names = sorted(
    [
        d
        for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ]
)


@st.cache_data
def get_available_models():
    model_files = glob.glob("src/*.keras")
    model_names = []
    for file in model_files:
        model_name = file.replace(".keras", "").replace("src/", "")
        model_names.append((model_name, file))
    return model_names


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


st.sidebar.title("Garbage Classification App")
st.sidebar.write("Upload an image of garbage and the model will predict its class.")

st.title("🗂️ Garbage Classification System")
st.markdown("### AI-Powered Waste Sorting")
st.markdown(
    "This application uses deep learning models to classify different types of garbage and waste materials."
)

if st.button("ℹ️ About This App"):
    st.info("""
    **How it works:**
    1. Choose a trained model from the sidebar
    2. Set your confidence threshold
    3. Upload an image of garbage/waste
    4. Get instant classification results with confidence scores
    
    **Supported Categories:** Cardboard, Glass, Metal, Paper, Plastic, Trash
    """)

available_models = get_available_models()
if available_models:
    model_options = [name for name, _ in available_models]
    selected_model_name = st.sidebar.selectbox("Choose Model:", model_options)

    selected_model_path = next(
        path for name, path in available_models if name == selected_model_name
    )

    st.sidebar.info(f"Selected: {selected_model_name}")
    st.sidebar.write(f"File: `{selected_model_path}`")

    model = load_model(selected_model_path)
else:
    st.sidebar.error("No model files (.keras) found in the current directory!")
    st.stop()

confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.05)

uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

result_placeholder = st.empty()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_size = (224, 224)
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    pred_class_idx = np.argmax(prediction)
    pred_class = class_names[pred_class_idx]
    confidence = prediction[0][pred_class_idx]

    if confidence >= confidence_threshold:
        st.markdown(f"## Prediction: **{pred_class.upper()}**")
        st.markdown(f"**Confidence:** {confidence:.2%}")
    else:
        st.markdown(f"## Low Confidence Prediction")
        st.markdown(f"**Best Guess:** {pred_class} ({confidence:.2%})")
        st.markdown("*Consider using a different model or better quality image*")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image, caption=f"Uploaded Image - Classified as: {pred_class}", width=300
        )

    with col2:
        prob_data = {}
        for i, class_name in enumerate(class_names):
            prob_data[class_name] = prediction[0][i]

        sorted_probs = sorted(prob_data.items(), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        labels = [item[0] for item in sorted_probs]
        sizes = [float(item[1]) for item in sorted_probs]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        def autopct_format(pct):
            return f"{pct:.1f}%" if pct > 5 else ""

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct=autopct_format, colors=colors, startangle=90
        )

        max_prob_idx = sizes.index(max(sizes))
        wedges[max_prob_idx].set_edgecolor("red")
        wedges[max_prob_idx].set_linewidth(3)

        ax.set_title(
            f"Classification Confidence\n(Model: {selected_model_name})",
            fontsize=14,
            fontweight="bold",
        )

        st.pyplot(fig)

    st.write("### Detailed Probabilities:")

    prob_df_data = []
    for i, (class_name, prob) in enumerate(sorted_probs):
        prob_df_data.append(
            {
                "Rank": i + 1,
                "Class": class_name.title(),
                "Probability": f"{prob:.4f}",
                "Percentage": f"{prob:.2%}",
            }
        )

    st.table(prob_df_data)
