import io
import os

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def predict_image_bytes(
    image_bytes,
    model_path="garbage_classifier.keras",
    dataset_path="dataset",
    img_size=(64, 64),
):
    le = LabelEncoder().fit(
        [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
    )
    model = load_model(model_path)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(img_size)
    img = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    return le.inverse_transform([np.argmax(model.predict(img))])[0]
