import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

dataset_path = "dataset"
img_size = (64, 64)


def load_data(dataset_path, img_size):
    X, y = [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            try:
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(X), np.array(y)


print("Loading data...")
X, y = load_data(dataset_path, img_size)
X = X / 255.0


le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)
model = Sequential(
    [
        Conv2D(
            32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 3)
        ),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(le.classes_), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

print("Evaluating model...")
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")

y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
y_true_labels = le.inverse_transform(np.argmax(y_test, axis=1))
print(classification_report(y_true_labels, y_pred_labels))

model.save("garbage_classifier.keras")
print("Model saved as garbage_classifier.keras")
