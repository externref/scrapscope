# ScrapScope: Garbage Classification System

ScrapScope is a machine learning-powered system for automatic garbage classification. It enables users to upload an image of waste, and the system predicts the type of garbage present. The project is built with TensorFlow/Keras for model training and FastAPI for serving predictions via a web API.

## Features
- **Image Classification Model:** Trains a Convolutional Neural Network (CNN) to classify images of garbage into categories such as battery, plastic, metal, paper, glass, etc.
- **REST API:** Provides a `/predict` endpoint for image uploads and real-time predictions.
- **Easy Dataset Structure:** Organize your dataset with one folder per class label for simple training.

## Project Structure
```
dataset/
  battery/
    battery1.jpg
    ...
  plastic/
    ...
  ...
backend/
  src/
    train.py        # Model training script
    predict.py      # Prediction function for image bytes
    server.py       # FastAPI server for predictions
```

## How It Works
### 1. Training the Model (`backend/src/train.py`)
- Loads images from the `dataset` directory, where each subfolder is a class label.
- Preprocesses images (resize to 64x64, normalize, encode labels).
- Trains a CNN for single-label classification.
- Evaluates and saves the trained model as `garbage_classifier.h5`.

### 2. Making Predictions (`backend/src/predict.py`)
- Loads the trained model and label encoder.
- Provides `predict_image_bytes(image_bytes, ...)` to classify an image from bytes.

### 3. Serving Predictions (`backend/src/server.py`)
- FastAPI server exposes a `/predict` endpoint.
- Accepts image uploads and returns the predicted garbage class as JSON.

## Example Usage
### Training
```bash
python backend/src/train.py
```

### Running the API Server
```bash
uvicorn backend.src.server:app --reload
```

### Predicting via API
```bash
curl -X POST "http://localhost:8000/predict" -F "image=@test.jpeg"
```

## Theory

### Problem Statement
Garbage classification is a computer vision problem where the goal is to automatically identify the type of waste in an image. This helps in automating waste sorting, improving recycling efficiency, and reducing environmental impact.

### Machine Learning Approach
This project uses supervised learning, where a model is trained on labeled images of garbage. Each image is associated with a class label (e.g., battery, plastic, metal). The model learns to map image features to these labels.

### Convolutional Neural Networks (CNNs)
CNNs are a type of deep neural network especially effective for image data. They use convolutional layers to automatically extract spatial features (edges, textures, shapes) from images. The model architecture typically includes:
- **Convolutional layers:** Extract local features using filters.
- **Pooling layers:** Downsample feature maps to reduce dimensionality.
- **Dense layers:** Combine features for final classification.
- **Softmax output:** Produces a probability distribution over classes.

### Training Process
- **Data Preparation:** Images are resized and normalized. Labels are encoded as integers.
- **Model Training:** The CNN is trained to minimize categorical cross-entropy loss, adjusting its weights to improve accuracy on the training data.
- **Evaluation:** The model's performance is measured on a held-out test set.

### Limitations of Single-label Classification
The current approach assumes each image contains only one type of garbage. For real-world scenarios where multiple objects may appear in one image, object detection or multi-label classification is required. These methods can localize and classify multiple items per image.

### Extending to Object Detection
Object detection models (e.g., SSD, YOLO, Faster R-CNN) predict both the class and location (bounding box) of each object in an image. This requires annotated data with bounding boxes and is more suitable for complex waste sorting tasks.

## Limitations & Next Steps
- **Single-label Only:** The current model predicts only one class per image. For images with multiple types of garbage, consider upgrading to object detection or multi-label classification.
- **Object Detection:** For multi-object images, use TensorFlow Object Detection API or YOLO for bounding box predictions.
- **Annotation:** Use tools like LabelImg or Roboflow to annotate images for object detection.

## Requirements
- Python 3.8+
- TensorFlow, scikit-learn, FastAPI, Pillow, Uvicorn


