from pathlib import Path

import typer
from dotenv import load_dotenv
from google.cloud import storage

import json
import numpy as np
import cv2
import os
from ultralytics import YOLO

MODEL_BUCKET_NAME = "yolo_model_storage"
MODEL_NAME = "pv_defection_classification_model.pt"
MODEL_FILE_NAME = "pv_defection_model.pt"

DRIFT_BUCKET_NAME = "data_drifting"
SOURCE_PATH  = ""

# Download user input data from GCS bucket
def download_data(DRIFT_BUCKET_NAME, SOURCE_PATH):
    """Download user data from GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(DRIFT_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=SOURCE_PATH)

    data_list = []  # Store data from all the files

    for blob in blobs:
        content = blob.download_as_text()
        data = json.loads(content)
        data_list.append(data)

    return data_list


# Get best model for training
def download_model(MODEL_BUCKET_NAME,MODEL_FILE_NAME,MODEL_NAME):
    """Download the model from GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET_NAME)
    blob = bucket.blob(MODEL_NAME)
    blob.download_to_filename(MODEL_FILE_NAME)

    model = YOLO(MODEL_FILE_NAME)
    #onnx_path = model.export(format="onnx", optimize=True)

    return model


# Convert user input data to image
def convert_rgb_to_image(rgb_input, output_path="user_input_image.jpg"):
    image = np.array(rgb_input, dtype=np.uint8)
    cv2.imwrite(output_path, image)
    #print("output path: ", output_path)
    return output_path


# Run model on images
def run_yolo_inference(image, model):
    results = model(image)  # Run inference

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])  # Class
            confidence = float(box.conf[0])  # Confidence score
            x_center, y_center, width, height = box.xywhn[0].tolist()  # Normalized bbox

            print(f"Class: {class_id}, Confidence: {confidence:.2f}, Box: {x_center:.2f}, {y_center:.2f}, {width:.2f}, {height:.2f}")

    return results


# Convert to yolo format
def format_yolo_output(results):
    formatted_boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x_center, y_center, width, height = box.xywhn[0].tolist()
            formatted_boxes.append(f"{cls} {x_center} {y_center} {width} {height}")

    return formatted_boxes

# RUNNING ALL FUNCTIONS

# Download user input data
data_list = download_data(DRIFT_BUCKET_NAME, SOURCE_PATH)

# Download model
model = download_model(MODEL_BUCKET_NAME, MODEL_FILE_NAME, MODEL_NAME)


for i, data in enumerate(data_list):
    # Convert user input data to an image
    image_path = convert_rgb_to_image(data_list[i]["input"])
    image = cv2.imread(image_path)

    #cv2.imshow("Image Window", image)
    #cv2.waitKey(1000)
    #cv2.destroyAllWindows()

    # Run model on the image
    results = run_yolo_inference(image, model)

    # Convert to yolo format
    formatted_results = format_yolo_output(results)
    # for r in formatted_results:
    #     print(r)


