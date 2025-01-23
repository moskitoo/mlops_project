import pandas as pd
from google.cloud import storage
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from sklearn.preprocessing import StandardScaler
import json
from utils.convert_drift_data_to_yolo_format import run_yolo_inference_on_user_data

# Parse YOLO label content
def parse_yolo_label_content(content):
    valid_data = []
    lines = content.strip().split(b"\n")  # Read as bytes and split lines

    for line in lines:
        decoded_line = line.decode('utf-8', errors='ignore').strip()  # Decode line-by-line
        parts = decoded_line.split()

        if len(parts) == 5 and all(p.replace('.', '', 1).isdigit() for p in parts):
            valid_data.append([int(parts[0])] + list(map(float, parts[1:])))

    return valid_data

# Download training data from GCS bucket
def download_training_data(DRIFT_BUCKET_NAME, SOURCE_PATH):
    """Download and process YOLO training data from GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(DRIFT_BUCKET_NAME)

    labels_path = SOURCE_PATH + "labels/train"
    blobs = bucket.list_blobs(prefix=labels_path)

    data_list = []

    for blob in blobs:
        if blob.name.endswith(".txt"): # ignore cache files
            binary_content = blob.download_as_bytes() 
            parsed_data = parse_yolo_label_content(binary_content)

            data_list.append(parsed_data)

    return data_list

# Convert user data (str) to numeric format
def convert_user_data_to_numeric(formatted_results):
    numeric_results = []
    for result in formatted_results:
        image_boxes = []
        for r in result:
            parts = r.split()
            parts = [int(parts[0])] + list(map(float, parts[1:]))
            image_boxes.append(parts)
        numeric_results.append(image_boxes)
    return numeric_results

# Convert data to DataFrame
def prepare_dataframe(data_list):
    flat_list = [item for sublist in data_list for item in sublist]
    df = pd.DataFrame(flat_list, columns=['class', 'x_center', 'y_center', 'width', 'height'])
    return df


# RUN DATA DRIFT ANALYSIS

DRIFT_BUCKET_NAME = "test-pv-2"
SOURCE_PATH  = "data/processed/pv_defection/"

training_data = download_training_data(DRIFT_BUCKET_NAME, SOURCE_PATH)
#print(training_data[0][0])
drift_data = convert_user_data_to_numeric(run_yolo_inference_on_user_data())
#print(drift_data[0][0])

training_df = prepare_dataframe(training_data)
drift_df = prepare_dataframe(drift_data)

# Standardize column names
reference_df = training_df.rename(
    columns={'class': 'target', 'x_center': 'x', 'y_center': 'y', 'width': 'w', 'height': 'h'}
)
current_df = drift_df.rename(
    columns={'class': 'target', 'x_center': 'x', 'y_center': 'y', 'width': 'w', 'height': 'h'}
)

# Standardize data
scaler = StandardScaler()
reference_df[['x', 'y', 'w', 'h']] = scaler.fit_transform(reference_df[['x', 'y', 'w', 'h']])
current_df[['x', 'y', 'w', 'h']] = scaler.transform(current_df[['x', 'y', 'w', 'h']])

# Create drift report
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])

# Run and save report
report.run(reference_data=reference_df, current_data=current_df)
report.save_html('pv_defection_data_drift_report.html')
