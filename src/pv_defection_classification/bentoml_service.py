import bentoml
import onnxruntime
import numpy as np
import cv2
import time
import yaml
from google.cloud import storage
from ultralytics import YOLO
import json
import datetime
import asyncio
from prometheus_client import Counter, Histogram

# Define Prometheus metrics
request_counter = Counter(
    name='summary_requests_total',
    documentation='Total number of inference requests',
    labelnames=['status']
)

inference_time_histogram = Histogram(
    name='inference_time_seconds',
    documentation='Time taken for yolo inference',
    labelnames=['status'],
    buckets=(0.1, 0.2, 0.5, 1, 2, 5, 10, float('inf')) 
)

input_size_request = Histogram(
    name='input_size_bytes',
    documentation='Size of the input image',
    labelnames=['status'],
    buckets=(0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, float('inf'))
)

BUCKET_NAME = "yolo_model_storage"
DRIFT_BUCKET_NAME = "data_drifting"
MODEL_NAME = "pv_defection_classification_model.pt"
MODEL_FILE_NAME = "pv_defection_model.pt"

def download_model_from_gcp():
    """Download the model from GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_NAME)
    blob.download_to_filename(MODEL_FILE_NAME)

    model = YOLO(MODEL_FILE_NAME)
    onnx_path = model.export(format="onnx", optimize=True)

    return onnx_path, model
@bentoml.service(metrics={"enabled": True})
class PVClassificationService:
    def __init__(self) -> None:
        # Download model from GCP bucket
        onnx_path, _ = download_model_from_gcp()
        self.model = onnxruntime.InferenceSession(onnx_path)
        self.model_inputs = self.model.get_inputs()

        self.img_width = 640
        self.img_height = 640
        self.input_width = self.model_inputs[0].shape[2]
        self.input_height = self.model_inputs[0].shape[3]
        self.confidence_thres = 0.5
        self.iou_thres = 0.6

        try:
            with open("yolo_class_label.yaml", "r") as file:
                try:
                    self.class_labels = yaml.safe_load(file)
                except Exception:
                    self.class_labels = {0: "working", 1: "defected"}
        except Exception:
            self.class_labels = {0: "working", 1: "defected"}
    
    # Extract basic properties from inference results
    def calculate_prediction(self, inference_result: list) -> tuple:
        """
        Calculate the number of detected, working, and defective PV modules based on the inference results.

        Args:
            inference_result (list): List containing the inference results.

        Returns:
            Tuple: Tuple containing the number of detected, working, and defective PV modules.
        """
        
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(inference_result[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Variables to store the number of detected, working, and defective PV modules
        n_detected_pv_modules = 0
        n_working_modules = 0
        n_defective_modules = 0

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Increment the number of detected PV modules
                n_detected_pv_modules += 1

                # Increment the number of working or defective modules
                if class_id == 0:
                    n_working_modules += 1
                elif class_id == 1:
                    n_defective_modules += 1

        return n_detected_pv_modules, n_working_modules, n_defective_modules


    # Save prediction results to GCP
    async def save_prediction_to_gcp(self, input: np.ndarray, inference_result: list) -> None:
        """
        Save the prediction results to GCP bucket.
        
        Args:
            input (np.ndarray): Input image data.
            inference_result (list): Inference results.
        
        Returns:
            None
        """
        client = storage.Client()
        bucket = client.bucket(DRIFT_BUCKET_NAME)
        time = datetime.datetime.now(tz=datetime.UTC)
        # Prepare prediction data
        n_detected_pv_modules, n_working_modules, n_defective_modules = self.calculate_prediction(inference_result)
        data = {
                "input":input.tolist(),
                "n_detected_pv_modules": n_detected_pv_modules,
                "n_working_modules": n_working_modules,
                "n_defective_modules": n_defective_modules,
            }
        blob = bucket.blob(f"prediction_{time}.json")
        blob.upload_from_string(json.dumps(data))

    def draw_detections(self, img: np.ndarray, box: list, score: float, class_id: int) -> None:
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = (0, 255, 0)

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        try:
            detected_object = self.class_labels[class_id]
        except Exception:
            detected_object = "Unknown"

        label = f"{detected_object}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for inference.

        Args:
            image (PIL.Image.Image): Input image to preprocess.

        Returns:
            List: List containing the preprocessed image.
        """
        image = image.astype(np.float32)

        image[..., (0, 1, 2)] = image[..., (2, 1, 0)]
    
        # Resize the image to match the input shape
        image_data_return = cv2.resize(image, (640, 640))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(image_data_return) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1)) 

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        
        return image_data, image_data_return
    
    async def postprocess(self, input: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Postprocess the inference result.

        Args:
            input (np.ndarray): Inference result to postprocess.
            output (np.ndarray): Inference output to postprocess.

        Returns:
            np.ndarray: Postprocessed inference result.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]
        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input, box, score, class_id)

        # Return the modified input image
        return input

    @bentoml.api(
        batchable=False,
        batch_dim=(0, 0),
        max_batch_size=8,
        max_latency_ms=1000,
    )
    async def detect_and_predict(self, input: np.ndarray) -> np.ndarray:
        """
        Detect and predict the defective PV modules in the input image.

        Args:
            params (Any): Input parameters.

        Returns:
            Dict: Dictionary containing the prediction results.
        """
        start_time = time.time()
        try:
            # Process the input image and perform inference
            preprocess_image, image_resized = self.preprocess(input)

            inference_result = self.model.run(None, {self.model_inputs[0].name: preprocess_image})

            # Post-processing draws the detected bounding boxes on the input image
            future = await asyncio.gather(
                        self.postprocess(image_resized, inference_result),
                        self.save_prediction_to_gcp(input, inference_result)
                        )
            
            postprocess_image = future[0]

            status = 'success'
            input_size = input.nbytes

        except Exception:
            # Generate white image with error message
            postprocess_image = np.ones((640, 640, 3), np.uint8) * 255
            cv2.putText(postprocess_image, 'Something went wrong :(',(10, 320), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 
                        2, cv2.LINE_AA
                        )
            status = 'failure'
            input_size = 0

        finally:
            # Measure how long the inference took and update the histogram
            inference_time_histogram.labels(status=status).observe(time.time() - start_time)
            input_size_request.labels(status=status).observe(input_size)
            # Increment the request counter
            request_counter.labels(status=status).inc()

        return postprocess_image
