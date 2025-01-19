import bentoml
from ultralytics import YOLO
import onnx
import onnxruntime
import numpy as np
import cv2
import glob
from pydantic import BaseModel, Field
from PIL import Image
from typing import Any, Dict, List


@bentoml.service #(resources={"cpu": 2}, traffic={'timeout': '60'})
class PVClassificationService:
    def __init__(self) -> None:
        self.model = onnxruntime.InferenceSession("yolo11n.onnx")
        self.model_inputs = self.model.get_inputs()

        self.img_width = 640
        self.img_height = 640
        self.input_width = self.model_inputs[0].shape[2]
        self.input_height = self.model_inputs[0].shape[3]
        self.confidence_thres = 0.5
        self.iou_thres = 0.5

    
    def draw_detections(self, img, box, score, class_id):
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
        label = f"{'test'}: {score:.2f}"

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
    
    def preprocess(self, image: np.ndarray) -> List:
        """
        Preprocess the input image for inference.

        Args:
            image (PIL.Image.Image): Input image to preprocess.

        Returns:
            List: List containing the preprocessed image.
        """
        image = np.array(image).astype(np.float32)
        image = cv2.resize(image, (640, 640 ))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    
    def postprocess(self, input: np.ndarray, output: np.ndarray) -> np.ndarray:
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
            self.draw_detections(self.img, box, score, class_id)

        # Return the modified input image
        return input

    
    @bentoml.api(batchable=True,
                    batch_dim=(0, 0),
                    max_batch_size=128,
                    max_latency_ms=1000,)
    def detect_and_predict(self, input: np.ndarray) -> np.ndarray:
        """
        Detect and predict the defective PV modules in the input image.

        Args:
            params (Any): Input parameters.

        Returns:
            Dict: Dictionary containing the prediction results.
        """
        preprocess_image = self.preprocess(input)

        inference_result = self.model.run(None, {self.model_inputs[0].name: preprocess_image})

        postprocess_image = self.postprocess(input, inference_result)

        return postprocess_image