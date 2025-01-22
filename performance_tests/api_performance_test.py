import numpy as np
from locust import HttpUser, between, task
from PIL import Image
import sys
import os

#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def prepare_image():
    """Load and preprocess the image as required."""
    image = Image.open("20180630_154039.jpg")
    image = np.array(image)
    return image.tolist()


image = prepare_image()


class BentoMLUser(HttpUser):
    """Locust user class for sending prediction requests to the server."""

    wait_time = between(1, 2)

    @task
    def send_prediction_request(self):
        """Send a prediction request to the server."""
        payload = {"input": image}  # Package the image as JSON
        self.client.post("/detect_and_predict", json=payload, headers={"Content-Type": "application/json"})