import numpy as np
from locust import HttpUser, between, task
from PIL import Image


def prepare_image():
    """Load and preprocess the image as required."""
    image = Image.open("data/processed/pv_defection/images/train/20180630_154039.jpg")
    image = np.array(image)
    return image


image = prepare_image()


class BentoMLUser(HttpUser):
    """Locust user class for sending prediction requests to the server."""

    wait_time = between(1, 2)

    @task
    def send_prediction_request(self):
        """Send a prediction request to the server."""
        payload = {"image": image}  # Package the image as JSON
        self.client.post("/predict", json=payload, headers={"Content-Type": "application/json"})