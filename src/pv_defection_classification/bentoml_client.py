import bentoml
import numpy as np
from PIL import Image
import cv2


if __name__ == "__main__":
    image = Image.open("data/processed/pv_defection/train/20180630_154039.jpg")
    # image = image.resize((224, 224))  # Resize to match the minimum input size of the model
    image = np.array(image)
    
    # image = np.transpose(image, (2, 0, 1))  # Change to CHW format
    # image = np.expand_dims(image, axis=0)  # Add batch dimension

    with bentoml.SyncHTTPClient("https://bento-service-38375731884.europe-west1.run.app") as client:
        resp = client.detect_and_predict(input=image)
        
        cv2.imwrite("output.jpg", resp)
