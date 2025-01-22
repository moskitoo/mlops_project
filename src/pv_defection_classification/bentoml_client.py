import bentoml
import numpy as np
from PIL import Image
import cv2


if __name__ == "__main__":
    image = Image.open("data/processed/pv_defection/images/train/20180630_154039.jpg")
    
    image = np.array(image)

#https://bento-service-38375731884.europe-west1.run.app
    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        resp = client.detect_and_predict(input=image)

        cv2.imwrite("output.jpg", resp)
