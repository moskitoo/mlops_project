import requests
import numpy as np
from PIL import Image

# Open or create your PIL Image
image = Image.open("data/processed/pv_defection/train/20180630_154039.jpg")  # Replace with your image path
image = np.array(image).tolist()
# Convert the PIL Image to bytes


payload = {
    "input": image  # Replace with actual input data
}

# image = image.resize((224, 224))  # Resize to match the minimum input size of the model

response = requests.post("http://127.0.0.1:3000/detect_and_predict", json=payload)

print(response.status_code)
