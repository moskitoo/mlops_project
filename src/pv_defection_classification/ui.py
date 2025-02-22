import io
import os
from google.cloud import run_v2

import numpy as np
import requests
import streamlit as st
from PIL import Image
import cv2

@st.cache_resource  
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/mlops-pv-classification/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "bento-service":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name

def predict(img, backend):
    """
    A function that sends a prediction request to the API and return a cuteness score.
    """
    # Convert the bytes image to a NumPy array
    #bytes_image = img.getvalue()
    numpy_image_array = np.array(Image.open(io.BytesIO(img))).tolist()

    predict_url = f"{backend}/detect_and_predict"

    payload = {
    "input": numpy_image_array  # Replace with actual input data
    }   

    # Send the image to the API
    response = requests.post(
        predict_url,
        json=payload,
    )

    if response.status_code == 200:
        return np.array(response.json()[:])
    else:
        raise Exception("Status: {}".format(response.status_code))

def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)
    
    # Create the header page content
    st.title("Pholtovoltaic Defection Detection App")
    st.markdown(
        "### Upload a thermography image of a solar panel to detect and predict the defective modules.",
        unsafe_allow_html=True,
    )

    # Upload a simple cover image
    # with open("data/processed/pv_defection/images/train/20180630_154039.jpg", "rb") as f:
    #     st.image(f.read(), use_column_width=True)

    st.text("Get a picture of your solar panel and upload it here.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    left_column, right_column = st.columns(2)

    if uploaded_file is not None:
        image = uploaded_file.read()

        with st.spinner("Detecting and predicting..."):
            result = predict(image, backend)

            if result is not None:

                st.success("Prediction successful!")
                
                result = result.astype(np.uint8)/255.0

                image = cv2.resize(np.array(Image.open(io.BytesIO(image))), (640, 640))

                # show the image and prediction
                # You can use a column just like st.sidebar:
                with left_column:
                    st.image(image, caption="Uploaded Image")

                # Or even better, call Streamlit functions inside a "with" block:
                with right_column:
                    st.image(result, caption="Processed Image", channels="BGR")

            else:

                with left_column:
                    st.image(image, caption="Uploaded Image")
                
                with right_column:
                    st.write("Failed to get prediction")


if __name__ == "__main__":
    main()