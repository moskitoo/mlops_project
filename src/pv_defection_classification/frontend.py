import io
import os
from pathlib import Path

import bentoml
import cv2
import numpy as np
import streamlit as st
from google.cloud import run_v2
from PIL import Image


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


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    try:
        # Convert bytes to PIL Image
        image_pil = Image.open(io.BytesIO(image))

        # Convert PIL Image to numpy array with float32 type
        img_array = np.array(image_pil).astype(np.float32)

        with bentoml.SyncHTTPClient(backend) as client:
            # Detect and predict
            resp = client.detect_and_predict(input=img_array)

            # Ensure the response is in the correct format
            if isinstance(resp, np.ndarray):
                # Convert to uint8 if necessary
                if resp.dtype != np.uint8:
                    resp = (resp * 255).astype(np.uint8)

                return resp
            else:
                st.error(f"Unexpected response type: {type(resp)}")
                return None

    except Exception as e:
        st.error(f"Error communicating with backend: {e}")
        return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    st.title("PV Module Defect Detection")

    # Get backend URL
    backend = get_backend_url()
    if backend is None:
        st.error("Backend service not found")
        return

    print(f"Backend URL: {backend}")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PV module image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = uploaded_file.read()

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Process image
        with st.spinner("Processing image..."):
            result_image = classify_image(image, backend=backend)

        # Display processed image
        if result_image is not None:
            st.image(result_image, caption="Defect Detection Result", use_container_width=True)
        else:
            st.error("Failed to process the image")


if __name__ == "__main__":
    main()
