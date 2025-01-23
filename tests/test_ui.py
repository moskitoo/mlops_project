from src.pv_defection_classification.ui import get_backend_url, predict
import numpy as np
import os
import subprocess
import time
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

@pytest.fixture(scope="module")
def streamlit_app():
    """
    Fixture to start and stop the Streamlit app.
    """
    # Start the Streamlit app
    process = subprocess.Popen(
        ["streamlit", "run", "src/pv_defection_classification/ui.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Wait for the app to launch
    time.sleep(5)  # Adjust if necessary
    yield
    # Terminate the Streamlit app after tests
    process.terminate()
    process.wait()

@pytest.fixture(scope="module")
def driver():
    """
    Fixture to initialize and quit the Selenium WebDriver.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    yield driver
    driver.quit()

def test_get_backend_url():
    backend = get_backend_url()

    assert isinstance(backend, str)
    assert "https://" in backend
    assert "bento-service" in backend

def test_predict():
    # Load a sample image
    with open("data/processed/pv_defection/images/train/20180630_154039.jpg", "rb") as f:
        img = f.read()

    backend = get_backend_url()
    result = predict(img, backend)

    assert isinstance(result, np.ndarray)
    assert result.shape == (640, 640, 3)
    assert result.max() <= 255.0
    assert result.min() >= 0.0
    assert result.dtype in (np.uint8, np.float32, np.float64)

def test_image_upload_and_column_display(streamlit_app, driver):
    """
    Test uploading an image and verifying both columns display results.
    """
    # Navigate to the Streamlit app
    driver.get("http://localhost:8501")

    wait = WebDriverWait(driver, 60)

    # Locate the file uploader element
    file_input = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
    )

    # Path to the test image
    current_dir = os.path.dirname(os.path.abspath(''))
    image_path = os.path.join(current_dir, "mlops_project/data/processed/pv_defection/images/train/20180630_154039.jpg")

    # Upload the image
    file_input.send_keys(image_path)

    # Wait for the spinner to disappear and success message to appear
    wait.until(
        EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Prediction successful!')]"))
    )

    # Verify that both columns display images
    # Streamlit typically uses <img> tags for images
    images = driver.find_elements(By.TAG_NAME, "img")
    assert len(images) >= 2, "Expected at least two images to be displayed in columns."

    # Optionally, verify captions
    captions = driver.find_elements(By.XPATH, "//div[contains(@class, 'element-container')]//div[contains(text(), 'Uploaded Image') or contains(text(), 'Processed Image')]")
    assert any("Uploaded Image" in caption.text for caption in captions), "Uploaded Image caption not found."
    assert any("Processed Image" in caption.text for caption in captions), "Processed Image caption not found."

