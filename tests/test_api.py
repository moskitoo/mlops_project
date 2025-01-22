import pytest
import numpy as np
import cv2
import onnxruntime
import bentoml
import subprocess
from src.pv_defection_classification.bentoml_service import PVClassificationService, download_model_from_gcp

EXAMPLE_INPUT = np.random.randint(0, 256, (800, 800, 3), dtype=np.uint8)

@pytest.fixture
def mock_download_model_from_gcp(mocker):
    mock_download = mocker.patch('src.pv_defection_classification.bentoml_service.download_model_from_gcp')
    mock_download.return_value = ('/path/to/model.onnx', None)
    return mock_download

def test_download_model_from_gcp():

    onnx_path, _ = download_model_from_gcp()

    assert type(onnx_path) == str

def test_init_success(mocker, mock_download_model_from_gcp):
    mocker.patch('src.pv_defection_classification.bentoml_service.onnxruntime.InferenceSession')
    service = PVClassificationService()

    # Verify that model was downloaded
    mock_download_model_from_gcp.assert_called_once()

    # Verify that InferenceSession was initialized with correct path
    onnxruntime.InferenceSession.assert_called_once_with('/path/to/model.onnx')

    # Verify that class labels were loaded from YAML
    assert service.class_labels == {0: 'working', 1: 'defected'}

    # Check other initialized attributes
    assert service.img_width == 640
    assert service.img_height == 640
    assert service.confidence_thres == 0.5
    assert service.iou_thres == 0.6

def test_preprocess():
    service = PVClassificationService()

    input_image = EXAMPLE_INPUT
    
    preprocessed = service.preprocess(input_image)
    
    # Check data type
    assert preprocessed.dtype == np.float32
    
    # Check shape: (1, 3, 640, 640)
    assert preprocessed.shape == (1, 3, 640, 640)
    
    # Check normalization
    assert preprocessed.max() <= 1.0
    assert preprocessed.min() >= 0.0
    
    # Check resizing plus normalization
    input_image[0, 0] = [10, 20, 30]  
    preprocessed = service.preprocess(input_image)

    # Resize input image to account for resizing in preprocess
    input_resized = cv2.resize(input_image, (640, 640))
    expected_rgb = [input_resized[0, 0, 2] / 255.0, input_resized[0, 0, 1] / 255.0, input_resized[0, 0, 0] / 255.0]

    actual_rgb = preprocessed[0, :, 0, 0].tolist()

    assert np.allclose(actual_rgb, expected_rgb, atol=1e-1)

def test_postprocess():
    service = PVClassificationService()

    output = np.random.rand(1, 10, 85).astype(np.float32)
    
    sample_result = [np.array([[[     4.4373,      12.381,      20.104],
        [     4.0638,      4.8612,      5.1262],
        [     128.17,      135.29,      137.27],
        [     126.28,      133.78,      136.49],
        [  0.9,  0.6,  0.2],
        [  0.1,  0.3,   0.7]]], dtype=np.float32)]


    output_image = service.postprocess(output, sample_result)

    # Check data type
    assert output_image.dtype == np.uint8

    # Check shape: (640, 640, 3)
    assert output_image.shape == (640, 640, 3)

    # Check normalization
    assert output_image.max() <= 255
    assert output_image.min() >= 0
    
    # Check resizing
    output_resized = cv2.resize(output_image, (85, 10))
    assert np.allclose(output_resized, output[0])

def test_detect_and_predict(mocker):
    service = PVClassificationService()

    mock_preprocess = mocker.patch.object(service, 'preprocess', return_value=np.random.rand(1, 3, 640, 640).astype(np.float32))
    mock_run = mocker.patch.object(service.model, 'run', return_value=[np.random.rand(1, 10, 85).astype(np.float32)])
    mock_postprocess = mocker.patch.object(service, 'postprocess', return_value=np.ones((640,640,3), dtype=np.uint8))

    input_image = np.zeros((800, 800, 3), dtype=np.uint8)
    output_image = service.detect_and_predict(input_image)

    mock_preprocess.assert_called_once_with(input_image)
    mock_run.assert_called_once()
    mock_postprocess.assert_called_once()

    assert output_image.shape == (640, 640, 3)
    assert output_image.dtype == np.uint8
    assert np.all(output_image == 1) #Check if the postprocess mock worked correctly

def test_defection_detection_service_integration():
    with subprocess.Popen(["bentoml", "serve", "src.pv_defection_classification.bentoml_service:PVClassificationService", "-p", "3000"]) as server_proc:
        try:
            client = bentoml.SyncHTTPClient("http://localhost:3000", server_ready_timeout=300)
            response = client.detect_and_predict(input=EXAMPLE_INPUT)

            # Ensure the response is not empty
            assert response.any(), "The response should not be empty."
            # Check the type of the response
            assert isinstance(response, np.ndarray), "The response should be a numpy array."
            # Check that the response is of the correct shape
            assert response.shape == (640,640,3), "The response should be of shape (640,640,3)."
        finally:
            server_proc.terminate()