import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.pv_defection_classification.model import load_pretrained_model, save_model


@pytest.fixture
def mock_yolo():
    """
    A pytest fixture to mock the YOLO class.
    """
    with patch("src.pv_defection_classification.model.YOLO") as mock_yolo:
        mock_instance = MagicMock()  
        mock_yolo.return_value = mock_instance
        yield mock_yolo, mock_instance


def test_load_pretrained_model_with_weights(mock_yolo):
    """
    Test loading a pretrained YOLO model with weights.
    """
    mock_yolo_class, mock_model_instance = mock_yolo
    config_path = Path("mock_config.yaml")
    weights_path = Path("mock_weights.pt")
    model = load_pretrained_model(config_path, weights_path)
    mock_yolo_class.assert_called_once_with(weights_path)  
    assert model == mock_model_instance 


def test_load_pretrained_model_without_weights(mock_yolo):
    """
    Test loading a YOLO model without weights.
    """
    mock_yolo_class, mock_model_instance = mock_yolo
    config_path = Path("mock_config.yaml")
    model = load_pretrained_model(config_path)
    mock_yolo_class.assert_called_once_with(config_path)  
    assert model == mock_model_instance  


@pytest.fixture
def mock_path_mkdir():
    """
    A pytest fixture to mock the Path.mkdir method.
    """
    with patch("src.pv_defection_classification.model.Path.mkdir") as mock_mkdir:
        yield mock_mkdir


def test_save_model(mock_path_mkdir):
    """
    Test saving a YOLO model to the specified path.
    """
    mock_model_instance = MagicMock()
    output_path = Path("mock_output/model.pt")
    save_model(mock_model_instance, output_path)
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)  
    mock_model_instance.save.assert_called_once_with(str(output_path))  
