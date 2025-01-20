import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
from src.pv_defection_classification.data import process_files
import pytest

@pytest.fixture
def setup_dirs():
    # Create a temporary directory
    test_dir = tempfile.TemporaryDirectory()
    source_images_dir = Path(test_dir.name) / "source_images"
    source_annotations_dir = Path(test_dir.name) / "source_annotations"
    target_dir = Path(test_dir.name) / "target"
    processed_annotations_dir = Path(test_dir.name) / "processed_annotations"

    # Create directories
    source_images_dir.mkdir(parents=True, exist_ok=True)
    source_annotations_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    processed_annotations_dir.mkdir(parents=True, exist_ok=True)

    yield source_images_dir, source_annotations_dir, target_dir, processed_annotations_dir

    # Cleanup the temporary directory
    test_dir.cleanup()

@pytest.fixture
def create_sample_image(setup_dirs):
    source_images_dir, source_annotations_dir, target_dir, processed_annotations_dir = setup_dirs
    image_path = source_images_dir / "test_image.jpg"
    image = Image.new("RGB", (100, 100))
    image.save(image_path)
    return image_path

@pytest.fixture
def create_sample_annotation(setup_dirs):
    source_images_dir, source_annotations_dir, target_dir, processed_annotations_dir = setup_dirs
    annotation_path = source_annotations_dir / "test_image.json"
    annotation_data = {
        "instances": [
            {
                "defected_module": True,
                "center": {"x": 50, "y": 50},
                "corners": [
                    {"x": 30, "y": 30},
                    {"x": 70, "y": 30},
                    {"x": 70, "y": 70},
                    {"x": 30, "y": 70},
                ],
            }
        ]
    }
    with open(annotation_path, "w") as f:
        json.dump(annotation_data, f)
    return annotation_path

def test_process_files(setup_dirs, create_sample_image, create_sample_annotation, monkeypatch):
    source_images_dir, source_annotations_dir, target_dir, processed_annotations_dir = setup_dirs

    mock_logger = MagicMock()
    monkeypatch.setattr("src.pv_defection_classification.data.logger", mock_logger)

    files = ["test_image.jpg"]
    process_files(
        files,
        source_images_dir,
        source_annotations_dir,
        target_dir,
        processed_annotations_dir,
    )

    # Check if the image was copied to the target directory
    copied_image_path = target_dir / "test_image.jpg"
    assert copied_image_path.exists()

    # Check if the processed annotation file was created
    processed_annotation_path = processed_annotations_dir / "test_image.txt"
    assert processed_annotation_path.exists()

def test_process_files_with_invalid_json(setup_dirs, monkeypatch):
    source_images_dir, source_annotations_dir, target_dir, processed_annotations_dir = setup_dirs
    mock_logger = MagicMock()
    monkeypatch.setattr("src.pv_defection_classification.data.logger", mock_logger)

    # Create test image
    files = ["invalid_json.jpg"]
    image_path = source_images_dir / "invalid_json.jpg"
    image = Image.new("RGB", (100, 100))
    image.save(image_path)

    # Create invalid JSON annotation
    annotation_path = source_annotations_dir / "invalid_json.json"
    with open(annotation_path, "w") as f:
        f.write("invalid json content")

    process_files(
        files, 
        source_images_dir,
        source_annotations_dir,
        target_dir,
        processed_annotations_dir,
    )

    # Image should be copied but no annotation created
    copied_image_path = target_dir / "invalid_json.jpg"
    assert copied_image_path.exists()

    processed_annotation_path = processed_annotations_dir / "invalid_json.txt"
    assert not processed_annotation_path.exists()

    # Check logger error 
    mock_logger.error.assert_called()

def test_process_files_with_missing_instances(setup_dirs, monkeypatch):
    source_images_dir, source_annotations_dir, target_dir, processed_annotations_dir = setup_dirs
    mock_logger = MagicMock()
    monkeypatch.setattr("src.pv_defection_classification.data.logger", mock_logger)

    # Create test files
    files = ["no_instances.jpg"]
    image_path = source_images_dir / "no_instances.jpg"
    image = Image.new("RGB", (100, 100))
    image.save(image_path)

    # Create annotation without instances
    annotation_path = source_annotations_dir / "no_instances.json"
    with open(annotation_path, "w") as f:
        json.dump({"some_key": "some_value"}, f)

    process_files(
        files,
        source_images_dir,
        source_annotations_dir, 
        target_dir,
        processed_annotations_dir,
    )

    # Verify empty annotation file is created
    copied_image_path = target_dir / "no_instances.jpg"
    assert copied_image_path.exists()

    processed_annotation_path = processed_annotations_dir / "no_instances.txt" 
    assert processed_annotation_path.exists()

    with open(processed_annotation_path) as f:
        assert f.read() == ""

def test_process_files_with_corrupt_image(setup_dirs, monkeypatch):
    source_images_dir, source_annotations_dir, target_dir, processed_annotations_dir = setup_dirs
    mock_logger = MagicMock()
    monkeypatch.setattr("src.pv_defection_classification.data.logger", mock_logger)

    # Create corrupt image file
    files = ["corrupt.jpg"]
    image_path = source_images_dir / "corrupt.jpg"
    with open(image_path, "wb") as f:
        f.write(b"corrupt image data")

    process_files(
        files,
        source_images_dir,
        source_annotations_dir,
        target_dir,
        processed_annotations_dir,  
    )

    # Verify no files created
    copied_image_path = target_dir / "corrupt.jpg"
    assert not copied_image_path.exists()

    processed_annotation_path = processed_annotations_dir / "corrupt.txt"
    assert not processed_annotation_path.exists()

    # Check logger error
    mock_logger.error.assert_called()