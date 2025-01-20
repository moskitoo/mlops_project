import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import os
import pytest
from PIL import Image

from src.pv_defection_classification.data import process_files


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

@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
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

@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
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

@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
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

@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
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

@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
def test_preprocess(setup_dirs, monkeypatch):
    # Unpack the directories from setup_dirs
    source_images_dir, source_annotations_dir, _, _ = setup_dirs

    # Create raw data directory structure
    raw_data_path = Path(source_images_dir.parent) / "raw" / "pv_defection" / "dataset_2"
    raw_data_path.mkdir(parents=True, exist_ok=True)

    images_dir = raw_data_path / "images"
    annotations_dir = raw_data_path / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_folder = Path(source_images_dir.parent) / "processed" / "pv_defection"

    # Create sample images and annotations
    image_files = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
    for img_file in image_files:
        # Create image
        img = Image.new("RGB", (100, 100))
        img.save(images_dir / img_file)

        # Create corresponding annotation
        annotation_data = {
            "instances": [
                {
                    "defected_module": True if "1" in img_file else False,
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
        with open(annotations_dir / f"{Path(img_file).stem}.json", "w") as f:
            json.dump(annotation_data, f)

    # Mock the logger
    mock_logger = MagicMock()
    monkeypatch.setattr("src.pv_defection_classification.data.logger", mock_logger)

    # Mock random.seed to ensure consistent test results
    mock_random = MagicMock()
    monkeypatch.setattr("random.seed", mock_random)

    # Import the preprocess function
    from src.pv_defection_classification.data import preprocess

    # Run preprocess with test parameters
    preprocess(raw_data_path=raw_data_path, output_folder=output_folder, split_ratio=0.75, random_seed=42)

    # Verify directories were created
    assert (output_folder / "images" / "train").exists()
    assert (output_folder / "images" / "val").exists()
    assert (output_folder / "labels" / "train").exists()
    assert (output_folder / "labels" / "val").exists()

    # Verify files were split and processed
    train_images = list((output_folder / "images" / "train").glob("*.jpg"))
    val_images = list((output_folder / "images" / "val").glob("*.jpg"))
    assert len(train_images) == 3  # 75% of 4 images
    assert len(val_images) == 1  # 25% of 4 images

    # Verify annotations were created
    train_annotations = list((output_folder / "labels" / "train").glob("*.txt"))
    val_annotations = list((output_folder / "labels" / "val").glob("*.txt"))
    assert len(train_annotations) == 3
    assert len(val_annotations) == 1

    # Verify YAML config was created
    yaml_path = output_folder / "pv_defection.yaml"
    assert yaml_path.exists()

    # Verify logger was called
    assert mock_logger.info.called
    assert mock_random.called

@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
def test_preprocess_error_handling(setup_dirs, monkeypatch):
    source_images_dir, source_annotations_dir, _, _ = setup_dirs

    # Create invalid raw data path
    raw_data_path = Path(source_images_dir.parent) / "nonexistent"
    output_folder = Path(source_images_dir.parent) / "processed" / "pv_defection"

    # Mock the logger
    mock_logger = MagicMock()
    monkeypatch.setattr("src.pv_defection_classification.data.logger", mock_logger)

    # Import the preprocess function
    from src.pv_defection_classification.data import preprocess

    # Test error handling
    with pytest.raises(Exception):
        preprocess(raw_data_path=raw_data_path, output_folder=output_folder, split_ratio=0.8, random_seed=42)

    # Verify error was logged
    mock_logger.error.assert_called()
