import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import List

import typer
from PIL import Image

if __name__ == "__main__":
    # When run as script, add src to Python path
    import sys

    src_path = str(Path(__file__).parent.parent)
    sys.path.append(src_path)
    from pv_defection_classification.utils.dataset_yaml_setup import create_dataset_config
else:
    # When imported as module
    from .utils.dataset_yaml_setup import create_dataset_config


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dataset_processing.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def save_annotations(annotations: List[List[int | float]], output_path: Path) -> None:
    """Save annotations to a JSON file.

    Args:
        annotations (Dict[str, Dict]): Dictionary of annotations to save.
        output_path (Path): Path to save the JSON file.
    """
    try:
        with open(output_path, "w") as f:
            for annotation in annotations:
                line = f"{int(annotation[0])} {' '.join(f'{x:.6f}' for x in annotation[1:])}\n"
                f.write(line)
        logger.info(f"Successfully saved annotations to {output_path}")
    except Exception as e:
        logger.error(f"Error saving annotations to {output_path}: {str(e)}")
        raise


def process_files(
    files: List[str],
    source_images_dir: Path,
    source_annotations_dir: Path,
    target_dir: Path,
    processed_annotations_dir: Path,
) -> None:
    """Process image files and generate annotations in normalized format.
    Handles corrupted images, invalid JSON, and missing instances.

    Args:
        files (List[str]): List of image file names to process.
        source_images_dir (Path): Directory containing source images.
        source_annotations_dir (Path): Directory containing source annotations.
        target_dir (Path): Target directory to save processed files.
        processed_annotations_dir (Path): Path to save the processed annotation files.
    """
    logger.info(f"Starting to process {len(files)} files")

    try:
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(processed_annotations_dir, exist_ok=True)
        logger.debug(f"Created directories: {target_dir}, {processed_annotations_dir}")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        raise

    total_modules, total_defected = 0, 0
    processed_files = 0

    for idx, filename in enumerate(files, 1):
        logger.debug(f"Processing file {idx}/{len(files)}: {filename}")

        source_image_path = source_images_dir / filename
        destination_image_path = target_dir / filename

        try:
            # Validate and copy image first
            try:
                with Image.open(source_image_path) as img:
                    img.verify()
                    # Reopen image after verify() as it closes the file
                    img = Image.open(source_image_path)
                    img_width, img_height = img.size
                    logger.debug(f"Image dimensions: {img_width}x{img_height}")

                    # Copy valid image
                    shutil.copy(source_image_path, destination_image_path)
            except Exception as img_error:
                logger.error(f"Corrupted or invalid image file {filename}: {str(img_error)}")
                continue

            # Process annotations
            json_path = source_annotations_dir / f"{Path(filename).stem}.json"
            try:
                with open(json_path, encoding="utf-8") as file:
                    img_annotations = json.load(file)
            except (json.JSONDecodeError, FileNotFoundError) as json_error:
                logger.error(f"Invalid or missing annotation file for {filename}: {str(json_error)}")
                # Keep the copied image but don't create annotation file
                continue

            instances = img_annotations.get("instances", [])
            annotations: List[List[int | float]] = []

            for instance in instances:
                try:
                    annotation: List[int | float] = []
                    is_defected = bool(instance.get("defected_module", False))

                    # Validate coordinates before processing
                    if not all(key in instance for key in ["center", "corners"]):
                        continue

                    # Create normalized annotation with bounds checking
                    annotation.append(int(is_defected))
                    annotation.append(max(0.0, min(instance["center"]["x"] / img_width, 1.0)))
                    annotation.append(max(0.0, min(instance["center"]["y"] / img_height, 1.0)))
                    annotation.append(
                        max(0.0, min(abs(instance["corners"][0]["x"] - instance["corners"][1]["x"]) / img_width, 1.0))
                    )
                    annotation.append(
                        max(0.0, min(abs(instance["corners"][0]["y"] - instance["corners"][3]["y"]) / img_height, 1.0))
                    )

                    total_modules += 1
                    if is_defected:
                        total_defected += 1

                    annotations.append(annotation)
                except (KeyError, TypeError, ValueError) as instance_error:
                    logger.warning(f"Skipping invalid instance in {filename}: {str(instance_error)}")
                    continue

            # Create empty annotation file even if there are no instances
            processed_annotations_file_path = processed_annotations_dir / f"{Path(filename).stem}.txt"
            save_annotations(annotations, processed_annotations_file_path)
            processed_files += 1
            logger.debug(f"Processed {filename} with {len(annotations)} annotations")

        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            continue

    logger.info(
        f"Processing complete. "
        f"Files processed: {processed_files}/{len(files)}, "
        f"Total modules: {total_modules}, "
        f"Defected: {total_defected}"
    )


def preprocess(
    raw_data_path: Path = Path("data/raw/dataset_2"),
    output_folder: Path = Path("data/processed/pv_defection"),
    split_ratio: float = 0.8,
    random_seed: int = 42,
) -> None:
    """Split raw data into training and validation sets, then preprocess them.

    Args:
        raw_data_path (Path): Path to the raw data.
        output_folder (Path): Path to save the processed data.
        split_ratio (float): Ratio for splitting training and validation data.
        random_seed (int): Seed for random operations.
    """
    logger.info(f"Starting preprocessing with raw_data_path={raw_data_path}, output_folder={output_folder}")
    logger.info(f"Split ratio: {split_ratio}, Random seed: {random_seed}")

    random.seed(random_seed)

    train_folder = output_folder / "images" / "train"
    val_folder = output_folder / "images" / "val"
    train_annotations_path = output_folder / "labels" / "train"
    val_annotations_path = output_folder / "labels" / "val"

    source_images_dir = raw_data_path / "images"
    source_annotations_dir = raw_data_path / "annotations"

    try:
        image_files = [
            f
            for f in os.listdir(source_images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))
        ]
        logger.info(f"Found {len(image_files)} image files")

        random.shuffle(image_files)
        split_index = int(len(image_files) * split_ratio)
        train_files, val_files = image_files[:split_index], image_files[split_index:]
        logger.info(f"Split data into {len(train_files)} training and {len(val_files)} validation files")

        logger.info("Processing training files...")
        process_files(train_files, source_images_dir, source_annotations_dir, train_folder, train_annotations_path)

        logger.info("Processing validation files...")
        process_files(val_files, source_images_dir, source_annotations_dir, val_folder, val_annotations_path)

        directory_name = output_folder.name
        yaml_file_path = output_folder / f"{directory_name}.yaml"
        create_dataset_config(yaml_file_path)
        logger.info(f"Created dataset config at {yaml_file_path}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    typer.run(preprocess)
