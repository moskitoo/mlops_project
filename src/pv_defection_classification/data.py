import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

import cv2
import typer

# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.structures import BoxMode
from torch.utils.data import Dataset
from utils.dataset_yaml_setup import create_dataset_config


class MyDataset(Dataset):
    """Custom dataset class for handling raw data."""

    def __init__(self, raw_data_path: Path) -> None:
        """
        Initialize the dataset with the raw data path.

        Args:
            raw_data_path (Path): Path to the raw data directory.
        """
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Dataset length calculation is not implemented.")

    def __getitem__(self, index: int):
        """Return a sample from the dataset at the specified index."""
        raise NotImplementedError("Dataset item retrieval is not implemented.")

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        raise NotImplementedError("Dataset preprocessing is not implemented.")


def save_annotations(annotations: Dict[str, Dict], output_path: Path) -> None:
    """Save annotations to a JSON file.

    Args:
        annotations (Dict[str, Dict]): Dictionary of annotations to save.
        output_path (Path): Path to save the JSON file.
    """
    # with open(output_path, "w", encoding="utf-8") as file:
    #     json.dump(annotations, file, indent=4)
    with open(output_path, "w") as f:
        for annotation in annotations:
            # Format each row: first number as integer, rest as floats with 6 decimal places
            line = f"{int(annotation[0])} {' '.join(f'{x:.6f}' for x in annotation[1:])}\n"
            f.write(line)


def process_files(
    files: List[str],
    source_images_dir: Path,
    source_annotations_dir: Path,
    target_dir: Path,
    processed_annotations_dir: Path,
) -> None:
    """Process image files and generate annotations in normalized format.

    Args:
        files (List[str]): List of image file names to process.
        source_images_dir (Path): Directory containing source images.
        source_annotations_dir (Path): Directory containing source annotations.
        target_dir (Path): Target directory to save processed files.
        processed_annotations_dir (Path): Path to save the processed annotation files.
    """
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(processed_annotations_dir, exist_ok=True)

    total_modules, total_defected = 0, 0

    for filename in files:
        source_image_path = source_images_dir / filename
        destination_image_path = target_dir / filename
        shutil.copy(source_image_path, destination_image_path)

        img = Image.open(source_image_path)
        img_width, img_height = img.size

        json_path = source_annotations_dir / f"{Path(filename).stem}.json"
        with open(json_path, encoding="utf-8") as file:
            img_annotations = json.load(file)

        instances = img_annotations.get("instances", [])

        annotations: List[List[int | float]] = []

        for instance in instances:
            annotation: List[int | float] = []

            is_defected = bool(instance.get("defected_module", False))
            annotation.append(int(is_defected))
            annotation.append(instance["center"]["x"] / img_width)
            annotation.append(instance["center"]["y"] / img_height)
            annotation.append(abs(instance["corners"][0]["x"] - instance["corners"][1]["x"]) / img_width)
            annotation.append(abs(instance["corners"][0]["y"] - instance["corners"][3]["y"]) / img_height)

            total_modules += 1
            if is_defected:
                total_defected += 1

            annotations.append(annotation)

        processed_annotations_file_path = processed_annotations_dir / f"{Path(filename).stem}.txt"

        save_annotations(annotations, processed_annotations_file_path)
    print(f"Processed {len(files)} files. Total modules: {total_modules}, Defected: {total_defected}")




def preprocess(
    raw_data_path: Path = Path("data/raw/pv_defection/dataset_2"),
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
    random.seed(random_seed)

    train_folder = output_folder / "images" / "train"
    val_folder = output_folder / "images" / "val"
    train_annotations_path = output_folder / "labels" / "train"
    val_annotations_path = output_folder / "labels" / "val"

    source_images_dir = raw_data_path / "images"
    source_annotations_dir = raw_data_path / "annotations"

    image_files = [
        f
        for f in os.listdir(source_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))
    ]

    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_files, val_files = image_files[:split_index], image_files[split_index:]

    process_files(train_files, source_images_dir, source_annotations_dir, train_folder, train_annotations_path)
    process_files(val_files, source_images_dir, source_annotations_dir, val_folder, val_annotations_path)

    directory_name = output_folder.name
    yaml_file_path = output_folder / f"{directory_name}.yaml"

    create_dataset_config(yaml_file_path)


def get_data_dicts(img_dir: Path) -> List[Dict]:
    """Load dataset dictionaries from VGG annotation format.

    Args:
        img_dir (Path): Path to the image directory.

    Returns:
        List[Dict]: List of dataset dictionaries.
    """
    json_file = img_dir / "via_region_data.json"
    with open(json_file, encoding="utf-8") as file:
        imgs_anns = json.load(file)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        filename = img_dir / v["filename"]
        height, width = cv2.imread(str(filename)).shape[:2]

        record = {
            "file_name": str(filename),
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": [
                {
                    "bbox": [
                        anno["shape_attributes"]["x"],
                        anno["shape_attributes"]["y"],
                        anno["shape_attributes"]["width"],
                        anno["shape_attributes"]["height"],
                    ],
                    # "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": anno["region_attributes"]["class"],
                }
                for anno in v["regions"].values()
            ],
        }
        dataset_dicts.append(record)
    return dataset_dicts


# def get_metadata(data_path: Path = Path("data/processed/pv_defection")) -> Tuple[DatasetCatalog, MetadataCatalog]:
#     """Register datasets and return catalogs.

#     Args:
#         data_path (Path): Path to the processed data.

#     Returns:
#         Tuple[DatasetCatalog, MetadataCatalog]: Registered catalogs.
#     """
#     for split in ["train", "val"]:
#         path = data_path / split
#         DatasetCatalog.register(f"pv_module_{split}", lambda split=split: get_data_dicts(path))
#         MetadataCatalog.get(f"pv_module_{split}").set(thing_classes=["pv_module", "pv_module_defected"])
#     return DatasetCatalog, MetadataCatalog


if __name__ == "__main__":
    typer.run(preprocess)
