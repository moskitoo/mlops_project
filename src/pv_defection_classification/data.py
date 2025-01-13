import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def save_annotations(annotations: Dict, output_path: Path) -> None:
    """Save annotations to the specified output path."""
    with open(output_path, "w") as f:
        json.dump(annotations, f)


def process_files(
    files: List[str],
    source_images_dir: Path,
    source_annotations_dir: Path,
    target_dir: Path,
    annotations_output_path: Path,
) -> None:
    """Process and copy files to the target directory, creating annotations."""
    os.makedirs(target_dir, exist_ok=True)
    annotations = {}

    for filename in files:
        source_image_path = source_images_dir / filename
        destination_image_path = target_dir / filename
        shutil.copy(source_image_path, destination_image_path)

        json_name = source_annotations_dir / (filename.split(".")[0] + ".json")
        with open(json_name) as f:
            img_og_anns = json.load(f)

        instances = img_og_anns["instances"]

        filesize = str(os.path.getsize(destination_image_path))
        img_new_anns = {
            "fileref": "",
            "size": filesize,
            "filename": filename + filesize,
            "base64_img_data": "",
            "file_attributes": {},
            "regions": {},
        }

        for i, instance in enumerate(instances):
            region = {
                "shape_attributes": {
                    "name": "rect",
                    "x": instance["corners"][1]["x"],
                    "y": instance["corners"][1]["y"],
                    "width": abs(instance["corners"][0]["x"] - instance["corners"][1]["x"]),
                    "height": abs(instance["corners"][0]["y"] - instance["corners"][3]["y"]),
                },
                "region_attributes": {},
            }
            img_new_anns["regions"][str(i)] = region

        annotations[img_new_anns["filename"]] = img_new_anns

    save_annotations(annotations, annotations_output_path)


def preprocess(
    raw_data_path: Path = Path("data/raw/pv_defection/dataset_2"),
    output_folder: Path = Path("data/processed/pv_defection"),
    split_ratio: float = 0.8,
) -> None:
    """Process raw data and save it to processed directory in VGG format."""
    print("Preprocessing data...")

    train_folder = output_folder / "train"
    eval_folder = output_folder / "eval"
    train_annotations_path = train_folder / "via_region_data.json"
    eval_annotations_path = eval_folder / "via_region_data.json"

    source_images_dir = raw_data_path / "images"
    source_annotations_dir = raw_data_path / "annotations"

    # Collect all image files
    image_files = [
        filename
        for filename in os.listdir(source_images_dir)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))
    ]

    # Shuffle and split files
    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    eval_files = image_files[split_index:]

    # Process train and eval sets
    process_files(train_files, source_images_dir, source_annotations_dir, train_folder, train_annotations_path)
    process_files(eval_files, source_images_dir, source_annotations_dir, eval_folder, eval_annotations_path)

    print("Preprocessing complete!")


# def preprocess(
#     raw_data_path: Path = "data/raw/pv_defection/dataset_2",
#     output_folder: Path = "data/processed/pv_defection",
#     split_ratio: float = 0.8,
# ) -> None:
#     """Process raw data and save it to processed directory in VGG format."""
#     print("Preprocessing data...")

#     # Create output directories for train and eval
#     train_folder = os.path.join(output_folder, "train")
#     eval_folder = os.path.join(output_folder, "eval")
#     os.makedirs(train_folder, exist_ok=True)
#     os.makedirs(eval_folder, exist_ok=True)

#     raw_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/pv_defection/dataset_2"))

#     # Collect all image files from the raw_data_path
#     image_files = [
#         filename
#         for filename in os.listdir(os.path.join(raw_data_path, "images"))
#         if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))
#     ]

#     # Shuffle the list of files to randomize train/eval split
#     random.shuffle(image_files)

#     # Calculate the split index
#     split_index = int(len(image_files) * split_ratio)

#     # Split the files into train and eval
#     train_files = image_files[:split_index]
#     eval_files = image_files[split_index:]

#     # Move files to their respective directories
#     train_annotations = {}
#     for filename in train_files:
#         source_path = os.path.join(os.path.join(raw_data_path, "images"), filename)
#         destination_path = os.path.join(train_folder, filename)
#         shutil.copy(source_path, destination_path)

#         json_name = filename.split(".")[0] + ".json"
#         json_name = os.path.join(os.path.join(raw_data_path, "annotations"), json_name)
#         with open(json_name) as f:
#             img_og_anns = json.load(f)

#         instances = img_og_anns["instances"]

#         filesize = str(os.path.getsize(destination_path))
#         img_new_anns = {}
#         img_new_anns["fileref"] = ""
#         img_new_anns["size"] = filesize
#         img_new_anns["filename"] = filename + filesize
#         img_new_anns["base64_img_data"] = ""
#         img_new_anns["file_attributes"] = {}
#         img_new_anns["regions"] = {}

#         for i, instance in enumerate(instances):
#             region = {}
#             region["shape_attributes"] = {}
#             region["shape_attributes"]["name"] = "rect"
#             region["shape_attributes"]["x"] = instance["corners"][1]["x"]
#             region["shape_attributes"]["y"] = instance["corners"][1]["y"]
#             region["shape_attributes"]["width"] = abs(instance["corners"][0]["x"] - instance["corners"][1]["x"])
#             region["shape_attributes"]["height"] = abs(instance["corners"][0]["y"] - instance["corners"][3]["y"])

#             region["region_attributes"] = {}

#             img_new_anns["regions"][str(i)] = region

#         train_annotations[img_new_anns["filename"]] = img_new_anns

#     with open(os.path.join(train_folder, "via_region_data.json"), "w") as f:
#         json.dump(train_annotations, f)

#     eval_annotations = {}
#     for filename in eval_files:
#         source_path = os.path.join(os.path.join(raw_data_path, "images"), filename)
#         destination_path = os.path.join(eval_folder, filename)
#         shutil.copy(source_path, destination_path)

#     print("Preprocessing complete!")


if __name__ == "__main__":
    typer.run(preprocess)
