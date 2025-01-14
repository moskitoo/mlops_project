import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import typer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
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
            "filename": filename,
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

        annotations[img_new_anns["filename"] + filesize] = img_new_anns

    save_annotations(annotations, annotations_output_path)


def preprocess(
    raw_data_path: Path = Path("data/raw/pv_defection/dataset_2"),
    output_folder: Path = Path("data/processed/pv_defection"),
    split_ratio: float = 0.8,
) -> None:
    """Process raw data and save it to processed directory in VGG format."""
    print("Preprocessing data...")

    train_folder = output_folder / "train"
    val_folder = output_folder / "val"
    train_annotations_path = train_folder / "via_region_data.json"
    val_annotations_path = val_folder / "via_region_data.json"

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
    val_files = image_files[split_index:]

    # Process train and val sets
    process_files(train_files, source_images_dir, source_annotations_dir, train_folder, train_annotations_path)
    process_files(val_files, source_images_dir, source_annotations_dir, val_folder, val_annotations_path)

    print("Preprocessing complete!")


def get_data_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["x"]
            py = anno["y"]
            width = anno["width"]
            height = anno["height"]

            obj = {
                "bbox": [px, py, width, height],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_metadata(data_path: Path = "data/processed/pv_defection/"):
    for d in ["train", "val"]:
        DatasetCatalog.register("pv_module_" + d, lambda d=d: get_data_dicts(data_path + d))
        MetadataCatalog.get("pv_module_" + d).set(thing_classes=["pv_module"])
    return DatasetCatalog, MetadataCatalog


if __name__ == "__main__":
    typer.run(preprocess)
