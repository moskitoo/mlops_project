from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from data import get_metadata, get_data_dicts
import cv2
import random
import os

# Get dataset metadata and dictionary
dataset_catalog, metadata_catalog = get_metadata()
pv_metadata = MetadataCatalog.get("pv_module_train")
dataset_dicts = get_data_dicts("data/processed/pv_defection/train")

# Create output directory if it doesn't exist
output_dir = "reports/verify_dataset"
os.makedirs(output_dir, exist_ok=True)

# Randomly visualize and save images
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=pv_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    img = out.get_image()[:, :, ::-1]

    # Define the file path where you want to save the image
    file_name = os.path.splitext(os.path.basename(d["file_name"]))[0]
    file_path = os.path.join(output_dir, f"{file_name}.jpg")
    print(file_path)

    # Save the image
    cv2.imwrite(file_path, img)
