import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import json
import os
import random

import cv2
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from data import get_metadata, get_data_dicts
from balloon_db import get_balloon_dicts

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

#pv_model
model_name = "2025-01-14_10-35-55"

#balloon model
# model_name = "2025-01-13_21-58-37"


# dataset_dicts = get_data_dicts("data/processed/pv_defection/eval")
dataset_dicts = get_balloon_dicts("data/raw/balloon/val")


cfg.MODEL.WEIGHTS = f"models/{model_name}/model_final.pth"  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

pv_metadata = MetadataCatalog.get("pv_module_train")

for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
        im[:, :, ::-1],
        metadata=pv_metadata,
        scale=0.5,
    )
    print(outputs)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = out.get_image()[:, :, ::-1]
    # Define the file path where you want to save the image
    file_path = f"reports/model_{model_name}_image_{os.path.splitext(os.path.basename(d["file_name"]))[0]}.jpg"
    print(file_path)
    # Save the image
    cv2.imwrite(file_path, img)
