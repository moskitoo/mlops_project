import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import json
import os
import random

import cv2
import numpy as np
from pathlib import Path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from data import get_metadata, get_data_dicts
from balloon_db import get_balloon_dicts

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

#pv_model
model_name = "pv_2025-01-15_21-14-02"
model_name = "merged_pv_2025-01-15_23-44-21"
model_name = "dataset_2"
model_name = "dataset_1"

#balloon model
# model_name = "2025-01-13_21-58-37"


dataset_dicts = get_data_dicts("data/processed/pv_defection/val")
# dataset_dicts = get_balloon_dicts("data/raw/balloon/val")


cfg.MODEL.WEIGHTS = f"models/{model_name}/model_final.pth"  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# pv_metadata = MetadataCatalog.get("pv_module_train")
dataset_catalog, metadata_catalog = get_metadata(Path("data/processed/pv_defection"))
pv_metadata = metadata_catalog.get("pv_module_train")
dataset_dicts = get_data_dicts("data/processed/pv_defection/val")


dataset_dicts = [{"file_name": "data/processed/pv_defection/val/20180630_154039.jpg"},
                 {"file_name": "data/processed/pv_defection/val/20180630_154201.jpg"},
                 {"file_name": "data/processed/pv_defection/val/033R.jpg"}]

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
