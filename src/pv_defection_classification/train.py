import os
from datetime import datetime
from pathlib import Path

from balloon_db import get_baloon_metadata
from detectron2.engine import DefaultTrainer
from model import *

from data import get_metadata

import typer

# default values
batch_size = 2
learning_rate = 0.00025
max_iteration = 300
number_of_classes = 2

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"models/{timestamp}"


def train_model(
    batch_size: int = batch_size,
    learning_rate: float = learning_rate,
    max_iteration: int = max_iteration,
    number_of_classes: int = number_of_classes,
    data_path: Path = "data/processed/pv_defection/"
):
    """
    this function creates the model and trains the model

    Args:
        batch_size: int, size of training batch
        learning_rate: float, initial learning rate
        max_iteration: int, maximum number of iterations
        number_of_classes: int, number of classes (no +1 needed for background)

    Returns:
        no return, logs and checkpoints are stored under /models/<timestamp>

    """
    model = get_model(number_of_classes)
    model.SOLVER.IMS_PER_BATCH = batch_size
    model.SOLVER.BASE_LR = learning_rate
    model.SOLVER.MAX_ITER = max_iteration
    model.SOLVER.STEPS = []
    model.OUTPUT_DIR = output_dir
    os.makedirs(model.OUTPUT_DIR, exist_ok=True)
    MetadataCatalog, DatasetCatalog = get_metadata(data_path)
    trainer = DefaultTrainer(model)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    typer.run(train_model)
