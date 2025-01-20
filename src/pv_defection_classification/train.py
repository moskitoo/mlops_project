import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import typer
from ultralytics import YOLO
from utils.yolo_settings import update_yolo_settings

import wandb

# default values
batch_size = 2
learning_rate = 0.00025
max_iteration = 2
number_of_classes = 2

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"models/{timestamp}"

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)


def train_model(
    batch_size: int = batch_size,
    learning_rate: float = learning_rate,
    max_iteration: int = max_iteration,
    number_of_classes: int = number_of_classes,
    data_path: Path = "data/processed/pv_defection/pv_defection.yaml",
    # data_path: Path = "data/processed/pv_defection",
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
    # model = YOLO("yolo11n.yaml")

    # model = YOLO("yolo11n.pt")

    # update_yolo_settings(data_path)

    # os.makedirs(output_dir, exist_ok=True)

    # results = model.train(data=data_path, epochs=3)

    # # Evaluate the model's performance on the validation set
    # results = model.val()

    # # Export the model to PyTorch format
    # success = model.export()
    # # Export the model to ONNX format
    # success = model.export(format="onnx")

    with wandb.init(
        project="pv_defection",
        entity="hndrkjs-danmarks-tekniske-universitet-dtu",
        # entity= "amirkfir93-danmarks-tekniske-universitet-dtu",
        name=f"{timestamp}",
        sync_tensorboard=True,
    ) as run:
        artifact = wandb.Artifact(
            type="model",
            name="run-%s-model" % wandb.run.id,
            metadata={
                "format": {"type": "detectron_model"},
                "timestamp": timestamp,
            },
        )

        model = YOLO("yolo11n.yaml")

        model = YOLO("yolo11n.pt")

        update_yolo_settings(data_path)

        os.makedirs(output_dir, exist_ok=True)

        results = model.train(data=data_path, epochs=3,project = output_dir,save = True)

        # Evaluate the model's performance on the validation set
        results = model.val()

        # Export the model to PyTorch format
        # Export the model to ONNX format
        success = model.export(format="onnx")
        output_paths = str(Path(success).parent)
        artifact.add_file(success)
        artifact.add_file(output_paths+"/last.pt")
        run.log_artifact(artifact)
        run.link_artifact(
            artifact=artifact,
            target_path="hndrkjs-danmarks-tekniske-universitet-dtu-org/wandb-registry-mlops final project/trained-detectron2-model",
        )
        run.finish()


if __name__ == "__main__":
    typer.run(train_model)
