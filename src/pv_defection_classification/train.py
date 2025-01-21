import logging
import os
from datetime import datetime
from pathlib import Path
import typer
from ultralytics import YOLO
from utils.yolo_settings import update_yolo_settings
from hydra import initialize,compose
import wandb

# default values
batch_size = 2
learning_rate = 0.00025


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"models/{timestamp}"

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)


def train_model(
    batch_size: int = batch_size,
    learning_rate: float = learning_rate,
    data_path: Path = "data/processed/pv_defection/pv_defection.yaml",
    epochs: int = 20,
    use_config: bool = True, #if True get configs from file
):
    """
    this function creates the model and trains the model

    Args:
        batch_size: int, size of training batch
        learning_rate: float, initial learning rate
        number_of_classes: int, number of classes (no +1 needed for background)

    Returns:
        no return, logs and checkpoints are stored under /models/<timestamp>

    """

    if use_config:
        # Load configuration using Hydra
        with initialize(config_path="../../configs/", version_base=None):
            config = compose(config_name="config")
            config = dict(config)
    else:
        config = {"batch": batch_size,
                  "lr0": learning_rate,
                  "data": data_path,
                  "epochs": epochs
                  }
    config["project"] = output_dir
    with wandb.init(
        project="pv_defection",
        entity="hndrkjs-danmarks-tekniske-universitet-dtu",
        name=f"{timestamp}",
        sync_tensorboard=True,
        config = config,
    ) as run:
        artifact = wandb.Artifact(
            type="model",
            name="run-%s-model" % wandb.run.id,
            metadata={
                "format": {"type": "detectron_model"},
                "timestamp": timestamp,
            },
        )


        model = YOLO("yolo11n.pt")

        update_yolo_settings(data_path)

        os.makedirs(output_dir, exist_ok=True)

        # model.train(data=data_path, epochs=20,batch = batch_size,
        #             lr0 = learning_rate,project=output_dir, save=True)
        # Evaluate the model's performance on the validation set
        model.train(**config)
        model.val()

        # Export the model to ONNX format
        success = model.export(format="onnx")
        output_paths = str(Path(success).parent)
        artifact.add_file(success)
        artifact.add_file(output_paths + "/last.pt")
        artifact.add_file(output_paths + "/last.pt")
        run.log_artifact(artifact)
        run.link_artifact(
            artifact=artifact,
            target_path="hndrkjs-danmarks-tekniske-universitet-dtu-org/wandb-registry-mlops final project/trained-detectron2-model",
        )
        run.finish()


if __name__ == "__main__":
    typer.run(train_model)
