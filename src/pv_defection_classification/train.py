from pathlib import Path

import typer
from dotenv import load_dotenv
from google.cloud import storage
from utils.update_yolo_settings import update_yolo_settings
from loguru import logger
import wandb
from hydra import initialize,compose
import logging
import sys

from ultralytics import settings
# Ensure the .env file has the wandb API key and the path to the GCP credentials
load_dotenv()
# # Update Ultralytics settings for wandb
# settings.update({"wandb": True})

BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_ITERATION = 100
OUTPUT_DIR = Path("models")
RUN_FOLDER_NAME = "current_run"
RUN_FOLDER = OUTPUT_DIR / RUN_FOLDER_NAME
GCP_BUCKET_NAME = "yolo_model_storage"
GCP_MODEL_NAME = "pv_defection_classification_model.pt"

# Configure W&B
wandb.login()
wandb.init(project="pv_defection_classification", entity="hndrkjs-danmarks-tekniske-universitet-dtu",config = {})



#Define a custom sink to send logs to wandb
__all__ = ["logger"]

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Retrieve the corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Redirect the log message to loguru
        logger.log(level, record.getMessage())

def setup_logging():
    # Clear existing logging handlers
    logging.root.handlers = []
    # Set the root logger level to NOTSET to capture all logs
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET)


def upload_best_model_to_gcp(local_best_model: Path, bucket_name: str, model_name: str):
    """
    Upload the best model to a GCP bucket.

    Args:
        local_best_model (Path): Path to the local best model file.
        bucket_name (str): Name of the GCP bucket.
        model_name (str): Name to save the model in GCP.
    """
    try:
        logger.info(f"Uploading {local_best_model} to GCP bucket {bucket_name}...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_name)
        blob.upload_from_filename(str(local_best_model))
        logger.info(f"Uploaded {local_best_model} to GCP bucket {bucket_name} as {model_name}")
    except Exception as e:
        logger.error(f"Failed to upload model to GCP: {e}")
        raise


def train_model(
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_iteration: int = MAX_ITERATION,
    data_path: Path = Path("data/processed/pv_defection/pv_defection.yaml"),
    enable_wandb: bool = True,
    ctx: typer.Context = None,
):
    """
    Train a YOLO model and perform validation, ensuring consistent output folder.

    Args:
        batch_size (int): Size of training batch.
        learning_rate (float): Initial learning rate.
        max_iteration (int): Maximum number of iterations.
        data_path (Path): Path to the YOLO dataset configuration file.
        enable_wandb (bool): Whether to enable W&B logging.
    """
    setup_logging()
    try:
        if ctx is None or not any(ctx.get_parameter_source(param).name == 'COMMANDLINE' for param in ctx.params):
            logger.info("No arguments were provided for training. Configurations will be loaded from configs/config.yaml")
            use_config = True,  # if True get configs from file

        else:
            logger.info(f"Arguments received for training: {ctx.params}")
            use_config = False

        if use_config:
            # Load configuration using Hydra
            with initialize(config_path="../../configs/", version_base=None):
                config = compose(config_name="config")
                config = dict(config)
        else:
            config = {"batch": batch_size,
                      "lr0": learning_rate,
                      "data": data_path,
                      "epochs": max_iteration,
                      "save": True,
                      "verbose": True,
                      }
        config["project"] = OUTPUT_DIR
        config["name"] = RUN_FOLDER_NAME

        wandb.config.update(config)

        update_yolo_settings(Path(config["data"]))

        # Update Ultralytics settings for wandb
        settings.update({"wandb": True})

        # import yolo after setting default dataset path
        from model import load_pretrained_model, save_model

        # Load YOLO model
        logger.info("Initializing YOLO model...")
        model = load_pretrained_model(config_path=Path("yolo11n.yaml"))

        # Start training
        logger.info("Starting training...")
        model.train(**config)

        # Save the trained model
        best_model_path = RUN_FOLDER / "weights" / "best.pt"
        if not best_model_path.exists():
            raise FileNotFoundError(f"'best.pt' not found at {best_model_path}")

        logger.info(f"Training complete. Best model saved at {best_model_path}")

        # Save the model
        save_model(model, best_model_path)

        # Upload the best model to GCP
        if enable_wandb:
            logger.info(f"Uploading best model to GCP bucket: {GCP_BUCKET_NAME}")
            upload_best_model_to_gcp(best_model_path, GCP_BUCKET_NAME, GCP_MODEL_NAME)

        logger.info("Model successfully uploaded to GCP.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    typer.run(train_model)
