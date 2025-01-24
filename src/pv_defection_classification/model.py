from ultralytics import YOLO
from pathlib import Path
from loguru import logger


def load_pretrained_model(config_path: Path, weights_path: Path = None):
    """
    Load a YOLO model, optionally with pretrained weights.

    Args:
        config_path (Path): Path to the YOLO model configuration file.
        weights_path (Path, optional): Path to pretrained weights.

    Returns:
        YOLO: The initialized YOLO model.
    """
    if weights_path:
        logger.info(f"Loading pretrained weights from {weights_path}...")
        model = YOLO(weights_path)
    else:
        logger.info(f"Loading model from configuration {config_path}...")
        model = YOLO(config_path)

    return model


def save_model(model, output_path: Path):
    """
    Save the YOLO model to the specified path.

    Args:
        model (YOLO): The YOLO model to save.
        output_path (Path): Path to save the model.
    """
    logger.info(f"Saving model to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
