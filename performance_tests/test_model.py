import wandb
import os
import time
import torch
from ultralytics import YOLO

def load_model(artifact_path):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(artifact_path)
    artifact.download()
    file_name = artifact.files()[0].name
    return YOLO(file_name)

def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 3, 28, 28))
    end = time.time()
    assert end - start < 1