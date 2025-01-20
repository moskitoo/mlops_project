import os
from datetime import datetime
from pathlib import Path
import wandb
from ultralytics import YOLO, settings
from dotenv import load_dotenv
import typer

load_dotenv()

settings.update({"wandb": True})
batch_size = 2
learning_rate = 0.00025
max_iteration = 100
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path("models")

sweep_config = {
    "method": "random",  
    "metric": {"name": "val/mAP50", "goal": "maximize"}, 
    "parameters": {
        "batch_size": {"values": [2, 8, 16, 32]},  
        "learning_rate": {"values": [0.0001, 0.00025, 0.001, 0.01]}, 
        "optimizer": {"values": ["SGD", "Adam", "AdamW"]},  
    },
}

def train_model(config=sweep_config):
    """
    Train a YOLO model with a specific configuration for W&B sweeps.
    Args:
        config: dict, configuration for training.
    """
    wandb.init(project="YOLO11n", config=config)
    config = wandb.config
    run_name = f"BS{config.batch_size}_LR{config.learning_rate}_OPT{config.optimizer}"
    wandb.run.name = run_name
    wandb.run.save()

    run_folder = output_dir / timestamp
    os.makedirs(run_folder, exist_ok=True)
    print(f"Output directory: {run_folder}")

    model = YOLO("yolo11n.yaml")

    print("Starting training...")
    results = model.train(
        data="data/processed/pv_defection/pv_defection.yaml",
        epochs=max_iteration,
        batch=config.batch_size,
        lr0=config.learning_rate,
        optimizer=config.optimizer,
        project=str(output_dir),
        name=run_name,
        save=True,
        verbose=True,
    )
    wandb.log({
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "train/box_loss": results.box_loss,
        "train/cls_loss": results.cls_loss,
        "train/dfl_loss": results.dfl_loss,
        "train/epochs": max_iteration,
    })

    print(f"Training complete. Model checkpoints are saved in: {run_folder}")

    print(f"Validating {run_folder / 'weights/best.pt'}...")
    validation_results = model.val(
        model=run_folder / "weights/best.pt",
        data="data/processed/pv_defection/pv_defection.yaml",
    )
    wandb.log({
        "val/precision": validation_results.box.p.tolist(),
        "val/recall": validation_results.box.r.tolist(),
        "val/mAP50": validation_results.box.map50,
        "val/mAP50-95": validation_results.box.map,
    })

    print("Validation completed. Results saved locally.")
    wandb.finish()

def sweep_main():
    """Set up and run W&B sweep."""
    sweep_id = wandb.sweep(sweep_config, project="YOLO11n")  
    wandb.agent(sweep_id, function=train_model) 

if __name__ == "__main__":
    typer.run(sweep_main)
