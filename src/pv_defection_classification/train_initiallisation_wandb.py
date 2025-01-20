import os
from datetime import datetime
from pathlib import Path

import typer
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv  

load_dotenv()

# Default values
batch_size = 2
learning_rate = 0.00025
max_iteration = 10
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path("models")  

def train_model(
    batch_size: int = batch_size,
    learning_rate: float = learning_rate,
    max_iteration: int = max_iteration,
    data_path: Path = Path("data/processed/pv_defection/pv_defection.yaml"),
    enable_wandb: bool = False,  
):
    """
    Train a YOLO model and perform validation.

    Args:
        batch_size: int, size of training batch
        learning_rate: float, initial learning rate
        max_iteration: int, maximum number of iterations
        data_path: Path, path to the YOLO dataset configuration file
        enable_wandb: bool, whether to enable W&B logging.
    """
    run_folder = output_dir / timestamp  
    os.makedirs(run_folder, exist_ok=True)
    print(f"Output directory: {run_folder}")

    if enable_wandb:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("W&B API key not found in environment or .env file. Set the 'WANDB_API_KEY' variable.")
        wandb.login(key=wandb_api_key)
        wandb.init(
            project="pv_defection",
            entity="hndrkjs-danmarks-tekniske-universitet-dtu",
            name=f"{timestamp}",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_iteration": max_iteration,
            },
        )

    model = YOLO("yolo11n.yaml")  

    print("Starting training...")
    results = model.train(
        data=data_path,
        epochs=max_iteration,
        batch=batch_size,
        lr0=learning_rate,
        project=str(output_dir),  
        name=timestamp,           
        save=True,                
        verbose=True              
    )

    print(f"Training complete. Model checkpoints are saved in: {run_folder}")

    print(f"Validating {run_folder / 'weights/best.pt'}...")
    validation_results = model.val(
        model=run_folder / "weights/best.pt",  
        data=data_path, )
    print("Validation completed. Results saved locally.")

    if enable_wandb:
        wandb.finish()

    print("Training and validation process completed.")


if __name__ == "__main__":
    typer.run(train_model)
