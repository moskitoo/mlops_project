

import os
from datetime import datetime
from pathlib import Path

import typer
from ultralytics import YOLO, settings
from dotenv import load_dotenv

load_dotenv()

settings.update({"wandb": True})

batch_size = 32
learning_rate = 0.01
max_iteration = 2
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path("models") 

def train_model(
    batch_size: int = batch_size,
    learning_rate: float = learning_rate,
    max_iteration: int = max_iteration,
    optimizer: str = "AdamW", 
    data_path: Path = Path("data/processed/pv_defection/pv_defection.yaml"),
    enable_wandb: bool = True,  
):
    """
    Train a YOLO model and perform validation.

    Args:
        batch_size: int, size of training batch
        learning_rate: float, initial learning rate
        max_iteration: int, maximum number of iterations
        optimizer: str, the optimizer to use for training
        data_path: Path, path to the YOLO dataset configuration file
        enable_wandb: bool, whether to enable W&B logging.
    """
    run_folder = output_dir / timestamp  
    os.makedirs(run_folder, exist_ok=True)
    print(f"Output directory: {run_folder}")

    model = YOLO("yolo11n.yaml")  

    print("Starting training...")
    results = model.train(
        data=data_path,
        epochs=max_iteration,
        batch=batch_size,
        lr0=learning_rate,
        optimizer=optimizer,  
        project=str(output_dir),  
        name=f"BS{batch_size}_LR{learning_rate}_OPT{optimizer}_{timestamp}",  
        save=True,                
        verbose=True              
    )

    print(f"Training complete. Model checkpoints are saved in: {run_folder}")

    print(f"Validating {run_folder / 'weights/best.pt'}...")
    validation_results = model.val(
        model=run_folder / "weights/best.pt", 
        data=data_path,
    )
    print("Validation completed. Results saved locally.")

    print("Training and validation process completed.")


if __name__ == "__main__":
    typer.run(train_model)
