from datetime import datetime
from pathlib import Path
import wandb
import typer
from dotenv import load_dotenv

from model import load_pretrained_model, save_model

load_dotenv()

batch_size = 32
learning_rate = 0.01
max_iteration = 100
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path("models")
entity_name = "hndrkjs-danmarks-tekniske-universitet-dtu"  

def train_model(
    batch_size: int = batch_size,
    learning_rate: float = learning_rate,
    max_iteration: int = max_iteration,
    optimizer: str = "AdamW", 
    data_path: Path = Path("data/processed/pv_defection/pv_defection.yaml"),
    model_config: Path = Path("yolo11n.yaml"),
    enable_wandb: bool = True,
):
    """
    Train a YOLO model.

    Args:
        batch_size (int): Size of training batch.
        learning_rate (float): Initial learning rate.
        max_iteration (int): Maximum number of iterations.
        optimizer (str): Optimizer to use for training.
        data_path (Path): Path to the YOLO dataset configuration file.
        model_config (Path): Path to the YOLO model configuration file.
        enable_wandb (bool): Whether to enable W&B logging.
    """
    run_folder = output_dir / timestamp
    weights_folder = run_folder / "weights"
    weights_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_folder}")

    # Initialize W&B
    if enable_wandb:
        wandb.init(
            project="models",
            entity=entity_name,
            name=f"BS{batch_size}_LR{learning_rate}_OPT{optimizer}_{timestamp}",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "epochs": max_iteration,
            },
        )

    # Load YOLO model
    model = load_pretrained_model(config_path=model_config)

    # Training
    print("Starting training...")
    model.train(
        data=data_path,
        epochs=max_iteration,
        batch=batch_size,
        lr0=learning_rate,
        optimizer=optimizer,
        project=str(output_dir),
        name=f"pv_defection_model_{timestamp}",
        save=True,
    )

    # Save model
    save_model(model, weights_folder / "best.pt")
    print("Training complete. Model checkpoints are saved in:", run_folder)

    if enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    typer.run(train_model)
