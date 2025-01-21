import os
from pathlib import Path
from ultralytics import YOLO, settings
from google.cloud import storage
from dotenv import load_dotenv
import typer

load_dotenv()

settings.update({"wandb": True})
batch_size = 32
learning_rate = 0.01
max_iteration = 2
output_dir = Path("models")
run_folder_name = "current_run"
run_folder = output_dir / run_folder_name  
gcp_bucket_name = "yolo_model_storage"
gcp_model_name = "pv_defection_classification_model.pt"

def upload_best_model_to_gcp(local_best_model, bucket_name, model_name):
    """
    Upload the best model to a GCP bucket.
    Args:
        local_best_model (Path): Path to the local best model file.
        bucket_name (str): Name of the GCP bucket.
        model_name (str): Name to save the model in GCP.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)
    blob.upload_from_filename(str(local_best_model))
    print(f"Uploaded {local_best_model} to GCP bucket {bucket_name} as {model_name}")

def train_model(
    batch_size: int = batch_size,
    learning_rate: float = learning_rate,
    max_iteration: int = max_iteration,
    data_path: Path = Path("data/processed/pv_defection/pv_defection.yaml"),
    enable_wandb: bool = True,
):
    """
    Train a YOLO model and perform validation, ensuring consistent output folder.
    Args:
        batch_size: int, size of training batch
        learning_rate: float, initial learning rate
        max_iteration: int, maximum number of iterations
        data_path: Path, path to the YOLO dataset configuration file
        enable_wandb: bool, whether to enable W&B logging.
    """
    os.makedirs(run_folder, exist_ok=True)  
    print(f"Output directory: {run_folder}")

    model = YOLO("yolo11n.yaml")

    print("Starting training...")
    results = model.train(
        data=data_path,
        epochs=max_iteration,
        batch=batch_size,
        lr0=learning_rate,
        project=str(output_dir), 
        name=run_folder_name,  
        save=True,
        verbose=True,
    )

    best_model_path = run_folder / "weights" / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(f"'best.pt' not found at {best_model_path}")

    print(f"Training complete. Best model saved at {best_model_path}")
    print(f"Uploading best model to GCP bucket: {gcp_bucket_name}")
    upload_best_model_to_gcp(best_model_path, gcp_bucket_name, gcp_model_name)
    print("Model successfully uploaded to GCP.")

    print("Training process completed.")


if __name__ == "__main__":
    typer.run(train_model)
