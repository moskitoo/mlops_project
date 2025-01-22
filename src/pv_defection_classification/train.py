from pathlib import Path
import typer
from dotenv import load_dotenv
from google.cloud import storage
from utils.update_yolo_settings import update_yolo_settings
import wandb
import os

# Ensure the .env file has the wandb API key and the path to the GCP credentials
load_dotenv()

# # Update Ultralytics settings for wandb
# settings.update({"wandb": True})

BATCH_SIZE = 2
LEARNING_RATE = 0.01
MAX_ITERATION = 3
OUTPUT_DIR = Path("models")
RUN_FOLDER_NAME = "current_run"
RUN_FOLDER = OUTPUT_DIR / RUN_FOLDER_NAME
GCP_BUCKET_NAME = "test-pv-2"
GCP_MODEL_NAME = "pv_defection_classification_model.pt"

# Download dataset from GS bucket
def download_dataset(bucket_name: str, source_prefix: str, local_dir: str):
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GCP credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_prefix)
    for blob in blobs:
        relative_path = blob.name[len(source_prefix):].lstrip("/")  
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file
        #print(f"Downloading gs://{bucket_name}/{blob.name} -> {local_path}")
        blob.download_to_filename(local_path)

bucket_name = "test-pv-2"
gcs_path = "data/processed/pv_defection"
local_path = "data/processed/pv_defection/"
yaml_path = "data/processed/pv_defection/pv_defection.yaml"  

download_dataset(bucket_name, gcs_path, local_path)

def upload_best_model_to_gcp(local_best_model: Path, bucket_name: str, model_name: str):
    """
    Upload the best model to a GCP bucket.

    Args:
        local_best_model (Path): Path to the local best model file.
        bucket_name (str): Name of the GCP bucket.
        model_name (str): Name to save the model in GCP.
    """
    try:
        print(f"Uploading {local_best_model} to GCP bucket {bucket_name}...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_name)
        blob.upload_from_filename(str(local_best_model))
        print(f"Uploaded {local_best_model} to GCP bucket {bucket_name} as {model_name}")
    except Exception as e:
        print(f"Failed to upload model to GCP: {e}")
        raise


BASE_DIR = Path(__file__).resolve().parent
#print(BASE_DIR)

def train_model(
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_iteration: int = MAX_ITERATION,
    data_path = BASE_DIR / "data" / "processed" / "pv_defection" / "pv_defection.yaml",
    #data_path: Path = Path(yaml_path),
    enable_wandb: bool = True,
):
    """
    Train a YOLO model and perform validation, ensuring consistent output folder.
    Train a YOLO model and perform validation, ensuring consistent output folder.

    Args:
        batch_size (int): Size of training batch.
        learning_rate (float): Initial learning rate.
        max_iteration (int): Maximum number of iterations.
        data_path (Path): Path to the YOLO dataset configuration file.
        enable_wandb (bool): Whether to enable W&B logging.
    """
    try:
        update_yolo_settings(Path(data_path))

        from ultralytics import settings
        from model import load_pretrained_model, save_model

        # Update Ultralytics settings for wandb
        settings.update({"wandb": True})

        # Configure W&B
        wandb.login()
        wandb.init(project="pv_defection_classification", entity="hndrkjs-danmarks-tekniske-universitet-dtu")

        # Load YOLO model
        print("Initializing YOLO model...")
        model = load_pretrained_model(config_path=Path("yolo11n.yaml"))
        # Load YOLO model
        print("Initializing YOLO model...")
        model = load_pretrained_model(config_path=Path("yolo11n.yaml"))

        # Start training
        print("Starting training...")
        model.train(
            data=str(data_path),
            epochs=max_iteration,
            batch=batch_size,
            lr0=learning_rate,
            project=str(OUTPUT_DIR),
            name=RUN_FOLDER_NAME,
            save=True,
            verbose=True,
        )

        # Save the trained model
        best_model_path = RUN_FOLDER / "weights" / "best.pt"
        if not best_model_path.exists():
            raise FileNotFoundError(f"'best.pt' not found at {best_model_path}")

        print(f"Training complete. Best model saved at {best_model_path}")

        # Save the model
        save_model(model, best_model_path)
        # Save the trained model
        best_model_path = RUN_FOLDER / "weights" / "best.pt"
        if not best_model_path.exists():
            raise FileNotFoundError(f"'best.pt' not found at {best_model_path}")

        print(f"Training complete. Best model saved at {best_model_path}")

        # Save the model
        save_model(model, best_model_path)

        # Upload the best model to GCP
        if enable_wandb:
            print(f"Uploading best model to GCP bucket: {GCP_BUCKET_NAME}")
            upload_best_model_to_gcp(best_model_path, GCP_BUCKET_NAME, GCP_MODEL_NAME)

        print("Model successfully uploaded to GCP.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise
        # Upload the best model to GCP
        if enable_wandb:
            print(f"Uploading best model to GCP bucket: {GCP_BUCKET_NAME}")
            upload_best_model_to_gcp(best_model_path, GCP_BUCKET_NAME, GCP_MODEL_NAME)

        print("Model successfully uploaded to GCP.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    typer.run(train_model)
