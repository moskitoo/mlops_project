import typer
from pathlib import Path
from model import load_pretrained_model


def evaluate_model(
    model_dir: Path = typer.Argument(..., help="Path to the model directory (e.g., models/2025-01-21_10-17-52)"),
    data_path: Path = Path("data/processed/pv_defection/pv_defection.yaml"),
):
    """
    Evaluate a YOLO model.

    Args:
        model_dir (Path): Path to the directory containing model weights.
        data_path (Path): Path to the YOLO dataset configuration file.
    """
    weights_path = model_dir / "weights/best.pt"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    model = load_pretrained_model(config_path=None, weights_path=weights_path)

    print(f"Evaluating model with weights from: {weights_path}")
    results = model.val(data=data_path)

    print("Available metrics:", dir(results))

    if hasattr(results, "results_dict"):
        print("Results dictionary:", results.results_dict)
    elif hasattr(results, "maps"):
        print("Mean average precision (mAP):", results.maps)
    else:
        print("No recognized metrics available.")

    print("Evaluation completed.")


if __name__ == "__main__":
    typer.run(evaluate_model)
