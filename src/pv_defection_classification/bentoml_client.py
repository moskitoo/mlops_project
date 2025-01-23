from pathlib import Path

import bentoml
import cv2
import typer
from PIL import Image


def main(
    input_image: Path = typer.Argument(
        Path("data/processed/pv_defection/images/train/20180630_154039.jpg"),
        help="Path to the input image. Defaults to a predefined image if not provided.",
    ),
    remote_endpoint: bool = typer.Option(False, "--remote", help="Use remote endpoint instead of local one."),
) -> None:
    endpoint_url = (
        "https://bento-service-38375731884.europe-west1.run.app" if remote_endpoint else "http://localhost:3000"
    )

    # Ensure the input image exists
    if not input_image.exists():
        typer.echo(f"Error: The file {input_image} does not exist.", err=True)
        raise typer.Exit(code=1)

    image = Image.open(input_image)

    with bentoml.SyncHTTPClient(endpoint_url) as client:
        resp = client.detect_and_predict(input=image)

        output_path = Path("output.jpg")
        cv2.imwrite(str(output_path), resp)
        typer.echo(f"Output saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
