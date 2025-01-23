from pathlib import Path
import typer

def verify_data_path(path: Path = Path("data/processed/pv_defection_gcp_mounted/pv_defection_gcp_mounted.yaml")):
    path_obj = Path(path)

    # Check if the original path exists
    if path_obj.exists():
        print(f"Path verified: {path_obj}")
        return str(path_obj)

    parts = path_obj.parts

    # Remove leading slash for consistent processing
    if parts[0] in ("/", "\\"):
        parts = parts[1:]

    # If path does not start with 'gcs', raise an error
    if parts[0] != "gcs":
        raise FileNotFoundError(f"Error: The path '{path}' does not exist and does not start with 'gcs'.")

    # Remove 'gcs' and the next directory
    modified_path = Path(*parts[2:])

    # Check if the modified path exists
    if modified_path.exists():
        print(f"Path verified: {modified_path}")
        return str(modified_path)

    # If the modified path still doesn't exist, raise an error
    raise FileNotFoundError(f"Error: Neither the original path '{path}' nor the modified path '{modified_path}' exist.")

if __name__ == "__main__":
    typer.run(verify_data_path)