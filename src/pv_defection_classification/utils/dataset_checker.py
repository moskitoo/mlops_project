from pathlib import Path
from typing import List, Tuple

import typer


def verify_dataset_structure(dataset_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify that a dataset follows the expected structure.

    Args:
        dataset_dir (Path): Path to the dataset directory

    Returns:
        tuple: (is_valid, list of errors)
    """
    errors = []

    # Check for required top-level items
    expected_items = [dataset_dir.name + ".yaml", "labels", "images"]
    present_items = [d.name for d in dataset_dir.iterdir()]
    missing_items = set(expected_items) - set(present_items)

    if missing_items:
        errors.append(f"Missing required items: {missing_items}")
        return False, errors

    # Check for required subdirectories in both images and labels
    for dir_name in ["images", "labels"]:
        dir_path = dataset_dir / dir_name
        if not dir_path.is_dir():
            errors.append(f"{dir_name} is not a directory")
            continue

        # Check for train and val subdirectories
        expected_subdirs = {"train", "val"}
        present_subdirs = {d.name for d in dir_path.iterdir() if d.is_dir()}
        missing_subdirs = expected_subdirs - present_subdirs

        if missing_subdirs:
            errors.append(f"Missing {missing_subdirs} directories in {dir_name}/")

    # Check that image and label files match
    for split in ["train", "val"]:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split

        if not (image_dir.exists() and label_dir.exists()):
            continue

        image_files = {f.stem for f in image_dir.glob("*.jpg")}
        label_files = {f.stem for f in label_dir.glob("*.txt")}

        # Check for images without labels
        images_without_labels = image_files - label_files
        if images_without_labels:
            errors.append(f"Images without labels in {split}: {images_without_labels}")

        # Check for labels without images
        labels_without_images = label_files - image_files
        if labels_without_images:
            errors.append(f"Labels without images in {split}: {labels_without_images}")

    return len(errors) == 0, errors


def count_defected_modules_in_file(file_path: Path) -> int:
    """
    Count number of defected modules in a single label file.

    Args:
        file_path (Path): Path to the label file with format:
        class x_center y_center width height (all values in range 0-1)
        where class 0 = working module, 1 = defected module

    Returns:
        int: Number of defected modules found in the file
    """
    try:
        defected_count = 0
        with open(file_path, "r") as f:
            for line in f:
                # Split line into values and convert first value (class) to int
                values = line.strip().split()
                if not values:  # Skip empty lines
                    continue

                class_id = int(values[0])

                # Validate format
                if len(values) != 5:
                    print(f"Warning: Invalid format in line '{line.strip()}' in {file_path}")
                    continue

                # Validate values are in range 0-1
                try:
                    x_center, y_center, width, height = map(float, values[1:])
                    if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                        print(f"Warning: Values out of range [0,1] in line '{line.strip()}' in {file_path}")
                        continue
                except ValueError:
                    print(f"Warning: Invalid numeric values in line '{line.strip()}' in {file_path}")
                    continue

                if class_id == 1:  # Defected module
                    defected_count += 1

        return defected_count

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return 0


def check_dataset(directory: Path) -> Tuple[int, int, int]:
    """
    Count defected modules in all label files in a directory.

    Args:
        directory (Path): Path to the dataset directory

    Returns:
        tuple: (total_defected, total_files_processed, files_with_errors)
    """
    total_defected = 0
    files_processed = 0
    files_without_defects = 0

    dir_path = Path(directory)

    # Look for label files in the labels directory and its subdirectories
    labels_dir = dir_path / "labels"
    if not labels_dir.exists():
        return 0, 0, 0

    for label_file in labels_dir.rglob("*.txt"):
        files_processed += 1
        defected_count = count_defected_modules_in_file(label_file)

        if defected_count == 0:
            files_without_defects += 1

        total_defected += defected_count

    return total_defected, files_processed, files_without_defects


def main(directory: Path = Path("data/processed")) -> None:
    """Check dataset structure and count defected modules in the processed data directory.
    This function verifies the structure of datasets in the specified directory and provides statistics
    about defected modules in each dataset.
    Args:
        directory (Path, optional): Path to the processed data directory. Defaults to Path("data/processed").
    Returns:
        None. Prints validation results and statistics to console:
            - Dataset structure validation status
            - Error messages for invalid datasets
            - Total number of defected modules
            - Total files processed
            - Number and percentage of files without defects
    """
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return

    datasets = [d for d in directory.iterdir() if d.is_dir()]

    if not datasets:
        print(f"No datasets found in {directory}")
        return

    for dataset in datasets:
        print(f"\nChecking dataset: {dataset.name}")
        is_valid, errors = verify_dataset_structure(dataset)

        if not is_valid:
            print("❌ Dataset structure is invalid:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✅ Dataset structure is valid")

        # # You can still keep the defected modules counting if needed
        total_defected, files_processed, files_without_defects = check_dataset(dataset)
        print(f"Total defected modules: {total_defected}")
        print(f"Files processed: {files_processed}")
        print(f"Files without defects: {files_without_defects} ({int(files_without_defects / files_processed * 100)}%)")


if __name__ == "__main__":
    typer.run(main)
