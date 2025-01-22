import random
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


def select_random_image(dataset_dir: Path) -> str:
    """
    Select a random image from the dataset directory.

    Args:
        dataset_dir (Path): Path to the dataset directory

    Returns:
        str: Path to the randomly selected image or empty string if no images are found
    """
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = []

    for ext in image_extensions:
        image_files.extend(dataset_dir.glob(f"images/train/*{ext}"))
        image_files.extend(dataset_dir.glob(f"images/val/*{ext}"))

    if not image_files:
        return ""

    return str(random.choice(image_files))


def generate_report(directory: Path, report_file: Path) -> None:
    """
    Generate a markdown report for dataset statistics and sample images.

    Args:
        directory (Path): Path to the processed data directory
        report_file (Path): Path to output markdown file

    Returns:
        None
    """
    with open(report_file, "w") as f:
        f.write("# Dataset Statistics Report\n\n")

        datasets = [d for d in directory.iterdir() if d.is_dir()]

        if not datasets:
            f.write("No datasets found.\n")
            return

        for dataset in datasets:
            f.write(f"## Dataset: {dataset.name}\n")

            # Verify dataset structure
            is_valid, errors = verify_dataset_structure(dataset)
            if is_valid:
                f.write("✅ Dataset structure is valid\n\n")
            else:
                f.write("❌ Dataset structure is invalid:\n")
                for error in errors:
                    f.write(f"  - {error}\n")
                f.write("\n")

            # Count defected modules
            total_defected, files_processed, files_without_defects = check_dataset(dataset)
            f.write(f"**Total defected modules:** {total_defected}\n")
            f.write(f"**Files processed:** {files_processed}\n")
            f.write(
                f"**Files without defects:** {files_without_defects} ({int(files_without_defects / files_processed * 100)}%)\n\n"
            )

            # Add a random image from the dataset
            random_image = select_random_image(dataset)
            if random_image:
                f.write(f'![Sample Image]({random_image} "Sample from {dataset.name}")\n\n')

        print(f"Report generated in {report_file}")


def main(directory: Path = Path("data/processed"), report_file: Path = Path("data_statistics.md")) -> None:
    """Check dataset structure, count defected modules, and generate a report."""
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return

    print(f"Generating dataset report for {directory}")
    generate_report(directory, report_file)


if __name__ == "__main__":
    typer.run(main)
