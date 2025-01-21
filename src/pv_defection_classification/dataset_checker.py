import json
import os
from pathlib import Path


def count_defected_modules_in_file(file_path):
    """
    Count number of defected modules in a single JSON file.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        int: Number of defected modules found in the file
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "instances" not in data:
            print(f"Warning: {file_path} does not have the expected structure")
            return 0

        defected_count = sum(1 for instance in data["instances"] if instance.get("defected_module", False))

        return defected_count

    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file: {file_path}")
        return 0
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return 0


def count_all_defected_modules(directory):
    """
    Recursively count defected modules in all JSON files in a directory.

    Args:
        directory (str): Path to the directory to search

    Returns:
        tuple: (total_defected, total_files_processed, files_with_errors)
    """
    total_defected = 0
    files_processed = 0
    files_with_errors = []

    # Convert directory to Path object for easier handling
    dir_path = Path(directory)

    # Find all JSON files in directory and subdirectories
    for json_file in dir_path.rglob("*.json"):
        files_processed += 1
        defected_count = count_defected_modules_in_file(json_file)

        if defected_count == 0:
            files_with_errors.append(str(json_file))

        total_defected += defected_count

    return total_defected, files_processed, files_with_errors


def main():
    # Get directory path from user
    directory = input("Enter the directory path to scan: ")

    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return

    total_defected, files_processed, files_with_errors = count_all_defected_modules(directory)

    # Print results
    print("\nResults:")
    print(f"Total number of defected modules found: {total_defected}")
    print(f"Total JSON files processed: {files_processed}")

    if files_with_errors:
        print("\nFiles with potential issues (no defected modules found):")
        for file in files_with_errors:
            print(f"- {file}")


if __name__ == "__main__":
    main()
