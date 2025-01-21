import yaml
from pathlib import Path

def create_dataset_config(
    output_file="dataset.yaml",
    root_path="./pv_defection",
    train_path="images/train",
    val_path="images/val",
    test_path="",
    class_names={0: "intact", 1: "defected"},
):
    """
    Create a YAML configuration file for the dataset structure.

    Args:
        output_file (str): Name of the output YAML file
        root_path (str): Root directory of the dataset
        train_path (str): Path to training images
        val_path (str): Path to validation images
        test_path (str): Path to test images (optional)
        class_names (dict): Dictionary of class names
    """
    # Define the configuration structure
    config = {"path": root_path, "train": train_path, "val": val_path, "test": test_path, "names": class_names}

    try:
        # Create the YAML file
        with open(output_file, "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        print(f"Successfully created {output_file}")

        # Print the content for verification
        print("\nGenerated YAML content:")
        with open(output_file, "r") as file:
            print(file.read())

    except Exception as e:
        print(f"Error creating YAML file: {e}")


# Run the script
if __name__ == "__main__":
    create_dataset_config()
