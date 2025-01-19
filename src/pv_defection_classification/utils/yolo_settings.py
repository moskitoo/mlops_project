#!/usr/bin/env python3
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_yolo_settings(data_path: str = None) -> None:
    """Update the datasets directory in YOLO settings file."""
    try:
        home_path = os.environ.get("HOME")
        if not home_path:
            raise Exception("HOME environment variable not set")

        settings_path = Path(home_path) / ".config" / "Ultralytics" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(settings_path, "r") as file:
                settings = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            settings = {}

        settings["datasets_dir"] = data_path or os.path.join(os.getcwd(), "data")
        logger.info(f"Setting datasets directory to: {settings['datasets_dir']}")

        with open(settings_path, 'w') as file:
            json.dump(settings, file, indent=4)

    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise

if __name__ == "__main__":
    try:
        update_yolo_settings()
        print("Settings updated successfully")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)