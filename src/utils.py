from pathlib import Path
import json
import logging
from datetime import datetime


def setup_logging(log_directory="logs"):
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True
    )

    return logging.getLogger("pm25_pipeline")


def ensure_directories(config):
    paths = config["paths"]

    directories = [
        paths["raw"],
        paths["processed"],
        paths["outputs"],
        paths["models"],
        paths["logs"],
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def timestamp():
    return datetime.now().isoformat()


def validate_config(config):
    required_sections = [
        "project",
        "study_area",
        "time",
        "spatial",
        "crs",
        "target",
        "paths",
        "datasets",
        "model",
    ]

    missing = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing:
        raise ValueError(
            f"Missing configuration sections: {missing}"
        )

    if config["spatial"]["coarse_resolution_m"] <= 0:
        raise ValueError("Coarse resolution must be positive.")

    if config["spatial"]["target_resolution_m"] <= 0:
        raise ValueError("Target resolution must be positive.")

    if (
        config["spatial"]["target_resolution_m"]
        >= config["spatial"]["coarse_resolution_m"]
    ):
        raise ValueError(
            "Target resolution must be finer than coarse resolution."
        )

    if config["target"]["variable"] != "PM2.5":
        raise ValueError("Target variable must be PM2.5.")

    return True
