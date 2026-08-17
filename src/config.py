from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("config.yaml is empty.")

    return config


def get_project_root():
    return PROJECT_ROOT


def get_path(relative_path):
    return PROJECT_ROOT / relative_path


if __name__ == "__main__":
    config = load_config()

    print("=" * 60)
    print("PM2.5 HYPERLOCAL MAPPING")
    print("=" * 60)
    print(f"Project : {config['project']['name']}")
    print(f"Mode    : {config['project']['mode']}")
    print(f"Target  : {config['target']['variable']}")
    print(f"AOI     : {config['study_area']['name']}")
    print("=" * 60)
