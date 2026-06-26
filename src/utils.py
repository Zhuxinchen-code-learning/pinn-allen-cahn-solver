from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML experiment configuration."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
