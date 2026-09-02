"""Load and validate model hyperparameter configuration."""

import json
from pathlib import Path
from typing import Any, Optional

DEFAULT_CONFIG_PATH = Path("config") / "hyperparameters.json"


def load_hyperparameters(path: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Load model hyperparameters from JSON, or return an empty configuration."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Hyperparameter configuration must be a JSON object")

    for model_name, parameters in config.items():
        if not isinstance(model_name, str) or not isinstance(parameters, dict):
            raise ValueError("Hyperparameter configuration must map model names to JSON objects")

    return config


def format_hyperparameters(hyperparameters: Optional[dict]) -> str:
    """Return a stable JSON representation for prompts and experiment logs."""
    return json.dumps(hyperparameters or {}, sort_keys=True, indent=2)
