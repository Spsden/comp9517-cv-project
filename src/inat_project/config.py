from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load one experiment configuration from YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config

