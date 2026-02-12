"""
Global configuration loader (TOML).

Stores defaults for input/output locations and OCR server.
Uses Pydantic for validation.
"""

from pathlib import Path
import tomllib

from .models import GlobalConfigModel

# Alias for backward compatibility
GlobalConfig = GlobalConfigModel


def load_global_config(config_path: Path) -> GlobalConfig:
    """
    Loads and validates the global configuration from a TOML file.

    Args:
        config_path: Path to the configuration file (usually defaults.toml).

    Returns:
        A validated GlobalConfig object.
    """
    if not config_path.exists():
        return GlobalConfig()

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    # Validation happens here
    return GlobalConfig(**data)


def load_prompt_config(config_path: Path) -> dict[str, str]:
    """
    Loads OCR prompt templates from a TOML file.

    Args:
        config_path: Path to the prompt configuration file.

    Returns:
        A dictionary mapping prompt names to their string templates.
    """
    if not config_path.exists():
        return {}
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return {k: str(v) for k, v in data.items()}
