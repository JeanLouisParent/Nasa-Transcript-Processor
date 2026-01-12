"""
Mission configuration loader (TOML).

Stores per-mission metadata like PDF file name and page offset.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


@dataclass
class MissionConfig:
    """Parsed mission configuration."""
    mission: int | None = None
    file_name: str = ""
    page_offset: int = 0
    rules: dict[str, Any] = field(default_factory=dict)


def load_mission_config(config_dir: Path, pdf_name: str) -> MissionConfig:
    """
    Load mission configuration for a given PDF name.

    Args:
        config_dir: Directory containing .toml configs
        pdf_name: PDF filename to match against config file_name
    """
    if not config_dir.exists():
        return MissionConfig(file_name=pdf_name)

    pdf_name_lower = pdf_name.lower()
    for path in sorted(config_dir.glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        file_name = str(data.get("file_name", "")).strip()
        if file_name and file_name.lower() == pdf_name_lower:
            return MissionConfig(
                mission=data.get("mission"),
                file_name=file_name,
                page_offset=int(data.get("page_offset", 0) or 0),
                rules=data.get("rules") or {},
            )

    return MissionConfig(file_name=pdf_name)
