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
    layout_overrides: dict[str, Any] = field(default_factory=dict)


def load_mission_config(config_dir: Path, pdf_name: str) -> MissionConfig:
    """
    Load mission configuration for a given PDF name from missions.toml.

    Args:
        config_dir: Directory containing missions.toml
        pdf_name: PDF filename to match against config file_name
    """
    missions_file = config_dir / "missions.toml"
    if not missions_file.exists():
        return MissionConfig(file_name=pdf_name)

    try:
        data = tomllib.loads(missions_file.read_text(encoding="utf-8"))
    except Exception:
        return MissionConfig(file_name=pdf_name)

    pdf_name_lower = pdf_name.lower()
    missions = data.get("mission", {})

    for mission_id, conf in missions.items():
        conf_file_name = str(conf.get("file_name", "")).strip()
        if conf_file_name and conf_file_name.lower() == pdf_name_lower:
            # Extract known fields
            page_offset = int(conf.get("page_offset", 0) or 0)

            # Collect everything else as layout overrides
            overrides = {
                k: v for k, v in conf.items()
                if k not in ("file_name", "page_offset")
            }

            return MissionConfig(
                mission=int(mission_id),
                file_name=conf_file_name,
                page_offset=page_offset,
                layout_overrides=overrides,
            )

    return MissionConfig(file_name=pdf_name)
