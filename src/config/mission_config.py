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
    """
    Data container for mission-specific configuration overrides.
    
    Attributes:
        mission: The mission identifier (e.g., 11).
        file_name: The expected filename of the source PDF.
        page_offset: Numerical offset between PDF pages and logical transcript pages.
        layout_overrides: Dictionary containing mission-specific layout and parsing settings.
    """
    mission: int | None = None
    file_name: str = ""
    page_offset: int = 0
    layout_overrides: dict[str, Any] = field(default_factory=dict)


def load_mission_config(config_dir: Path, pdf_name: str) -> MissionConfig:
    """
    Identifies and loads mission-specific settings based on the PDF filename.

    Args:
        config_dir: Directory containing 'missions.toml'.
        pdf_name: Name of the PDF file being processed.

    Returns:
        A MissionConfig object with overrides for the identified mission, 
        or defaults if no match is found.
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
