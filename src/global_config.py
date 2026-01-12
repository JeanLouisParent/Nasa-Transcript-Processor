"""
Global configuration loader (TOML).

Stores defaults for input/output locations and OCR server.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class GlobalConfig:
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    ocr_url: str = "http://localhost:1234"
    dpi: int = 300
    parallel: bool = True
    workers: int = 4


def load_global_config(config_path: Path) -> GlobalConfig:
    if not config_path.exists():
        return GlobalConfig()

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    input_dir = Path(data.get("input_dir", "input"))
    output_dir = Path(data.get("output_dir", "output"))
    ocr_url = str(data.get("ocr_url", "http://localhost:1234"))
    dpi = int(data.get("dpi", 300))
    parallel = bool(data.get("parallel", True))
    workers = int(data.get("workers", 4))

    return GlobalConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        ocr_url=ocr_url,
        dpi=dpi,
        parallel=parallel,
        workers=workers,
    )
