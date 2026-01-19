"""
Global configuration loader (TOML).

Stores defaults for input/output locations and OCR server.
"""

from dataclasses import dataclass, field
from pathlib import Path

import tomllib


@dataclass
class GlobalConfig:
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    ocr_url: str = "http://localhost:1234"
    ocr_model: str = "qwen/qwen3-vl-4b"
    ocr_prompt: str = "plain"
    dpi: int = 300
    parallel: bool = True
    workers: int = 4
    # Store other config keys to pass to PipelineConfig
    pipeline_defaults: dict[str, object] = field(default_factory=dict)


def load_global_config(config_path: Path) -> GlobalConfig:
    if not config_path.exists():
        return GlobalConfig()

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    # Extract known global fields
    input_dir = Path(data.get("input_dir", "input"))
    output_dir = Path(data.get("output_dir", "output"))
    ocr_url = str(data.get("ocr_url", "http://localhost:1234"))
    ocr_model = str(data.get("ocr_model", "qwen3-vl-4b"))
    ocr_prompt = str(data.get("ocr_prompt", "plain")).lower()
    dpi = int(data.get("dpi", 300))
    parallel = bool(data.get("parallel", True))
    workers = int(data.get("workers", 4))

    # Collect everything else as pipeline defaults
    pipeline_defaults = {
        k: v for k, v in data.items()
        if k not in (
            "input_dir", "output_dir", "ocr_url", "ocr_model",
            "dpi", "parallel", "workers"
        )
    }

    return GlobalConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        ocr_url=ocr_url,
        ocr_model=ocr_model,
        ocr_prompt=ocr_prompt,
        dpi=dpi,
        parallel=parallel,
        workers=workers,
        pipeline_defaults=pipeline_defaults
    )


def load_prompt_config(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return {k: str(v) for k, v in data.items()}
