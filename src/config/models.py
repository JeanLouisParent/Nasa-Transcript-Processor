"""
Pydantic models for configuration validation.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LexiconConfig(BaseModel):
    """Configuration for the lexicon-based corrections."""
    path: Path = Path("resources/lexicon/apollo11_lexicon.json")
    mission_keywords: list[str] = Field(default_factory=list)


class ParserConfig(BaseModel):
    """Configuration for the transcript parser."""
    header_keywords: list[str] = Field(default_factory=list)
    transition_keywords: list[str] = Field(default_factory=list)
    end_of_tape_keyword: str = "END OF TAPE"
    text_replacements: dict[str, str] = Field(default_factory=dict)


class CorrectorsConfig(BaseModel):
    """Configuration for the post-processing correctors."""
    speaker_ocr_fixes: dict[str, str] = Field(default_factory=dict)
    invalid_location_annotations: dict[str, list[str]] = Field(default_factory=dict)


class GlobalConfigModel(BaseModel):
    """
    Main configuration model for the application.
    Validates input from defaults.toml.
    """
    # I/O
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    state_dir: Path = Path("state")

    # OCR Settings
    ocr_url: str = "http://localhost:1234"
    ocr_model: str = "qwen/qwen3-vl-4b"
    ocr_prompt: str = "plain"
    ocr_timeout: int = 120
    ocr_max_tokens: int = 4096
    ocr_text_column_pass: bool = True
    ocr_dual_pass: bool = True
    ocr_faint_pass: bool = True

    # Processing Settings
    dpi: int = 300
    parallel: bool = True
    workers: int = 4
    timing: bool = True

    # Image Enhancement
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    unsharp_amount: float = 1.5
    unsharp_sigma: float = 1.0
    deskew_angle_threshold: float = 0.1
    deskew_max_angle: float = 10.0

    # Normalization settings (Letter size at 300 DPI)
    target_width: int = 2550
    target_height: int = 3300
    margin_px: int = 75
    output_format: str = "png"

    # Layout Detection
    col2_end: float = 0.30
    min_block_area: int = 1000
    max_block_area_ratio: float = 0.9
    line_kernel_width: int = 50
    line_kernel_height: int = 1
    block_kernel_width: int = 5
    block_kernel_height: int = 10

    # Debug settings
    debug: bool = False

    # Sub-configs
    lexicon: LexiconConfig = Field(default_factory=LexiconConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    correctors: CorrectorsConfig = Field(default_factory=CorrectorsConfig)

    @property
    def pipeline_defaults(self) -> dict[str, Any]:
        """
        Return a dictionary of configuration values to be passed to the pipeline components.
        Excludes top-level CLI/I/O settings.
        """
        # Exclude main fields to mimic old behavior
        exclude = {
            "input_dir", "output_dir", "state_dir",
            "ocr_url", "ocr_model", "ocr_prompt",
            "dpi", "parallel", "workers"
        }
        return self.model_dump(exclude=exclude)
