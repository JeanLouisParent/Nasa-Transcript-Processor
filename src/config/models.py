"""
Pydantic models for configuration validation.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LexiconConfig(BaseModel):
    """Configuration for the lexicon-based corrections."""
    path: Path
    mission_keywords: list[str] = Field(default_factory=list)


class ParserConfig(BaseModel):
    """Configuration for the transcript parser."""
    header_keywords: list[str] = Field(default_factory=list)
    transition_keywords: list[str] = Field(default_factory=list)
    end_of_tape_keyword: str
    text_replacements: dict[str, str] = Field(default_factory=dict)


class CorrectorsConfig(BaseModel):
    """Configuration for the post-processing correctors."""
    speaker_ocr_fixes: dict[str, str] = Field(default_factory=dict)
    invalid_location_annotations: dict[str, list[str]] = Field(default_factory=dict)


class GlobalConfigModel(BaseModel):
    """
    Main configuration model for the application.
    Validates input from defaults.toml - all values come from TOML, not here.
    """
    # I/O
    input_dir: Path
    output_dir: Path
    state_dir: Path

    # OCR Settings
    ocr_url: str
    ocr_model: str
    ocr_prompt: str
    ocr_timeout: int
    ocr_max_tokens: int
    ocr_text_column_pass: bool
    ocr_dual_pass: bool
    ocr_faint_pass: bool

    # Processing Settings
    dpi: int
    parallel: bool
    workers: int
    timing: bool

    # Image Enhancement
    clahe_clip_limit: float
    clahe_grid_size: int
    bilateral_d: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    unsharp_amount: float
    unsharp_sigma: float
    deskew_angle_threshold: float
    deskew_max_angle: float

    # Normalization settings (Letter size at 300 DPI)
    target_width: int
    target_height: int
    margin_px: int
    output_format: str

    # Layout Detection
    col2_end: float
    min_block_area: int
    max_block_area_ratio: float
    line_kernel_width: int
    line_kernel_height: int
    block_kernel_width: int
    block_kernel_height: int

    # Debug settings
    debug: bool

    # Sub-configs
    lexicon: LexiconConfig
    parser: ParserConfig
    correctors: CorrectorsConfig

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
