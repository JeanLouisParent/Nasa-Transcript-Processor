"""
Configuration module for NASA Transcript Processing Pipeline.

This module defines the PipelineConfig dataclass that centralizes all
configuration parameters for the processing pipeline. Configuration can
be loaded from YAML files for different missions.

For AI Agents:
    - All parameters have sensible defaults for Apollo 11 transcripts
    - Column ratios (col1_end, col2_end) may need adjustment for other missions
    - DPI affects output quality and processing time (300 is standard)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class PipelineConfig:
    """
    Central configuration for the transcript processing pipeline.

    Attributes:
        dpi: Output resolution in dots per inch (default: 300)
        output_format: Image format for enhanced output (default: "png")
        parallel: Enable parallel processing (default: True)
        max_workers: Number of parallel workers (default: 4)
        target_width: Normalized page width in pixels at target DPI
        target_height: Normalized page height in pixels at target DPI
        margin_px: Uniform margin in pixels after normalization
        clahe_clip_limit: CLAHE contrast enhancement clip limit
        bilateral_d: Bilateral filter diameter
        bilateral_sigma_color: Bilateral filter color sigma
        bilateral_sigma_space: Bilateral filter space sigma
        min_block_area: Minimum block area to consider (filters noise)
        max_block_area_ratio: Maximum block area as ratio of page (filters page-sized blocks)
        col1_end: End of timestamp column as ratio of page width
        col2_end: End of speaker column as ratio of page width
        header_ratio: Header region as ratio of page height
        debug: Enable debug mode with intermediate outputs
    """

    # Extraction settings
    dpi: int = 300
    output_format: str = "png"

    # Parallelism settings
    parallel: bool = True
    max_workers: int = 4

    # Normalization settings (Letter size at 300 DPI)
    target_width: int = 2550   # 8.5 inches * 300 DPI
    target_height: int = 3300  # 11 inches * 300 DPI
    margin_px: int = 75        # ~0.25 inches * 300 DPI

    # Image enhancement settings
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    unsharp_amount: float = 1.5
    unsharp_sigma: float = 1.0

    

    # Morphological cleaning
    morph_kernel_size: int = 2
    noise_max_area: int = 50  # Maximum area for noise removal

    # Deskew settings
    deskew_angle_threshold: float = 0.5  # Minimum angle to correct (degrees)
    deskew_max_angle: float = 10.0       # Maximum expected skew angle

    # Layout detection settings
    min_block_area: int = 1000
    max_block_area_ratio: float = 0.9
    line_kernel_width: int = 50   # Horizontal dilation kernel width
    line_kernel_height: int = 1   # Horizontal dilation kernel height
    block_kernel_width: int = 5   # Vertical grouping kernel width
    block_kernel_height: int = 10 # Vertical grouping kernel height

    # Column detection ratios (relative to page width)
    col1_end: float = 0.15    # End of timestamp column
    col2_end: float = 0.30    # End of speaker column
    header_ratio: float = 0.10  # Header region height ratio

    # Tesseract-assisted header detection (no output text)
    tesseract_header: bool = True
    tesseract_header_lang: str = "eng"
    tesseract_header_psm: int = 6

    # Debug settings
    debug: bool = False
    save_intermediates: bool = False

    # Block classification colors (BGR format for OpenCV)
    block_colors: dict = field(default_factory=lambda: {
        "header": (255, 100, 100),      # Blue
        "timestamp": (100, 255, 100),   # Green
        "speaker": (100, 255, 255),     # Yellow
        "transcript": (100, 100, 255),  # Red
        "annotation": (255, 100, 255),  # Magenta
        "continuation": (255, 255, 100),# Cyan
        "unknown": (180, 180, 180),     # Gray
    })

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            PipelineConfig instance with values from file

        Example YAML:
            dpi: 300
            parallel: true
            max_workers: 8
            col1_end: 0.12
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path to save YAML configuration
        """
        # Convert dataclass to dict, handling non-serializable types
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                data[key] = dict(value)
            else:
                data[key] = value

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> list[str]:
        """
        Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.dpi < 72 or self.dpi > 600:
            errors.append(f"DPI should be between 72 and 600, got {self.dpi}")

        if self.max_workers < 1:
            errors.append(f"max_workers must be >= 1, got {self.max_workers}")

        if not 0 < self.col1_end < self.col2_end < 1:
            errors.append(
                f"Column ratios must satisfy 0 < col1_end < col2_end < 1, "
                f"got col1_end={self.col1_end}, col2_end={self.col2_end}"
            )

        if not 0 < self.header_ratio < 0.5:
            errors.append(f"header_ratio should be between 0 and 0.5, got {self.header_ratio}")

        if self.output_format not in ("png", "tiff", "webp"):
            errors.append(f"output_format must be png, tiff, or webp, got {self.output_format}")

        return errors


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
