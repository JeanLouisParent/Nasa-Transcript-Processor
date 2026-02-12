"""
Output Generation Module.

This module handles the creation of output files for each processed page:
- Raw PDF (single page extracted)
- Enhanced image (PNG)

For AI Agents:
    - Each page gets its own directory: output/<PDF_STEM>/Page_NNN/
    - Blocks images are not generated
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from src.config.global_config import GlobalConfig


@dataclass
class PageOutput:
    """
    Paths to generated output files for a page.
    """
    page_num: int
    page_dir: Path
    assets_dir: Path
    ocr_dir: Path
    raw_pdf: Path
    enhanced_image: Path


class OutputGenerator:
    """
    Generates output files for processed pages.
    """

    def __init__(
        self,
        output_dir: Path,
        pdf_stem: str,
        config: GlobalConfig | None = None
    ):
        self.output_dir = Path(output_dir)
        self.pdf_stem = pdf_stem
        self.config = config or GlobalConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_page_dir(self, page_num: int) -> Path:
        """Get the output directory for a specific page."""
        page_dir = self.output_dir / "pages" / f"Page_{page_num + 1:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir

    def get_assets_dir(self, page_num: int) -> Path:
        """Get the assets output directory for a specific page."""
        assets_dir = self.get_page_dir(page_num) / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        return assets_dir

    def get_ocr_dir(self, page_num: int) -> Path:
        """Get the OCR output directory for a specific page."""
        ocr_dir = self.get_page_dir(page_num) / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        return ocr_dir

    def get_raw_pdf_path(self, page_num: int) -> Path:
        """Get the output path for the extracted single-page PDF."""
        page_dir = self.get_assets_dir(page_num)
        page_id = f"{self.pdf_stem}_page_{page_num + 1:04d}"
        return page_dir / f"{page_id}_raw.pdf"

    def generate(
        self,
        page_num: int,
        enhanced_image: np.ndarray,
    ) -> PageOutput:
        """
        Generate all output files for a page.
        """
        page_dir = self.get_page_dir(page_num)
        assets_dir = self.get_assets_dir(page_num)
        ocr_dir = self.get_ocr_dir(page_num)

        ext = self.config.output_format
        page_id = f"{self.pdf_stem}_page_{page_num + 1:04d}"
        enhanced_path = assets_dir / f"{page_id}_enhanced.{ext}"
        raw_pdf_path = self.get_raw_pdf_path(page_num)

        # Save enhanced image
        self._save_image(enhanced_image, enhanced_path)

        logger.debug(f"Generated outputs for page {page_num + 1} in {page_dir}")

        return PageOutput(
            page_num=page_num,
            page_dir=page_dir,
            assets_dir=assets_dir,
            ocr_dir=ocr_dir,
            raw_pdf=raw_pdf_path,
            enhanced_image=enhanced_path
        )

    def _save_image(self, image: np.ndarray, path: Path) -> None:
        """Save image with appropriate compression."""
        if path.suffix.lower() == ".png":
            cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(str(path), image)
