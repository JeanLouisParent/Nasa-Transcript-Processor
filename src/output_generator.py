"""
Output Generation Module.

This module handles the creation of output files for each processed page:
- Raw PDF (single page extracted)
- Enhanced image (PNG)
- Blocks image (enhanced + block overlays)

For AI Agents:
    - Each page gets its own directory: output/<PDF_STEM>/Page_NNN/
    - Blocks images show HEADER (blue), ANNOTATION (magenta), FOOTER (gray)
    - COMM blocks are shown with light green fill + green outline
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .config import PipelineConfig
from .layout_detector import Block, BlockType, LayoutResult, SubColumn

# Colors (BGR format for OpenCV)
COLORS = {
    'header': (255, 150, 50),      # Blue
    'footer': (150, 150, 150),     # Gray
    'annotation': (255, 100, 255), # Magenta
    'comm': (100, 200, 100),       # Green (block outline)
    'comm_fill': (190, 230, 190),  # Light green (block fill)
    'timestamp': (50, 220, 220),   # Yellow
    'speaker': (220, 220, 50),     # Cyan
    'text': (100, 100, 255),       # Red
}


@dataclass
class PageOutput:
    """
    Paths to generated output files for a page.
    """
    page_num: int
    page_dir: Path
    raw_pdf: Path
    enhanced_image: Path
    blocks_image: Path


class OutputGenerator:
    """
    Generates output files for processed pages.
    """

    def __init__(
        self,
        output_dir: Path,
        pdf_stem: str,
        config: PipelineConfig | None = None
    ):
        self.output_dir = Path(output_dir)
        self.pdf_stem = pdf_stem
        self.config = config or PipelineConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_page_dir(self, page_num: int) -> Path:
        """Get the output directory for a specific page."""
        page_dir = self.output_dir / f"Page_{page_num + 1:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir

    def get_raw_pdf_path(self, page_num: int) -> Path:
        """Get the output path for the extracted single-page PDF."""
        page_dir = self.get_page_dir(page_num)
        page_id = f"{self.pdf_stem}_page_{page_num + 1:04d}"
        return page_dir / f"{page_id}_raw.pdf"

    def generate(
        self,
        page_num: int,
        enhanced_image: np.ndarray,
        layout: LayoutResult,
    ) -> PageOutput:
        """
        Generate all output files for a page.
        """
        page_dir = self.get_page_dir(page_num)

        ext = self.config.output_format
        page_id = f"{self.pdf_stem}_page_{page_num + 1:04d}"
        enhanced_path = page_dir / f"{page_id}_enhanced.{ext}"
        blocks_path = page_dir / f"{page_id}_blocks.{ext}"
        raw_pdf_path = self.get_raw_pdf_path(page_num)

        # Save enhanced image
        self._save_image(enhanced_image, enhanced_path)

        # Generate and save blocks image
        blocks_image = self._create_blocks_image(enhanced_image, layout)
        self._save_image(blocks_image, blocks_path)

        logger.debug(f"Generated outputs for page {page_num + 1} in {page_dir}")

        return PageOutput(
            page_num=page_num,
            page_dir=page_dir,
            raw_pdf=raw_pdf_path,
            enhanced_image=enhanced_path,
            blocks_image=blocks_path
        )

    def _save_image(self, image: np.ndarray, path: Path) -> None:
        """Save image with appropriate compression."""
        if path.suffix.lower() == ".png":
            cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(str(path), image)

    def _create_blocks_image(
        self,
        enhanced_image: np.ndarray,
        layout: LayoutResult
    ) -> np.ndarray:
        """
        Create blocks visualization with block overlays.

        - HEADER blocks: blue outline
        - ANNOTATION blocks: magenta outline
        - COMM blocks: light green fill + green outline
        - COMM sub-columns: timestamp (yellow), speaker (cyan), text (red)
        """
        # Convert grayscale to BGR
        if len(enhanced_image.shape) == 2:
            blocks_img = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        else:
            blocks_img = enhanced_image.copy()

        # Draw each block
        for block in layout.blocks:
            if block.block_type == BlockType.HEADER:
                self._draw_block(blocks_img, block, COLORS['header'], "HEADER")

            elif block.block_type == BlockType.FOOTER:
                self._draw_block(blocks_img, block, COLORS['footer'], "FOOTER")

            elif block.block_type == BlockType.ANNOTATION:
                self._draw_block(blocks_img, block, COLORS['annotation'], "ANNOT")

            elif block.block_type == BlockType.COMM:
                self._draw_block_fill(blocks_img, block, COLORS['comm_fill'], alpha=0.18)
                self._draw_block_outline(blocks_img, block, COLORS['comm'], thickness=1)
                for subcol in block.sub_columns:
                    color = COLORS.get(subcol.col_type, (180, 180, 180))
                    self._draw_subcol(blocks_img, subcol, color)

        return blocks_img

    def _draw_block(
        self,
        img: np.ndarray,
        block: Block,
        color: tuple[int, int, int],
        label: str
    ) -> None:
        """Draw a block with semi-transparent fill and label."""
        # Semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay, (block.x, block.y), (block.x2, block.y2), color, -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # Border
        cv2.rectangle(img, (block.x, block.y), (block.x2, block.y2), color, 2)

        # Label
        self._draw_label(img, label, block.x, block.y, color)

    def _draw_block_outline(
        self,
        img: np.ndarray,
        block: Block,
        color: tuple[int, int, int],
        thickness: int = 2
    ) -> None:
        """Draw just the outline of a block."""
        cv2.rectangle(img, (block.x, block.y), (block.x2, block.y2), color, thickness)

    def _draw_block_fill(
        self,
        img: np.ndarray,
        block: Block,
        color: tuple[int, int, int],
        alpha: float = 0.2
    ) -> None:
        """Draw a semi-transparent fill for a block."""
        overlay = img.copy()
        cv2.rectangle(overlay, (block.x, block.y), (block.x2, block.y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _draw_subcol(
        self,
        img: np.ndarray,
        subcol: SubColumn,
        color: tuple[int, int, int]
    ) -> None:
        """Draw a sub-column with semi-transparent fill and small label."""
        overlay = img.copy()
        cv2.rectangle(overlay, (subcol.x, subcol.y), (subcol.x2, subcol.y2), color, -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        cv2.rectangle(img, (subcol.x, subcol.y), (subcol.x2, subcol.y2), color, 1)

        label = subcol.col_type[0].upper()
        if subcol.col_type == 'text':
            label = 'TX'
        cv2.putText(
            img, label,
            (subcol.x + 2, subcol.y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
        )

    def _draw_label(
        self,
        img: np.ndarray,
        label: str,
        x: int,
        y: int,
        color: tuple[int, int, int]
    ) -> None:
        """Draw a label above a block."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y - 5, text_h + 5)

        # Background
        cv2.rectangle(
            img,
            (x, label_y - text_h - 5),
            (x + text_w + 4, label_y + baseline),
            color, -1
        )

        # Text
        cv2.putText(img, label, (x + 2, label_y - 2), font, font_scale, (255, 255, 255), thickness)
