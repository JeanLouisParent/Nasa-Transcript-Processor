"""
Output Generation Module - Calibrated Professional Visualization.
"""

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from src.core.config import PipelineConfig
from src.processors.layout_detector import Block, BlockType, LayoutResult

COLORS = {
    'header': (255, 100, 0),       # Blue/Orange
    'comm': (0, 255, 0),           # Bright Pure Green
    'comm_fill': (200, 255, 200),  # Light Green Fill
    'separator': (0, 255, 0),      # Bright Pure Green for separators
}

@dataclass
class PageOutput:
    page_num: int; page_dir: Path; raw_pdf: Path; enhanced_image: Path; blocks_image: Path

class OutputGenerator:
    def __init__(self, output_dir: Path, pdf_stem: str, config: PipelineConfig | None = None):
        self.output_dir = Path(output_dir); self.pdf_stem = pdf_stem
        self.config = config or PipelineConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_page_dir(self, page_num: int) -> Path:
        page_dir = self.output_dir / f"Page_{page_num + 1:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir

    def get_raw_pdf_path(self, page_num: int) -> Path:
        page_dir = self.get_page_dir(page_num)
        page_id = f"{self.pdf_stem}_page_{page_num + 1:04d}"
        return page_dir / f"{page_id}_raw.pdf"

    def generate(self, page_num: int, enhanced_image: np.ndarray, layout: LayoutResult) -> PageOutput:
        page_dir = self.get_page_dir(page_num)
        page_id = f"{self.pdf_stem}_page_{page_num + 1:04d}"
        enhanced_path = page_dir / f"{page_id}_enhanced.png"
        blocks_path = page_dir / f"{page_id}_blocks.png"
        raw_pdf_path = self.get_raw_pdf_path(page_num)

        cv2.imwrite(str(enhanced_path), enhanced_image)
        
        # Draw Blocks
        blocks_img = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR) if len(enhanced_image.shape) == 2 else enhanced_image.copy()
        
        # Use Separators from LayoutResult
        sep1_x = getattr(layout, 'sep1_x', 720)
        sep2_x = getattr(layout, 'sep2_x', 1030)

        for b in layout.blocks:
            if b.block_type == BlockType.HEADER:
                overlay = blocks_img.copy()
                cv2.rectangle(overlay, (b.x, b.y), (b.x2, b.y2), COLORS['header'], -1)
                cv2.addWeighted(overlay, 0.1, blocks_img, 0.9, 0, blocks_img)
                cv2.rectangle(blocks_img, (b.x, b.y), (b.x2, b.y2), COLORS['header'], 1)
            elif b.block_type == BlockType.COMM:
                overlay = blocks_img.copy()
                cv2.rectangle(overlay, (b.x, b.y), (b.x2, b.y2), COLORS['comm_fill'], -1)
                cv2.addWeighted(overlay, 0.2, blocks_img, 0.8, 0, blocks_img)
                cv2.rectangle(blocks_img, (b.x, b.y), (b.x2, b.y2), COLORS['comm'], 2)
                # Dynamic separators
                cv2.line(blocks_img, (sep1_x, b.y), (sep1_x, b.y2), COLORS['separator'], 2)
                cv2.line(blocks_img, (sep2_x, b.y), (sep2_x, b.y2), COLORS['separator'], 2)

        cv2.imwrite(str(blocks_path), blocks_img)
        return PageOutput(page_num, page_dir, raw_pdf_path, enhanced_path, blocks_path)