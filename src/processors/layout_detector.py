"""
Layout Detection Module V21 - Relative Anchoring & Structural Header Detection.

1. Finds the horizontal start of content (X_OFFSET) to handle shift.
2. Defines columns relative to X_OFFSET.
3. Identifies Header by finding the first row that spans all 3 columns (Timestamp, Speaker, Text).
"""

from dataclasses import dataclass, field
from enum import Enum
import cv2
import numpy as np
from src.core.config import PipelineConfig

class BlockType(Enum):
    HEADER = "header"
    FOOTER = "footer"
    ANNOTATION = "annotation"
    COMM = "comm"

@dataclass
class SubColumn:
    x: int; y: int; width: int; height: int; col_type: str
    @property
    def x2(self) -> int: return self.x + self.width
    @property
    def y2(self) -> int: return self.y + self.height

@dataclass
class Block:
    x: int; y: int; width: int; height: int
    block_type: BlockType
    sub_columns: list[SubColumn] = field(default_factory=list)
    @property
    def x2(self) -> int: return self.x + self.width
    @property
    def y2(self) -> int: return self.y + self.height

@dataclass
class LayoutResult:
    blocks: list[Block]
    page_width: int; page_height: int
    sep1_x: int
    sep2_x: int

class LayoutDetector:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def detect(self, image: np.ndarray) -> LayoutResult:
        h, w = image.shape[:2]
        binary = self._binarize(image)
        
        # 1. Detect X_OFFSET (Start of content)
        # Scan vertical projection of the middle band
        v_proj = np.sum(binary[int(h*0.3):int(h*0.7), :], axis=0)
        x_offset = 0
        for x in range(50, w//2):
            # Look for a block of ink (Timestamp start)
            if np.mean(v_proj[x:x+20]) > 500:
                x_offset = x
                break
        
        # Define Relative Separators
        # TS width ~280px, Gap ~50px -> Sep1 ~ +330
        # Spk width ~150px, Gap ~50px -> Sep2 ~ +550
        sep1_x = x_offset + 330
        sep2_x = x_offset + 550
        
        # 2. Detect Rows
        line_regions = self._find_line_regions(binary)
        if not line_regions: return LayoutResult([], w, h, sep1_x, sep2_x)
        rows = self._cluster_rows(sorted(line_regions, key=lambda r: r[1]))
        
        final_blocks = []
        header_rows = []
        body_rows = []
        
        # 3. Find First Valid COMM Row (Structure Check)
        first_comm_index = 0
        for i, row_parts in enumerate(rows):
            ry = min(r[1] for r in row_parts)
            rh = max(r[1] + r[3] for r in row_parts) - ry
            
            # Check presence in all 3 columns
            has_ts = self._has_ink(binary, ry, ry+rh, x_offset, sep1_x)
            has_spk = self._has_ink(binary, ry, ry+rh, sep1_x, sep2_x)
            has_txt = self._has_ink(binary, ry, ry+rh, sep2_x, w - 50)
            
            # A valid COMM start must have TS and (Speaker OR Text)
            # We relax slightly: TS is mandatory.
            if has_ts and (has_spk or has_txt):
                first_comm_index = i
                break
        
        header_rows = rows[:first_comm_index]
        body_rows = rows[first_comm_index:]

        # 4. Build Header
        if header_rows:
            hx = min(r[0] for row in header_rows for r in row)
            hy = min(r[1] for row in header_rows for r in row)
            hw = max(r[0] + r[2] for row in header_rows for r in row) - hx
            hh = max(r[1] + r[3] for row in header_rows for r in row) - hy
            final_blocks.append(Block(0, hy, w, hh + 10, BlockType.HEADER))

        # 5. Build COMMs
        current_comm = None
        for row_parts in body_rows:
            ry = min(r[1] for r in row_parts)
            rh = max(r[1] + r[3] for r in row_parts) - ry
            
            # Anchor check
            has_ts = self._has_ink(binary, ry, ry+rh, x_offset, sep1_x)
            has_spk = self._has_ink(binary, ry, ry+rh, sep1_x, sep2_x)
            
            if has_ts or has_spk:
                # NEW BLOCK
                current_comm = Block(0, ry - 5, w, rh + 10, BlockType.COMM)
                current_comm.sub_columns.append(SubColumn(0, ry, sep1_x, rh, "timestamp"))
                current_comm.sub_columns.append(SubColumn(sep1_x, ry, sep2_x - sep1_x, rh, "speaker"))
                current_comm.sub_columns.append(SubColumn(sep2_x, ry, w - sep2_x, rh, "text"))
                final_blocks.append(current_comm)
            elif current_comm:
                # CONTINUATION
                current_comm.height = (ry + rh + 5) - current_comm.y
                for s in current_comm.sub_columns:
                    if s.col_type == "text": s.height = (ry + rh) - s.y
            else:
                # Orphan before first real comm? Add to header or make orphan block
                if final_blocks and final_blocks[-1].block_type == BlockType.HEADER:
                    # Extend Header
                    final_blocks[-1].height = (ry + rh) - final_blocks[-1].y
                else:
                    current_comm = Block(0, ry - 5, w, rh + 10, BlockType.COMM)
                    final_blocks.append(current_comm)

        return LayoutResult(final_blocks, w, h, sep1_x, sep2_x)

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        return binary

    def _find_line_regions(self, binary: np.ndarray) -> list:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 300]

    def _cluster_rows(self, regions: list) -> list:
        if not regions: return []
        rows = []
        current_row = [regions[0]]
        for next_reg in regions[1:]:
            if next_reg[1] < (current_row[-1][1] + current_row[-1][3] + 15):
                current_row.append(next_reg)
            else:
                rows.append(current_row)
                current_row = [next_reg]
        rows.append(current_row)
        return rows

    def _has_ink(self, binary, y1, y2, x1, x2):
        if y1 >= y2 or x1 >= x2: return False
        roi = binary[y1:y2, x1:x2]
        return np.sum(roi) > 500
