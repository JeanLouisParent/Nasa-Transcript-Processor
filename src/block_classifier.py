"""
Block Classification Module.

This module classifies detected text blocks based on their geometric
properties (position, size, aspect ratio). Classification is approximate
and based on typical NASA transcript layouts.

For AI Agents:
    - Classification is purely geometric (no OCR)
    - Typical layout: header + 3 columns (timestamp, speaker, transcript)
    - Special cases: annotations (centered), continuations (no timestamp)
    - Classification heuristics may need adjustment for other missions
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from loguru import logger

from .config import PipelineConfig
from .layout_detector import Block, LayoutResult


class BlockType(Enum):
    """
    Types of content blocks in NASA transcripts.

    Values:
        HEADER: Page header (usually 2-4 lines at top)
        TIMESTAMP: Time code in leftmost column
        SPEAKER: Speaker identifier in middle column
        TRANSCRIPT: Transcribed speech in rightmost column
        ANNOTATION: Short centered annotation or note
        CONTINUATION: Transcript continuation without timestamp
        UNKNOWN: Could not determine block type
    """
    HEADER = "header"
    TIMESTAMP = "timestamp"
    SPEAKER = "speaker"
    TRANSCRIPT = "transcript"
    ANNOTATION = "annotation"
    CONTINUATION = "continuation"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedBlock:
    """
    A block with its classified type.

    Attributes:
        block: The original detected block
        block_type: Classified type of the block
        confidence: Classification confidence (0-1)
        column: Column index (0=timestamp, 1=speaker, 2=transcript, -1=other)
    """
    block: Block
    block_type: BlockType
    confidence: float = 1.0
    column: int = -1

    @property
    def x(self) -> int:
        return self.block.x

    @property
    def y(self) -> int:
        return self.block.y

    @property
    def width(self) -> int:
        return self.block.width

    @property
    def height(self) -> int:
        return self.block.height

    @property
    def x2(self) -> int:
        return self.block.x2

    @property
    def y2(self) -> int:
        return self.block.y2


@dataclass
class ClassificationResult:
    """
    Result of block classification.

    Attributes:
        blocks: List of classified blocks
        header_blocks: Blocks identified as headers
        content_blocks: Non-header blocks
        has_continuation: True if page starts with continuation (no timestamp)
        column_counts: Number of blocks in each column
    """
    blocks: list[ClassifiedBlock]
    header_blocks: list[ClassifiedBlock]
    content_blocks: list[ClassifiedBlock]
    has_continuation: bool = False
    column_counts: tuple[int, int, int] = (0, 0, 0)


class BlockClassifier:
    """
    Classifies detected blocks based on geometric properties.

    Classification rules:
    1. Header: Block in top 10% of page
    2. Timestamp: Left column (0-15% of width), narrow
    3. Speaker: Middle column (15-30% of width), narrow
    4. Transcript: Right column (30-100% of width), wide
    5. Annotation: Centered block, narrow, not in columns
    6. Continuation: First transcript block without corresponding timestamp

    Attributes:
        config: Pipeline configuration
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the block classifier.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()

    def classify(self, layout: LayoutResult) -> ClassificationResult:
        """
        Classify all blocks in a layout.

        Args:
            layout: Layout detection result

        Returns:
            ClassificationResult with classified blocks
        """
        page_h = layout.page_height
        page_w = layout.page_width
        header_threshold = int(page_h * self.config.header_ratio)
        col1_boundary = int(page_w * self.config.col1_end)
        col2_boundary = int(page_w * self.config.col2_end)

        classified = []
        header_blocks = []
        content_blocks = []
        column_counts = [0, 0, 0]

        # First pass: classify each block
        for block in layout.blocks:
            block_type, confidence, column = self._classify_block(
                block, page_w, page_h,
                header_threshold, col1_boundary, col2_boundary
            )

            classified_block = ClassifiedBlock(
                block=block,
                block_type=block_type,
                confidence=confidence,
                column=column
            )
            classified.append(classified_block)

            if block_type == BlockType.HEADER:
                header_blocks.append(classified_block)
            else:
                content_blocks.append(classified_block)

            if 0 <= column <= 2:
                column_counts[column] += 1

        # Second pass: detect continuations
        has_continuation = self._detect_continuations(content_blocks, header_threshold)

        logger.debug(
            f"Classified {len(classified)} blocks: "
            f"{len(header_blocks)} headers, "
            f"columns={column_counts}, "
            f"continuation={has_continuation}"
        )

        return ClassificationResult(
            blocks=classified,
            header_blocks=header_blocks,
            content_blocks=content_blocks,
            has_continuation=has_continuation,
            column_counts=tuple(column_counts)
        )

    def _classify_block(
        self,
        block: Block,
        page_w: int,
        page_h: int,
        header_threshold: int,
        col1_boundary: int,
        col2_boundary: int
    ) -> tuple[BlockType, float, int]:
        """
        Classify a single block.

        Args:
            block: Block to classify
            page_w: Page width
            page_h: Page height
            header_threshold: Y coordinate below which is header
            col1_boundary: X boundary between col1 and col2
            col2_boundary: X boundary between col2 and col3

        Returns:
            Tuple of (block_type, confidence, column_index)
        """
        center_x = block.center_x
        center_y = block.center_y

        # Check if header (top of page)
        if block.y < header_threshold and block.y2 < header_threshold * 1.5:
            return BlockType.HEADER, 0.9, -1

        # Check if annotation (centered, narrow block not in typical column positions)
        is_centered = abs(center_x - page_w / 2) < page_w * 0.15
        is_narrow = block.width < page_w * 0.4
        is_short = block.height < page_h * 0.1

        if is_centered and is_narrow and is_short:
            # Check if it's not clearly in a column
            if center_x > col1_boundary and center_x < page_w * 0.7:
                return BlockType.ANNOTATION, 0.7, -1

        # Classify by column position
        if center_x < col1_boundary:
            # Timestamp column
            column = 0
            if block.width < page_w * 0.15:
                return BlockType.TIMESTAMP, 0.85, column
            else:
                return BlockType.UNKNOWN, 0.5, column

        elif center_x < col2_boundary:
            # Speaker column
            column = 1
            if block.width < page_w * 0.20:
                return BlockType.SPEAKER, 0.85, column
            else:
                return BlockType.UNKNOWN, 0.5, column

        else:
            # Transcript column
            column = 2
            return BlockType.TRANSCRIPT, 0.85, column

    def _detect_continuations(
        self,
        content_blocks: list[ClassifiedBlock],
        header_threshold: int
    ) -> bool:
        """
        Detect if page starts with a continuation (transcript without timestamp).

        A continuation occurs when:
        - First transcript block starts near top of page
        - No timestamp block at same vertical position

        Args:
            content_blocks: Non-header blocks
            header_threshold: Y coordinate below which is header

        Returns:
            True if page has a continuation at top
        """
        if not content_blocks:
            return False

        # Find topmost non-header blocks
        top_blocks = [b for b in content_blocks if b.y < header_threshold * 2]

        if not top_blocks:
            return False

        # Check for transcript without timestamp
        has_timestamp = any(b.block_type == BlockType.TIMESTAMP for b in top_blocks)
        has_transcript = any(b.block_type == BlockType.TRANSCRIPT for b in top_blocks)

        if has_transcript and not has_timestamp:
            # Mark first transcript as continuation
            for b in content_blocks:
                if b.block_type == BlockType.TRANSCRIPT and b.y < header_threshold * 2:
                    b.block_type = BlockType.CONTINUATION
                    b.confidence = 0.75
                    break
            return True

        return False

    def get_block_color(self, block_type: BlockType) -> tuple[int, int, int]:
        """
        Get the debug visualization color for a block type.

        Args:
            block_type: Type of block

        Returns:
            BGR color tuple
        """
        return self.config.block_colors.get(
            block_type.value,
            (180, 180, 180)  # Default gray
        )


def classify_layout(
    layout: LayoutResult,
    config: Optional[PipelineConfig] = None
) -> ClassificationResult:
    """
    Convenience function to classify a layout.

    Args:
        layout: Layout detection result
        config: Pipeline configuration

    Returns:
        ClassificationResult with classified blocks
    """
    classifier = BlockClassifier(config)
    return classifier.classify(layout)
