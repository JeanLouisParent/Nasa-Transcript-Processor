"""
Layout Detection Module.

Detects structured blocks in NASA transcript pages:
- HEADER: Page header (network info, tape/page numbers)
- FOOTER: Page footer (e.g., "*** Three asterisks...")
- ANNOTATION: Centered station/event markers (e.g., "VANGUARD (REV 1)")
- COMM: Communication entries with 3 sub-columns (timestamp, speaker, text)

For AI Agents:
    - COMM blocks are triplets: timestamp | speaker | text (can be multi-line)
    - ANNOTATION blocks are detected by vertical isolation (gaps above/below)
    - FOOTER is always at the very bottom of the page
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import cv2
from loguru import logger

from .config import PipelineConfig


class BlockType(Enum):
    """Types of content blocks in NASA transcripts."""
    HEADER = "header"
    FOOTER = "footer"
    ANNOTATION = "annotation"
    COMM = "comm"


@dataclass
class SubColumn:
    """A sub-column within a COMM block."""
    x: int
    y: int
    width: int
    height: int
    col_type: str  # 'timestamp', 'speaker', 'text'

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass
class Block:
    """A detected content block."""
    x: int
    y: int
    width: int
    height: int
    block_type: BlockType
    sub_columns: list[SubColumn] = field(default_factory=list)

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class LayoutResult:
    """Result of layout detection."""
    blocks: list[Block]
    header_blocks: list[Block]
    footer_blocks: list[Block]
    annotation_blocks: list[Block]
    comm_blocks: list[Block]
    page_width: int
    page_height: int


class LayoutDetector:
    """
    Detects structured layout in NASA transcript pages.

    Strategy:
    1. Find all text line regions
    2. Identify HEADER (top of page)
    3. Identify FOOTER (bottom of page)
    4. Identify ANNOTATION (centered, vertically isolated)
    5. Group remaining into COMM blocks with sub-columns
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def detect(self, image: np.ndarray) -> LayoutResult:
        """Detect layout blocks in an image."""
        h, w = image.shape[:2]

        # Binarize
        binary = self._binarize(image)

        # Find all text line regions
        line_regions = self._find_line_regions(binary)

        if not line_regions:
            return LayoutResult([], [], [], [], [], w, h)

        # Detect column boundaries from content (fallback to config defaults)
        col1_end, col2_end, content_left, content_right = self._detect_column_boundaries(
            binary=binary,
            line_regions=line_regions,
            page_w=w,
            page_h=h
        )

        # Cluster regions into text rows
        line_regions = sorted(line_regions, key=lambda r: r[1])
        row_groups = self._cluster_rows(line_regions)
        if not row_groups:
            return LayoutResult([], [], [], [], [], w, h)

        def build_rows(col1_end: int, col2_end: int) -> list[dict]:
            rows = []
            for regions in row_groups:
                min_x = min(r[0] for r in regions)
                min_y = min(r[1] for r in regions)
                max_x = max(r[0] + r[2] for r in regions)
                max_y = max(r[1] + r[3] for r in regions)
                has_timestamp = self._row_has_content(
                    binary=binary,
                    y1=min_y,
                    y2=max_y,
                    x1=content_left,
                    x2=col1_end
                )
                has_speaker = self._row_has_content(
                    binary=binary,
                    y1=min_y,
                    y2=max_y,
                    x1=col1_end,
                    x2=col2_end
                )
                speaker_density = self._row_ink_ratio(
                    binary=binary,
                    y1=min_y,
                    y2=max_y,
                    x1=col1_end,
                    x2=col2_end
                )
                text_density = self._row_ink_ratio(
                    binary=binary,
                    y1=min_y,
                    y2=max_y,
                    x1=col2_end,
                    x2=content_right
                )
                text_regions = [
                    r for r in regions
                    if (r[0] + r[2] / 2) >= col2_end
                ]
                text_left = min(r[0] for r in text_regions) if text_regions else None
                boundary_density_1 = self._row_ink_ratio(
                    binary=binary,
                    y1=min_y,
                    y2=max_y,
                    x1=max(0, col1_end - 8),
                    x2=min(w, col1_end + 8)
                )
                boundary_density_2 = self._row_ink_ratio(
                    binary=binary,
                    y1=min_y,
                    y2=max_y,
                    x1=max(0, col2_end - 8),
                    x2=min(w, col2_end + 8)
                )
                left_header = False
                right_header = False
                center_header = False
                for rx, ry, rw, rh in regions:
                    cx = rx + rw / 2
                    if rx < w * 0.25 and rw < w * 0.35:
                        left_header = True
                    if rx > w * 0.6 and rw < w * 0.3:
                        right_header = True
                    if (
                        rw > w * 0.35
                        and rw < w * 0.7
                        and abs(cx - w / 2) < w * 0.12
                    ):
                        center_header = True
                rows.append({
                    "regions": regions,
                    "x": min_x,
                    "y": min_y,
                    "x2": max_x,
                    "y2": max_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                    "has_timestamp": has_timestamp,
                    "has_speaker": has_speaker,
                    "speaker_density": speaker_density,
                    "text_density": text_density,
                    "text_left": text_left,
                    "boundary_density_1": boundary_density_1,
                    "boundary_density_2": boundary_density_2,
                    "left_header": left_header,
                    "right_header": right_header,
                    "center_header": center_header,
                })
            return rows

        rows = build_rows(col1_end, col2_end)

        default_col1 = int(w * self.config.col1_end)
        default_col2 = int(w * self.config.col2_end)
        timestamp_rows = sum(1 for r in rows if r["has_timestamp"])
        speaker_rows = sum(1 for r in rows if r["has_speaker"])
        if (
            timestamp_rows < max(2, int(len(rows) * 0.1))
            and speaker_rows >= max(3, int(len(rows) * 0.2))
            and col1_end < default_col1
        ):
            col1_end = default_col1
            col2_end = max(col2_end, default_col2, col1_end + max(30, int(w * 0.04)))
            rows = build_rows(col1_end, col2_end)

        rows = sorted(rows, key=lambda r: r["y"])
        line_spacing = self._calculate_row_spacing(rows)

        # Classify regions
        header_rows = []
        footer_rows = []
        annotation_rows = []
        comm_rows = []

        header_threshold = int(h * self.config.header_ratio)
        footer_threshold = int(h * 0.92)  # Bottom 8% is footer zone
        timestamp_col_end = col1_end
        comm_text_min = 0.025
        comm_speaker_min = 0.02

        first_comm_y = header_threshold
        text_width = max(1, content_right - col2_end)
        text_left_margin = max(30, int(text_width * 0.3))
        header_text_left_guard = max(20, int(text_width * 0.12))
        header_marker_default = int(h * self.config.header_ratio * 1.2)
        header_anchor_rows = [row for row in rows if row["right_header"] or row["center_header"]]
        for row in rows:
            is_text_aligned = (
                row["text_left"] is not None
                and row["text_left"] <= col2_end + text_left_margin
            )
            is_comm_like = (
                row["text_density"] >= comm_text_min
                and (row["has_timestamp"] or row["has_speaker"] or is_text_aligned)
            )
            if (
                row["has_timestamp"]
                and row["text_density"] >= comm_text_min
                and row["speaker_density"] >= comm_speaker_min
                and is_text_aligned
            ):
                first_comm_y = row["y"]
                break

        header_marker_threshold = header_marker_default
        if first_comm_y > header_threshold:
            header_marker_threshold = min(
                int(h * 0.35),
                max(header_marker_default, int(first_comm_y - line_spacing * 0.3))
            )

        footer_gap_threshold = max(line_spacing * 1.5, 50)
        footer_boundary_gap = max(line_spacing * 0.3, 15)
        footer_boundary_threshold = 0.08
        footer_width_ratio_min = 0.7
        footer_start_y = None
        footer_left_max = col1_end + max(12, int(w * 0.02))
        footer_text_left_max = col1_end + max(20, int(w * 0.03))
        footer_zone_start_y = max(
            int(h * 0.88),
            int(rows[-1]["y"] - max(line_spacing * 2.0, 60))
        )

        def gap_above_nonoverlap(index: int) -> float:
            y = rows[index]["y"]
            j = index - 1
            while j >= 0 and rows[j]["y2"] >= y:
                j -= 1
            if j < 0:
                return y
            return y - rows[j]["y2"]

        for i, row in enumerate(rows):
            x = row["x"]
            y = row["y"]
            x2 = row["x2"]
            y2 = row["y2"]
            center_x = (x + x2) / 2
            width_ratio = row["width"] / text_width
            is_centered = abs(center_x - w / 2) < w * 0.18
            is_narrow = row["width"] < w * 0.6
            left_margin = x - col2_end
            is_centered_in_text = left_margin >= text_width * 0.15
            min_anno_w = max(80, int(w * 0.05))
            min_anno_h = max(12, int(h * 0.006))
            is_large_enough = row["width"] >= min_anno_w and row["height"] >= min_anno_h

            # HEADER: top region
            is_top_text_only = (
                not row["has_timestamp"]
                and not row["has_speaker"]
                and row["text_left"] is not None
                and row["text_left"] <= col2_end + header_text_left_guard
            )
            is_text_aligned = (
                row["text_left"] is not None
                and row["text_left"] <= col2_end + text_left_margin
            )
            is_comm_like = (
                row["text_density"] >= comm_text_min
                and (row["has_timestamp"] or row["has_speaker"] or is_text_aligned)
            )
            is_header_marker = (
                (row["right_header"] or row["center_header"])
                and y < header_marker_threshold
            )
            is_header_neighbor = (
                row["left_header"]
                and y < header_marker_threshold
                and not is_comm_like
                and any(abs(y - anchor["y"]) <= line_spacing * 1.5 for anchor in header_anchor_rows)
            )
            if (is_header_marker or is_header_neighbor) and y < first_comm_y:
                header_rows.append(row)
                continue
            if (
                y < header_threshold
                and y < first_comm_y
                and not is_top_text_only
                and not is_comm_like
            ):
                header_rows.append(row)
                continue

            # FOOTER: bottom of page
            if footer_start_y is not None and y >= footer_start_y:
                footer_rows.append(row)
                continue

            if y > footer_threshold or y >= footer_zone_start_y:
                gap_above = gap_above_nonoverlap(i)
                footer_candidate = (
                    not row["has_timestamp"]
                    and not row["has_speaker"]
                    and is_large_enough
                )
                footer_boundary_candidate = is_large_enough
                footer_text_left = row["text_left"]
                is_footer_left_aligned = (
                    footer_text_left is not None
                    and footer_text_left <= footer_text_left_max
                )
                if (
                    footer_candidate
                    and is_centered
                    and is_centered_in_text
                    and is_narrow
                    and width_ratio >= 0.12
                    and gap_above > footer_gap_threshold
                ):
                    footer_rows.append(row)
                    if footer_start_y is None:
                        footer_start_y = y
                    continue
                if (
                    footer_boundary_candidate
                    and (x <= footer_left_max or is_footer_left_aligned)
                    and width_ratio >= footer_width_ratio_min
                    and row["boundary_density_1"] > footer_boundary_threshold
                    and row["boundary_density_2"] > footer_boundary_threshold
                    and gap_above > footer_boundary_gap
                ):
                    footer_rows.append(row)
                    if footer_start_y is None:
                        footer_start_y = y
                    continue

            # Check if ANNOTATION (centered, no timestamp, vertically isolated)
            has_timestamp = row["has_timestamp"]
            has_speaker = row["has_speaker"]
            is_narrow_text = width_ratio <= 0.55

            if (
                not has_timestamp
                and not has_speaker
                and is_centered
                and is_centered_in_text
                and is_narrow
                and is_narrow_text
                and is_large_enough
                and y < h * 0.85
            ):
                # Check vertical isolation (gap above and below)
                if i == 0:
                    gap_above = y
                else:
                    gap_above = y - rows[i - 1]["y2"]
                if i >= len(rows) - 1:
                    gap_below = line_spacing * 2
                else:
                    gap_below = rows[i + 1]["y"] - y2

                # Annotation if isolated with significant gaps on both sides
                gap_threshold = max(line_spacing * 1.3, 40)
                if gap_above > gap_threshold and gap_below > gap_threshold:
                    annotation_rows.append(row)
                    continue

            # Otherwise it's a COMM candidate
            comm_rows.append(row)

        if not header_rows:
            header_limit = min(header_marker_threshold, int(h * 0.2))
            fallback_rows = []
            prev_y2 = None
            for row in rows:
                if row["y"] >= header_limit:
                    break
                if prev_y2 is not None:
                    gap = row["y"] - prev_y2
                    if gap > line_spacing * 2.5:
                        break
                fallback_rows.append(row)
                prev_y2 = row["y2"]

            if not fallback_rows and rows:
                fallback_rows = [rows[0]]

            if fallback_rows:
                header_rows = fallback_rows
                header_set = set(id(r) for r in header_rows)
                comm_rows = [r for r in comm_rows if id(r) not in header_set]

        # Group COMM candidates into blocks
        comm_blocks = self._group_comm_blocks(
            comm_rows,
            page_w=w,
            page_h=h,
            line_spacing=line_spacing,
            col1_end=col1_end,
            col2_end=col2_end,
            content_left=content_left,
            content_right=content_right,
            binary=binary
        )

        header_blocks = self._merge_rows_to_block(header_rows, BlockType.HEADER)
        footer_blocks = self._merge_rows_to_block(footer_rows, BlockType.FOOTER)
        annotation_blocks = [
            self._row_to_block(row, BlockType.ANNOTATION)
            for row in annotation_rows
        ]

        # Combine all blocks and sort by Y
        all_blocks = header_blocks + footer_blocks + annotation_blocks + comm_blocks
        all_blocks.sort(key=lambda b: b.y)

        logger.debug(
            f"Detected: {len(header_blocks)} headers, {len(footer_blocks)} footers, "
            f"{len(annotation_blocks)} annotations, {len(comm_blocks)} comms"
        )

        return LayoutResult(
            blocks=all_blocks,
            header_blocks=header_blocks,
            footer_blocks=footer_blocks,
            annotation_blocks=annotation_blocks,
            comm_blocks=comm_blocks,
            page_width=w,
            page_height=h
        )

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image."""
        _, binary = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def _find_line_regions(self, binary: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Find text line regions."""
        # Dilate horizontally to connect characters
        kernel_w = max(5, int(self.config.line_kernel_width))
        kernel_h = max(1, int(self.config.line_kernel_height))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        dilated = cv2.dilate(binary, h_kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 100:  # Filter tiny noise
                regions.append((x, y, w, h))

        return regions

    def _cluster_rows(
        self,
        regions: list[tuple[int, int, int, int]]
    ) -> list[list[tuple[int, int, int, int]]]:
        """Cluster line regions into rows based on vertical proximity."""
        if not regions:
            return []

        rows = []
        current = [regions[0]]
        curr_bottom = regions[0][1] + regions[0][3]
        curr_height = regions[0][3]

        for region in regions[1:]:
            top = region[1]
            bottom = region[1] + region[3]
            height = region[3]
            gap = top - curr_bottom

            # Overlap -> same row, otherwise use gap/height heuristics
            height_diff = abs(height - curr_height) / max(1.0, float(curr_height))
            gap_threshold = max(2, int(np.median([r[3] for r in current]) * 0.2))
            if gap <= 0 or (gap <= gap_threshold and height_diff <= 0.6):
                current.append(region)
                curr_bottom = max(curr_bottom, bottom)
                curr_height = int(np.median([r[3] for r in current]))
            else:
                rows.append(current)
                current = [region]
                curr_bottom = bottom
                curr_height = height

        rows.append(current)
        return rows

    def _calculate_row_spacing(self, rows: list[dict]) -> float:
        """Calculate typical vertical spacing between rows."""
        if len(rows) < 2:
            return 30

        gaps = []
        for i in range(1, len(rows)):
            gap = rows[i]["y"] - rows[i - 1]["y2"]
            if gap > 0:
                gaps.append(gap)

        if gaps:
            return np.median(gaps)
        return 30

    def _detect_column_boundaries(
        self,
        binary: np.ndarray,
        line_regions: list[tuple[int, int, int, int]],
        page_w: int,
        page_h: int
    ) -> tuple[int, int, int, int]:
        """
        Detect column boundaries from vertical content projection.

        Returns:
            Tuple of (col1_end, col2_end, content_left, content_right)
        """
        default_col1 = int(page_w * self.config.col1_end)
        default_col2 = int(page_w * self.config.col2_end)

        # Use middle band to avoid header/footer noise
        top = int(page_h * self.config.header_ratio)
        bottom = int(page_h * 0.9)
        band = binary[top:bottom, :]

        if band.size == 0:
            return default_col1, default_col2, 0, page_w - 1

        proj = np.sum(band > 0, axis=0).astype(np.float32)
        if proj.max() == 0:
            return default_col1, default_col2, 0, page_w - 1

        # Smooth projection to reduce noise
        window = max(15, int(page_w * 0.01))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(proj, kernel, mode="same")

        # Estimate content bounds from projection and line regions
        thresh = max(3.0, smoothed.max() * 0.08)
        content_cols = np.where(smoothed > thresh)[0]
        if content_cols.size:
            content_left = int(content_cols[0])
            content_right = int(content_cols[-1])
        else:
            content_left = 0
            content_right = page_w - 1

        if line_regions:
            xs = [r[0] for r in line_regions]
            x2s = [r[0] + r[2] for r in line_regions]
            content_left = min(content_left, min(xs))
            content_right = max(content_right, max(x2s))

        content_left = max(0, content_left)
        content_right = min(page_w - 1, content_right)
        if content_right <= content_left:
            return default_col1, default_col2, 0, page_w - 1

        col1_end = self._find_valley(smoothed, default_col1, content_left, content_right, page_w)
        col2_end = self._find_valley(smoothed, default_col2, content_left, content_right, page_w)

        # Enforce ordering and minimum separation
        min_gap = max(30, int(page_w * 0.04))
        min_col1_span = max(30, int(page_w * 0.03))
        min_text_span = max(100, int(page_w * 0.20))

        if col1_end - content_left < min_col1_span:
            col1_end = default_col1

        if content_right - col2_end < min_text_span:
            col2_end = default_col2

        if col2_end - col1_end < min_gap:
            col1_end = default_col1
            col2_end = default_col2

        return col1_end, col2_end, content_left, content_right

    def _find_valley(
        self,
        profile: np.ndarray,
        center: int,
        content_left: int,
        content_right: int,
        page_w: int
    ) -> int:
        """Find a low-density valley near a target center."""
        radius = int(page_w * 0.08)
        start = max(content_left + 1, center - radius)
        end = min(content_right - 1, center + radius)

        if end <= start:
            return center

        window = profile[start:end]
        valley_offset = int(np.argmin(window))
        valley = start + valley_offset

        # Accept valley if sufficiently low in its window
        local_min = profile[valley]
        if local_min <= np.percentile(window, 30):
            return valley

        return center

    def _calculate_line_spacing(self, regions: list[tuple[int, int, int, int]]) -> float:
        """Calculate typical vertical spacing between lines."""
        if len(regions) < 2:
            return 30  # Default

        # Calculate gaps between consecutive regions
        gaps = []
        sorted_regions = sorted(regions, key=lambda r: r[1])

        for i in range(1, len(sorted_regions)):
            prev_bottom = sorted_regions[i-1][1] + sorted_regions[i-1][3]
            curr_top = sorted_regions[i][1]
            gap = curr_top - prev_bottom
            if gap > 0:
                gaps.append(gap)

        if gaps:
            # Use median to avoid outliers
            return np.median(gaps)
        return 30

    def _get_gap_above(
        self,
        index: int,
        regions: list[tuple[int, int, int, int]],
        default_spacing: float
    ) -> float:
        """Get vertical gap above a region."""
        if index == 0:
            return regions[0][1]  # Distance from top

        curr_top = regions[index][1]
        prev_bottom = regions[index-1][1] + regions[index-1][3]
        return curr_top - prev_bottom

    def _get_gap_below(
        self,
        index: int,
        regions: list[tuple[int, int, int, int]],
        default_spacing: float
    ) -> float:
        """Get vertical gap below a region."""
        if index >= len(regions) - 1:
            return default_spacing * 2  # Assume gap at end

        curr_bottom = regions[index][1] + regions[index][3]
        next_top = regions[index+1][1]
        return next_top - curr_bottom

    def _group_comm_blocks(
        self,
        rows: list[dict],
        page_w: int,
        page_h: int,
        line_spacing: float,
        col1_end: int,
        col2_end: int,
        content_left: int,
        content_right: int,
        binary: np.ndarray
    ) -> list[Block]:
        """
        Group text regions into COMM blocks.

        A COMM block starts with a timestamp row and includes subsequent
        continuation rows (without timestamp) until the next timestamp
        or a large vertical gap.
        """
        if not rows:
            return []

        comm_blocks = []
        current_rows = []
        pending_rows = []
        min_timestamp_height = max(10, int(line_spacing * 0.6))
        min_noise_density = 0.01

        def flush_rows(rows_to_flush: list[dict]) -> None:
            if not rows_to_flush:
                return
            regions = [r for cr in rows_to_flush for r in cr["regions"]]
            block = self._create_comm_block(
                regions,
                page_w=page_w,
                page_h=page_h,
                col1_end=col1_end,
                col2_end=col2_end,
                content_left=content_left,
                content_right=content_right,
                binary=binary
            )
            if block:
                comm_blocks.append(block)

        for row in rows:
            is_noise_row = (
                row["height"] < min_timestamp_height
                and not row["has_speaker"]
                and row["text_density"] < min_noise_density
                and row["speaker_density"] < min_noise_density
            )
            if is_noise_row:
                continue

            is_timestamp_row = (
                row["has_timestamp"]
                and row["height"] >= min_timestamp_height
                and (row["has_speaker"] or row["text_density"] >= 0.015)
            )

            if is_timestamp_row:
                if current_rows:
                    flush_rows(current_rows)
                    current_rows = []

                if pending_rows:
                    gap = row["y"] - pending_rows[-1]["y2"]
                    if gap <= line_spacing * 2.2:
                        current_rows = pending_rows + [row]
                    else:
                        flush_rows(pending_rows)
                        current_rows = [row]
                    pending_rows = []
                else:
                    current_rows = [row]
                continue

            if current_rows:
                gap = row["y"] - current_rows[-1]["y2"]
                if gap > line_spacing * 1.8:
                    flush_rows(current_rows)
                    current_rows = []
                    pending_rows = [row]
                else:
                    current_rows.append(row)
            else:
                pending_rows.append(row)

        if current_rows:
            flush_rows(current_rows)
        elif pending_rows:
            flush_rows(pending_rows)

        return comm_blocks

    def _merge_rows_to_block(self, rows: list[dict], block_type: BlockType) -> list[Block]:
        """Merge multiple rows into a single block."""
        if not rows:
            return []

        min_x = min(r["x"] for r in rows)
        min_y = min(r["y"] for r in rows)
        max_x = max(r["x2"] for r in rows)
        max_y = max(r["y2"] for r in rows)

        return [Block(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            block_type=block_type
        )]

    def _row_to_block(self, row: dict, block_type: BlockType) -> Block:
        """Convert a row to a Block."""
        return Block(
            x=row["x"],
            y=row["y"],
            width=row["width"],
            height=row["height"],
            block_type=block_type
        )

    def _create_comm_block(
        self,
        regions: list[tuple[int, int, int, int]],
        page_w: int,
        page_h: int,
        col1_end: int,
        col2_end: int,
        content_left: int,
        content_right: int,
        binary: np.ndarray
    ) -> Optional[Block]:
        """Create a COMM block from a list of line regions."""
        if not regions:
            return None

        # Calculate bounding box for the whole COMM block
        min_x = min(r[0] for r in regions)
        min_y = min(r[1] for r in regions)
        max_x = max(r[0] + r[2] for r in regions)
        max_y = max(r[1] + r[3] for r in regions)

        pad = 3
        block_x = max(min_x - pad, 0)
        block_y = max(min_y - pad, 0)
        block_x2 = min(max_x + pad, page_w - 1)
        block_y2 = min(max_y + pad, page_h - 1)

        block = Block(
            x=block_x,
            y=block_y,
            width=block_x2 - block_x,
            height=block_y2 - block_y,
            block_type=BlockType.COMM
        )

        block.sub_columns = self._build_subcolumns_from_binary(
            binary=binary,
            y1=block.y,
            y2=block.y2,
            col1_end=col1_end,
            col2_end=col2_end,
            content_left=content_left,
            content_right=content_right
        )

        return block

    def _region_column(self, region: tuple[int, int, int, int], col1_end: int, col2_end: int) -> str:
        """Assign a region to a column based on its center."""
        x, y, w, h = region
        center_x = x + w / 2
        if center_x < col1_end:
            return "timestamp"
        if center_x < col2_end:
            return "speaker"
        return "text"

    def _build_subcolumns(
        self,
        regions: list[tuple[int, int, int, int]],
        col1_end: int,
        col2_end: int,
        content_left: int,
        content_right: int
    ) -> list[SubColumn]:
        """Build tight sub-columns from actual region geometry."""
        buckets = {"timestamp": [], "speaker": [], "text": []}

        for region in regions:
            col_type = self._region_column(region, col1_end, col2_end)
            buckets[col_type].append(region)

        subcols = []
        for col_type, col_regions in buckets.items():
            if not col_regions:
                continue
            subcol = self._merge_to_subcol(col_regions, col_type)

            # Clamp to detected content bounds
            subcol.x = max(subcol.x, content_left)
            subcol.width = max(1, min(subcol.x2, content_right) - subcol.x)

            subcols.append(subcol)

        return subcols

    def _build_subcolumns_from_binary(
        self,
        binary: np.ndarray,
        y1: int,
        y2: int,
        col1_end: int,
        col2_end: int,
        content_left: int,
        content_right: int
    ) -> list[SubColumn]:
        """Build sub-columns by scanning ink within column bands."""
        subcols = []

        columns = [
            ("timestamp", content_left, col1_end),
            ("speaker", col1_end, col2_end),
            ("text", col2_end, content_right),
        ]

        for col_type, x1, x2 in columns:
            x1 = max(0, x1)
            x2 = min(binary.shape[1] - 1, x2)
            if x2 <= x1 or y2 <= y1:
                continue

            band = binary[y1:y2, x1:x2]
            ys, xs = (band > 0).nonzero()
            if len(xs) == 0:
                continue

            min_x = x1 + int(xs.min())
            max_x = x1 + int(xs.max())
            min_y = y1 + int(ys.min())
            max_y = y1 + int(ys.max())

            subcols.append(SubColumn(
                x=min_x,
                y=min_y,
                width=max(1, max_x - min_x + 1),
                height=max(1, max_y - min_y + 1),
                col_type=col_type
            ))

        return subcols

    def _row_has_content(
        self,
        binary: np.ndarray,
        y1: int,
        y2: int,
        x1: int,
        x2: int
    ) -> bool:
        """Check if a row band has enough ink within an x-range."""
        if y2 <= y1 or x2 <= x1:
            return False

        band = binary[y1:y2, x1:x2]
        if band.size == 0:
            return False

        ink = int((band > 0).sum())
        min_ink = max(10, int(band.size * 0.003))
        return ink >= min_ink

    def _row_ink_ratio(
        self,
        binary: np.ndarray,
        y1: int,
        y2: int,
        x1: int,
        x2: int
    ) -> float:
        """Return ink density ratio for a row band within an x-range."""
        if y2 <= y1 or x2 <= x1:
            return 0.0

        band = binary[y1:y2, x1:x2]
        if band.size == 0:
            return 0.0

        return float((band > 0).sum()) / float(band.size)

    def _merge_to_subcol(
        self,
        regions: list[tuple[int, int, int, int]],
        col_type: str
    ) -> SubColumn:
        """Merge regions into a SubColumn."""
        min_x = min(r[0] for r in regions)
        min_y = min(r[1] for r in regions)
        max_x = max(r[0] + r[2] for r in regions)
        max_y = max(r[1] + r[3] for r in regions)

        return SubColumn(
            x=min_x, y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            col_type=col_type
        )
