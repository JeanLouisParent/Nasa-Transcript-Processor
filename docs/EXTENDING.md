# Extension Guide

This document explains how to extend the pipeline for different missions or requirements.

## Adding a New Mission

### Step 1: Create Mission Configuration

Create a YAML configuration file for the new mission:

```yaml
# configs/apollo12.yaml

# Extraction
dpi: 300
output_format: png

# Parallelism
parallel: true
max_workers: 8

# Normalization (adjust if page size differs)
target_width: 2550
target_height: 3300
margin_px: 75

# Enhancement (adjust for scan quality)
clahe_clip_limit: 2.5
bilateral_d: 11
bilateral_sigma_color: 80
bilateral_sigma_space: 80

# Layout (adjust for column positions)
col1_end: 0.12      # Timestamp column end
col2_end: 0.28      # Speaker column end
header_ratio: 0.08  # Header height ratio

# Block detection (adjust for font size)
line_kernel_width: 45
block_kernel_height: 8
min_block_area: 800
```

### Step 2: Run with Configuration

The CLI does not accept a `--config` flag yet. Load the YAML in a small script:

```python
from pathlib import Path
from src.config import PipelineConfig
from src.pipeline import TranscriptPipeline

config = PipelineConfig.from_yaml(Path("configs/apollo12.yaml"))
pipeline = TranscriptPipeline(Path("APOLLO12.PDF"), Path("output"), config)
pipeline.process_range(0, pipeline.page_count)
```

---

## Adding a New Block Type

### Step 1: Add to BlockType Enum

Edit `src/layout_detector.py`:

```python
class BlockType(Enum):
    HEADER = "header"
    FOOTER = "footer"
    ANNOTATION = "annotation"
    COMM = "comm"
    FOOTNOTE = "footnote"  # New type
```

### Step 2: Add Detection Rule

In `LayoutDetector.detect()`, add the new rule and append rows to a new block list
(similar to header/footer/annotation handling). Then merge rows into a block.

### Step 3: Update Output Visualization

In `src/output_generator.py`, add color and drawing logic if you want it visible
in `*_blocks.png`.

---

## Adding a Processing Step

### Step 1: Add Configuration Parameters

In `src/config.py`:

```python
@dataclass
class PipelineConfig:
    # ... existing params ...

    # New: Adaptive binarization for faded text
    adaptive_binarize: bool = False
    adaptive_block_size: int = 31
    adaptive_c: int = 10
```

### Step 2: Implement Processing Step

In `src/image_processor.py`:

```python
class ImageProcessor:
    def _adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive binarization for severely faded text.

        This creates a high-contrast binary image that may improve
        readability of very faint text.
        """
        if not self.config.adaptive_binarize:
            return image

        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )
        return binary
```

### Step 3: Add to Pipeline

In `ImageProcessor.process()`:

```python
def process(self, image: np.ndarray) -> ProcessingResult:
    # ... existing steps ...

    # Step 8: Adaptive binarization (optional)
    if self.config.adaptive_binarize:
        gray = self._adaptive_binarize(gray)
        result.image = gray
        result.processing_steps.append("adaptive_binarization")

    return result
```

---

## Adding a New Output Format

### Step 1: Add Format Option

In `src/config.py`:

```python
output_format: str = "png"  # Options: png, tiff, webp, jpeg
```

### Step 2: Handle in Output Generator

In `src/output_generator.py`:

```python
def _save_enhanced(self, image: np.ndarray, path: Path) -> None:
    ext = path.suffix.lower()

    if ext == ".png":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext == ".tiff":
        cv2.imwrite(str(path), image)
    elif ext == ".webp":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, 100])
    elif ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(path), image)
```

---

## Adding OCR (Future Extension)

The pipeline is designed to support OCR as a post-processing step:

### Suggested Architecture

```python
# src/ocr_processor.py (future)

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: tuple[int, int, int, int]

class OCRProcessor:
    """
    OCR processor for extracting text from classified blocks.

    This is a future extension point. The geometric layout detection
    should be completed first to provide clean, classified blocks
    for OCR processing.
    """

    def __init__(self, engine: str = "tesseract"):
        self.engine = engine

    def process_block(
        self,
        image: np.ndarray,
        block: "ClassifiedBlock"
    ) -> OCRResult:
        """
        Extract text from a single block.

        Args:
            image: Full page image
            block: Classified block with bounding box

        Returns:
            OCRResult with extracted text
        """
        # Crop block region
        region = image[block.y:block.y2, block.x:block.x2]

        # Apply OCR (implementation depends on engine)
        # ...

        return OCRResult(text="", confidence=0.0, bounding_box=block.block.as_rect())
```

### Integration Point

In `pipeline.py`, add after classification:

```python
# Future: OCR processing
if self.config.enable_ocr:
    ocr_result = self.ocr_processor.process(
        processing_result.image,
        classification_result
    )
    result.ocr = ocr_result
```

---

## Testing Extensions

### Unit Test Template

```python
# tests/test_custom_block.py

import cv2
from src.layout_detector import LayoutDetector, BlockType

def test_custom_block_detection():
    """Test that custom blocks are detected from a fixture image."""
    image = cv2.imread("tests/fixtures/page_with_footnote.png", cv2.IMREAD_GRAYSCALE)
    detector = LayoutDetector()
    layout = detector.detect(image)

    assert any(b.block_type == BlockType.FOOTNOTE for b in layout.blocks)
```

### Integration Test

```python
# tests/test_pipeline_integration.py

def test_pipeline_with_custom_config():
    """Test pipeline with custom configuration."""
    config = PipelineConfig(
        dpi=200,
        col1_end=0.12,
        adaptive_binarize=True
    )

    pipeline = TranscriptPipeline(
        pdf_path=Path("test.pdf"),
        output_dir=Path("test_output"),
        config=config
    )

    result = pipeline.process_page(0)
    assert result.success
```
