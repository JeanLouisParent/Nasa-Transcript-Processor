# Extension Guide

How to extend the pipeline for different missions or requirements.

## Adding a New Mission

### Step 1: Create Mission Config

Create a TOML file in `config/`:

```toml
# config/apollo_12.toml
file_name = "AS12_TEC.PDF"
page_offset = 0
```

### Step 2: Adjust Parameters (if needed)

Create a custom `PipelineConfig` for different layouts:

```python
from pathlib import Path
from src.config import PipelineConfig
from src.pipeline import TranscriptPipeline

config = PipelineConfig(
    dpi=300,
    # Column positions (adjust for different layouts)
    col1_end=0.12,      # Timestamp/speaker boundary
    col2_end=0.28,      # Speaker/text boundary
    header_ratio=0.08,  # Header zone height
    # Detection sensitivity
    line_kernel_width=45,
    min_block_area=800,
)

pipeline = TranscriptPipeline(Path("APOLLO12.PDF"), Path("output"), config)
pipeline.process_range(0, pipeline.page_count)
```

### Common Adjustments

| Parameter | When to adjust |
|-----------|----------------|
| `col1_end`, `col2_end` | Different column widths |
| `header_ratio` | Larger/smaller headers |
| `line_kernel_width` | Wider/narrower character spacing |
| `clahe_clip_limit` | Low contrast scans |
| `bilateral_d` | Noisy scans |

---

## Adding a New Block Type

### Step 1: Add to Enum

Edit `src/layout_detector.py`:

```python
class BlockType(Enum):
    HEADER = "header"
    FOOTER = "footer"
    ANNOTATION = "annotation"
    COMM = "comm"
    FOOTNOTE = "footnote"  # New type
```

### Step 2: Add Detection Logic

In `LayoutDetector.detect()`, add classification rules:

```python
# Example: detect footnotes at bottom with specific pattern
if row_y > page_h * 0.9 and row_width < page_w * 0.5:
    block_type = BlockType.FOOTNOTE
```

### Step 3: Add Visualization

In `src/output_generator.py`, add color:

```python
COLORS = {
    # ... existing colors ...
    'footnote': (100, 100, 200),  # Light red
}
```

Blocks overlays are no longer generated.

---

## Adding a Processing Step

### Step 1: Add Config Parameter

In `src/config.py`:

```python
@dataclass
class PipelineConfig:
    # ... existing params ...

    # New: Adaptive binarization
    adaptive_binarize: bool = False
    adaptive_block_size: int = 31
```

### Step 2: Implement Step

In `src/image_processor.py`:

```python
def _adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
    """Apply adaptive binarization for faded text."""
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        self.config.adaptive_block_size, 10
    )
```

### Step 3: Add to Pipeline

In `ImageProcessor.process()`:

```python
if self.config.adaptive_binarize:
    gray = self._adaptive_binarize(gray)
    result.processing_steps.append("adaptive_binarization")
```

---

## Adding Output Formats

### Step 1: Update Config

In `src/config.py`, the `output_format` field already supports:
- `png` (default, lossless)
- `tiff` (lossless)
- `webp` (smaller files)

### Step 2: Add New Format

In `src/output_generator.py`:

```python
def _save_image(self, image: np.ndarray, path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".png":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext == ".webp":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, 100])
    elif ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(path), image)
```

---

## Customizing OCR

### Change Model

Edit LM Studio client parameters in `main.py`:

```python
client = LMStudioOCRClient(
    base_url=config.ocr_url,
    model="your-model-name",
    timeout_s=180,  # Increase for large pages
    max_tokens=8192,
)
```

### Custom Prompt

```python
from src.ocr.ocr_client import LMStudioOCRClient

client = LMStudioOCRClient(
    base_url="http://localhost:1234",
    model="qwen3-vl-4b",
    prompt="Extract text preserving exact layout and spacing.",
)
```

### Optional Classification Pass

Enable the second-pass classifier (OCR text + image) via config:

```toml
ocr_postprocess = "classify"
```

The classifier output is strictly validated for line count/order. If invalid, the parser falls back to the raw OCR text.

### Prompt Customization

Edit `config/prompts.toml` to override the OCR and classification prompts without changing code.

### Custom Parser

Create your own parser in `src/ocr_parser.py`:

```python
def parse_custom_format(text: str, page_num: int) -> list[dict]:
    """Parse custom OCR output format."""
    rows = []
    for line in text.splitlines():
        if line.startswith("CUSTOM:"):
            rows.append({
                "type": "custom",
                "text": line[7:].strip(),
            })
    return rows
```

---

## Testing

### Unit Test

```python
# tests/test_layout.py
import cv2
from src.layout_detector import LayoutDetector, BlockType

def test_header_detection():
    image = cv2.imread("tests/fixtures/page.png", cv2.IMREAD_GRAYSCALE)
    detector = LayoutDetector()
    layout = detector.detect(image)

    headers = [b for b in layout.blocks if b.block_type == BlockType.HEADER]
    assert len(headers) >= 1
```

### Integration Test

```python
# tests/test_pipeline.py
from pathlib import Path
from src.config import PipelineConfig
from src.pipeline import TranscriptPipeline

def test_process_page():
    config = PipelineConfig(dpi=150)  # Lower DPI for speed
    pipeline = TranscriptPipeline(
        Path("tests/fixtures/test.pdf"),
        Path("tests/output"),
        config
    )
    result = pipeline.process_page(0)
    assert result.success
```

---

## Programmatic Usage

```python
from pathlib import Path
from src.config import PipelineConfig
from src.pipeline import TranscriptPipeline
from src.ocr.ocr_client import LMStudioOCRClient
from src.ocr_parser import parse_ocr_text, build_page_json

# Process images
config = PipelineConfig(parallel=True, max_workers=8)
pipeline = TranscriptPipeline(Path("doc.pdf"), Path("output"), config)
result = pipeline.process_all()

print(f"Processed: {result.successful_pages}/{result.total_pages}")

# Run OCR separately
client = LMStudioOCRClient(base_url="http://localhost:1234", model="qwen3-vl-4b")

for page_result in result.page_results:
    if page_result.success:
        # Read enhanced image
        import cv2
        img = cv2.imread(str(page_result.output.enhanced_image), cv2.IMREAD_GRAYSCALE)

        # OCR
        # Optional classification pass (text + image)
        text = client.ocr_image(img)
        rows = parse_ocr_text(text, page_result.page_num)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        json_data = build_page_json(rows, lines, page_result.page_num)

        print(f"Page {page_result.page_num + 1}: {len(json_data['blocks'])} blocks")
```
