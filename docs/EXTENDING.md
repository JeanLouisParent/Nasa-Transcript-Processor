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

Create a custom `PipelineConfig` for different scan conditions:

```python
from pathlib import Path
from src.config import PipelineConfig
from src.pipeline import TranscriptPipeline

config = PipelineConfig(
    dpi=300,
    clahe_clip_limit=2.0,
    bilateral_d=9,
)

pipeline = TranscriptPipeline(Path("APOLLO12.PDF"), Path("output"), config)
pipeline.process_range(0, pipeline.page_count)
```

### Common Adjustments

| Parameter | When to adjust |
|-----------|----------------|
| `clahe_clip_limit` | Low contrast scans |
| `bilateral_d` | Noisy scans |


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
    prompt="Extract text with original line breaks.",
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
