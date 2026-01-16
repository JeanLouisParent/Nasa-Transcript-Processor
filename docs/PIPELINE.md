# Pipeline Documentation

This document describes each stage of the processing pipeline.

## Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0B3D91', 'primaryTextColor': '#fff', 'primaryBorderColor': '#FC3D21', 'lineColor': '#8BA1B4', 'secondaryColor': '#8BA1B4', 'tertiaryColor': '#fff'}}}%%
flowchart LR
    A[Extraction] --> B[Processing]
    B --> C[Detection]
    C --> D[Generation]
    D --> E[OCR & Intelligence]
    
    style A fill:#0B3D91,stroke:#FC3D21,color:#fff
    style B fill:#0B3D91,stroke:#FC3D21,color:#fff
    style C fill:#0B3D91,stroke:#FC3D21,color:#fff
    style D fill:#0B3D91,stroke:#FC3D21,color:#fff
    style E fill:#FC3D21,stroke:#0B3D91,color:#fff
```

---

## Stage 1: Page Extraction

**Module**: `page_extractor.py`

Extracts individual pages from PDF as high-resolution images.

**Operations**:

1. Open PDF with pymupdf
2. Load specific page by index
3. Render at target DPI (default: 300)
4. Convert to numpy array (BGR format)
5. Optionally extract single-page PDF

**Output**:

- `numpy.ndarray`: BGR image at target resolution
- `*_raw.pdf`: Single-page PDF file

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dpi` | 300 | Output resolution (72-600) |

---

## Stage 2: Image Processing

**Module**: `image_processor.py`

Enhances scanned images for better OCR and layout detection.

### 2.1 Grayscale Conversion

Convert BGR to grayscale for consistent processing.

### 2.2 Deskew

Detect and correct page rotation using Hough line detection.

**Algorithm**:

1. Create binary image (Otsu threshold)
2. Detect edges (Canny)
3. Find lines (Probabilistic Hough Transform)
4. Calculate median angle of near-horizontal lines
5. Rotate if angle exceeds threshold

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `deskew_angle_threshold` | 0.5° | Minimum angle to correct |
| `deskew_max_angle` | 10.0° | Maximum expected skew |

### 2.3 Size Normalization

Standardize page dimensions to Letter size at 300 DPI.

**Algorithm**:

1. Find content bounding box (non-white pixels)
2. Crop to content
3. Scale to fit target area (preserve aspect ratio)
4. Center on canvas with uniform margins

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_width` | 2550 px | 8.5" at 300 DPI |
| `target_height` | 3300 px | 11" at 300 DPI |
| `margin_px` | 75 px | ~0.25" margin |

### 2.4 CLAHE Contrast Enhancement

Improve local contrast using Contrast Limited Adaptive Histogram Equalization.

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `clahe_clip_limit` | 2.0 | Contrast limit |
| `clahe_grid_size` | 8 | Tile size |

### 2.5 Bilateral Noise Removal

Reduce noise while preserving edges.

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bilateral_d` | 9 | Filter diameter |
| `bilateral_sigma_color` | 75 | Color sigma |
| `bilateral_sigma_space` | 75 | Space sigma |

### 2.6 Spot Cleaning

Remove small artifacts using connected component analysis.

**Algorithm**:

1. Threshold to binary (dark pixels = foreground)
2. Find connected components
3. Remove components < 15 px² or small squares < 50 px²
4. Replace with white

### 2.7 Unsharp Mask

Enhance text edges for better readability.

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `unsharp_amount` | 1.5 | Sharpening strength |
| `unsharp_sigma` | 1.0 | Blur sigma |

---

## Stage 3: Layout Detection

**Module**: `layout_detector.py`

Detects text blocks using geometric analysis (no OCR).

### 3.1 Binarization

Convert to binary using adaptive threshold.

```python
binary = cv2.adaptiveThreshold(image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, blockSize=15, C=5)
```

### 3.2 Text Connection

Connect characters into blocks using morphological dilation.

**Horizontal** (connect characters in lines):

```python
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
```

**Vertical** (connect lines in blocks):

```python
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
```

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `line_kernel_width` | 50 | Horizontal connection |
| `line_kernel_height` | 1 | |
| `block_kernel_width` | 5 | Vertical connection |
| `block_kernel_height` | 10 | |

### 3.3 Block Detection

Find contours, filter by size, merge overlaps.

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_block_area` | 1000 px² | Minimum block size |
| `max_block_area_ratio` | 0.9 | Maximum as ratio of page |

### 3.4 Block Classification

Classify blocks using geometric heuristics.

**Block Types**:

- **HEADER**: Top zone, network/page/tape markers
- **FOOTER**: Bottom zone, asterisk lines
- **ANNOTATION**: Centered, isolated vertically
- **COMM**: Triplet structure (timestamp | speaker | text)

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `col1_end` | 0.15 | Timestamp/speaker boundary |
| `col2_end` | 0.30 | Speaker/text boundary |
| `header_ratio` | 0.10 | Header zone height |

---

## Stage 4: Output Generation

**Module**: `output_generator.py`

Generates output files for each processed page.

### Files Generated

| File             | Description                 |
| ---------------- | --------------------------- |
| `*_raw.pdf`      | Single page from source PDF |
| `*_enhanced.png` | Processed grayscale image   |

---

## Stage 5: OCR (Optional)

**Modules**: `ocr_client.py`, `ocr_parser.py`

Sends enhanced images to LM Studio for text extraction. Optionally runs a second AI pass to tag each line using the OCR text plus the page image.

Prompt text is loaded from `config/prompts.toml` when present. See `docs/PROMPTS.md`.

### OCR Client

Uses OpenAI-compatible API with optimized workflow for speed:

1. **Compression**: Encodes image as JPEG (85% quality) to minimize payload
2. **Standard First**: Tries standard OpenAI `image_url` format (fastest)
3. **Fallback**: Retries with various vision tokens (`<image>`, `<img>`) if standard fails
4. **Validation**: Checks for minimum alphabetic content (2+ chars) to avoid false negatives

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| Model | qwen3-vl-4b | Vision model |
| Timeout | 120s | Request timeout |
| Max tokens | 4096 | Response limit |

### Optional Classification Pass

When `ocr_postprocess = "classify"`:

- **Input**: OCR text (line-numbered) + the page image.
- **Output**: Same number of lines, same order, each prefixed with a tag:
  `HEADER`, `COMM`, `ANNOTATION`, `FOOTER`, `META`.
- **Guardrails**: The result is rejected if the line count or numbering does not match.


### OCR Parser

Parses plain text into structured blocks. When classification is enabled, it consumes tagged lines to improve block labeling, footer detection, and annotation handling.

**Detection patterns**:

- Timestamp: `\d{2} \d{2} \d{2} \d{1,2}`
- Speaker: `[A-Z][A-Z0-9]{1,6}`
- Header keywords: GOSS, NET, TAPE, PAGE, APOLLO

### Output Format

**JSON** (`*_page_XXXX.json`):

```json
{
  "header": {
    "page": 42,
    "tape": "1/2",
    "is_apollo_title": true
  },
  "blocks": [
    {
      "type": "comm",
      "timestamp": "00 00 00 00",
      "speaker": "CDR",
      "text": "Roger, Houston."
    },
    {
      "type": "continuation",
      "text": "We copy.",
      "continuation_from_prev": true
    }
  ]
}
```

**Raw text** (`*_ocr_raw.txt`): Unprocessed OCR output.
**Classified text** (`*_ocr_classified.txt`): Tagged OCR lines (only when classification is accepted).
**Rejected classification** (`*_ocr_classified_rejected.txt`): Classifier output that failed validation.

### Skip OCR

Use `--no-ocr` to skip this stage:

```bash
python main.py process AS11_TEC.PDF --no-ocr
```
