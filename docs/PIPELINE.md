# Pipeline Documentation

This document describes each stage of the image processing pipeline in detail.

## Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Extract    │ → │  Process    │ → │   Detect    │ → │  Classify   │
│    Page     │    │   Image     │    │   Layout    │    │   Blocks    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
                                                               ▼
                                                        ┌─────────────┐
                                                        │  Generate   │
                                                        │   Output    │
                                                        └─────────────┘
```

## Stage 1: Page Extraction

**Module**: `page_extractor.py`

### Input
- PDF file path
- Page number (0-indexed)

### Operations
1. Open PDF with pymupdf
2. Load specific page by index
3. Create pixmap at target DPI (default: 300)
4. Convert to numpy array (BGR format)
5. Optionally extract single-page PDF

### Output
- `numpy.ndarray`: BGR image at target resolution
- Single-page PDF file

### Parameters
- `dpi`: Resolution (72-600, default: 300)

---

## Stage 2: Image Processing

**Module**: `image_processor.py`

### 2.1 Grayscale Conversion

Convert BGR to grayscale for consistent processing.

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### 2.2 Deskew (Rotation Correction)

Detect and correct page skew using line detection.

**Algorithm**:
1. Create binary image (Otsu thresholding)
2. Detect edges (Canny)
3. Find lines (Probabilistic Hough Transform)
4. Calculate median angle of near-horizontal lines
5. Rotate image if angle > threshold

**Parameters**:
- `deskew_angle_threshold`: Minimum angle to correct (default: 0.5°)
- `deskew_max_angle`: Maximum expected skew (default: 10°)

### 2.3 Size Normalization

Standardize page dimensions and margins.

**Algorithm**:
1. Find content bounding box (non-white pixels)
2. Crop to content
3. Scale to fit target area (preserving aspect ratio)
4. Center on canvas with uniform margins

**Parameters**:
- `target_width`: 2550 px (8.5" at 300 DPI)
- `target_height`: 3300 px (11" at 300 DPI)
- `margin_px`: 75 px (~0.25")

### 2.4 Contrast Enhancement (CLAHE)

Improve local contrast using Contrast Limited Adaptive Histogram Equalization.

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(image)
```

**Parameters**:
- `clahe_clip_limit`: Contrast limit (default: 2.0)
- `clahe_grid_size`: Tile size (default: 8)

### 2.5 Noise Removal (Bilateral Filter)

Reduce noise while preserving edges.

```python
denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
```

**Parameters**:
- `bilateral_d`: Filter diameter (default: 9)
- `bilateral_sigma_color`: Color sigma (default: 75)
- `bilateral_sigma_space`: Space sigma (default: 75)

### 2.6 Spot Cleaning (Morphology)

Remove small artifacts and spots.

**Algorithm**:
1. Invert image (text = white)
2. Apply morphological opening (remove small white spots)
3. Find connected components
4. Remove components smaller than threshold
5. Invert back

**Parameters**:
- `morph_kernel_size`: Opening kernel size (default: 2)
- `noise_max_area`: Maximum noise component area (default: 50 px²)

### 2.7 Text Sharpening (Unsharp Mask)

Enhance text edges and partially faded characters.

```python
blurred = cv2.GaussianBlur(image, (0, 0), sigma)
sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
```

**Parameters**:
- `unsharp_amount`: Sharpening strength (default: 1.5)
- `unsharp_sigma`: Blur sigma (default: 1.0)

---

## Stage 3: Layout Detection

**Module**: `layout_detector.py`

### 3.1 Binarization

Convert to binary for contour detection.

```python
binary = cv2.adaptiveThreshold(
    image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=15, C=5
)
```

### 3.2 Text Connection

Connect characters into blocks using morphological dilation.

**Horizontal Dilation** (connect characters in lines):
```python
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
connected = cv2.dilate(binary, h_kernel)
```

**Vertical Dilation** (connect lines in blocks):
```python
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
connected = cv2.dilate(connected, v_kernel)
```

**Parameters**:
- `line_kernel_width`: 50 (horizontal connection)
- `line_kernel_height`: 1
- `block_kernel_width`: 5 (vertical connection)
- `block_kernel_height`: 10

### 3.3 Contour Detection

Find block boundaries.

```python
contours, _ = cv2.findContours(
    connected,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

### 3.4 Filtering

Remove invalid blocks.

**Criteria**:
- Area > `min_block_area` (default: 1000 px²)
- Area < page_area × `max_block_area_ratio` (default: 0.9)
- Content density > 1%

### 3.5 Merging

Merge overlapping blocks.

**Overlap threshold**: 30% of smaller block area

### 3.6 Sorting

Sort in reading order (top-to-bottom, left-to-right).

---

## Stage 4: Layout Detection

**Module**: `layout_detector.py`

This stage groups line regions into rows, detects column boundaries, and
classifies rows into HEADER, FOOTER, ANNOTATION, and COMM blocks using
geometric heuristics (no OCR).

### Key Heuristics
- Column boundaries from vertical ink projection with fallback to defaults when timestamps are sparse.
- Row clustering merges overlapping regions to avoid duplicate rows.
- HEADER detection via left/right/center marker geometry; header zone expands
  down toward the first COMM row when needed, and a fallback ensures at least
  one header per page.
- COMM grouping starts on timestamp-like rows; continuation lines follow until
  a large vertical gap; tiny noise rows are ignored.
- FOOTER detection uses bottom-zone geometry and column-boundary ink density.

### Parameters
- `col1_end`: 0.15 (timestamp/speaker boundary)
- `col2_end`: 0.30 (speaker/text boundary)
- `header_ratio`: 0.10 (baseline header region height)

---

## Stage 5: Output Generation

**Module**: `output_generator.py`

### Files Generated

| File | Description |
|------|-------------|
| `<PDF>_page_XXXX_raw.pdf` | Single page extracted from source |
| `<PDF>_page_XXXX_enhanced.png` | Processed grayscale image |
| `<PDF>_page_XXXX_blocks.png` | Enhanced image with colored block overlays |

### Blocks Visualization Colors

| Block Type | Color (BGR) |
|------------|-------------|
| HEADER | Blue (255, 150, 50) |
| FOOTER | Gray (150, 150, 150) |
| ANNOTATION | Magenta (255, 100, 255) |
| COMM (outline) | Green (100, 200, 100) |
| COMM (fill) | Light green (190, 230, 190) |

### Blocks Overlay

- HEADER/FOOTER/ANNOTATION: semi-transparent fill + label
- COMM: light green fill + outline (no label)

---

## Stage 6: OCR (LM Studio)

**Module**: `ocr_client.py`

This stage sends the enhanced page image to a local LM Studio server using
the OpenAI-compatible API. It produces a per-page raw text file plus a JSON
with normalized blocks.

**Command**:
```bash
python main.py process AS11_TEC.PDF --pages 1-5 --ocr-url http://localhost:1234
```

**Output**:
`output/<PDF>/Page_XXX/<PDF>_page_XXXX.json` (per-page OCR blocks)

The JSON includes page header info plus a list of blocks:
- `type`
- `timestamp` and `speaker` for comm blocks (when present)
- `text`
