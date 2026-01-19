# NASA Transcript Processing Pipeline

Pipeline for processing scanned NASA mission transcripts. Performs
page-by-page image enhancement and OCR via LM Studio with an optional
AI-assisted classification pass.

## Features

- **Page-by-page processing**: Processes one page at a time without
  loading entire PDF.
- **Image enhancement**: Deskew, normalization, spot cleaning, and
  contrast improvement.
- **Speaker Location Extraction**: Automatically identifies the origin
  of the speaker (e.g., `TRANQ`, `COLUMBIA`) and separates it from the
  dialogue.
- **Global Timestamp Indexing**: Maintains chronological integrity across
  the entire document, fixing OCR noise and duplicate timecodes.
- **LM Studio OCR**: High-performance AI OCR integration (OpenAI-compatible).
- **OCR Output**: Plain OCR output with optional column-aware mode.
- **Right-Column OCR Fill**: Optional second OCR pass for the text column
  to fill missing dialogue.
- **Header/Tape Reconstruction**: Ignores OCR page/tape lines and
  recomputes metadata consistently.
- **Parallel processing**: Multi-threaded image processing with progress
  tracking.

## Getting Started

### 1. Requirements & Prerequisites

- **Python 3.10+**
- **LM Studio** (running with a vision model like `qwen3-vl-4b`) for the
  OCR stage.
- **Poppler** (optional, for some PDF operations)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ocr_transcript_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Preparation

1. Place your source PDF files in the `input/` directory.

2. Ensure your LM Studio server is running and accessible
   (default: `http://localhost:1234`).

3. Check `config/missions.toml` if your mission requires specific page
   offsets.

---

## Usage Guide

The pipeline is operated via the `main.py` CLI.

### Processing a Transcript

The `process` command is the main entry point. It first runs the **Image
Pipeline** (extraction & enhancement) and then triggers the **OCR Loop**.

```bash
# Basic processing (looks for the file in the 'input/' folder)
python main.py process AS11_TEC.PDF

# Processing specific page ranges
# Format: 'start-end', 'single', or 'multiple,ranges'
python main.py process AS11_TEC.PDF --pages 1-50
python main.py process AS11_TEC.PDF --pages 10,12,14-16

# Skip OCR (Image processing only)
python main.py process AS11_TEC.PDF --no-ocr

# Clean previous output before starting
python main.py process AS11_TEC.PDF --clean

# Overriding OCR URL at runtime
python main.py process AS11_TEC.PDF --ocr-url http://192.168.1.50:1234

# Use column-aware prompt (no tags)
python main.py process AS11_TEC.PDF --ocr-prompt column

# Print per-page timing breakdowns
python main.py process AS11_TEC.PDF --pages 1-5 --timing
```

### Checking PDF Info

To verify the number of pages and basic metadata before processing:

```bash
python main.py info AS11_TEC.PDF
```

---

## Configuration

**Global defaults**: `config/defaults.toml`

```toml
# I/O
input_dir = "input"
output_dir = "output"

# OCR Settings
ocr_url = "http://localhost:1234"
ocr_model = "qwen/qwen3-vl-4b"
ocr_timeout = 120
ocr_max_tokens = 4096
ocr_prompt = "plain" # "plain" or "column"
ocr_text_column_pass = true

# Processing Settings
dpi = 300
parallel = true
workers = 4
timing = true

# Image Enhancement
clahe_clip_limit = 2.0  # (Legacy param, kept for compatibility)
bilateral_d = 9         # (Legacy param, kept for compatibility)
deskew_angle_threshold = 0.5

# Right-Column Crop Settings
# Used ONLY for the optional 'ocr_text_column_pass' to isolate the text column
col2_end = 0.30
header_ratio = 0.10
```

**Mission configs**: `config/missions.toml`

```toml
[mission.11]
file_name = "AS11_TEC.PDF"
page_offset = -2
```

## Prompts

OCR and classification prompts live in `config/prompts.toml`.
See `docs/PROMPTS.md` for the full reference.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and data flow.
- [Pipeline](docs/PIPELINE.md) - Detailed breakdown of processing stages.
- [Post-Processing](docs/POST_PROCESSING.md) - Text intelligence and
  structural parsing.
- [Extending](docs/EXTENDING.md) - Adding missions, block types,
  processing steps.

## License

MIT License
