# NASA Transcript Processing Pipeline

Pipeline for processing scanned NASA mission transcripts. Performs page-by-page image enhancement, geometric layout detection, and OCR via LM Studio with an optional AI-assisted classification pass.

## Features

- **Page-by-page processing**: Processes one page at a time without loading entire PDF.
- **Image enhancement**: Deskew, contrast improvement, noise removal, text sharpening.
- **Geometric layout detection**: Detects text blocks using visual analysis (no OCR).
- **Speaker Location Extraction**: Automatically identifies the origin of the speaker (e.g., `TRANQ`, `COLUMBIA`) and separates it from the dialogue.
- **Global Timestamp Indexing**: Maintains chronological integrity across the entire document, fixing OCR noise and duplicate timecodes.
- **LM Studio OCR**: High-performance AI OCR (optimized JPEG payload, <5s/page).
- **OCR Classification (Optional)**: Second-pass AI tagging using OCR text + image to improve block detection and light OCR cleanup (no hallucination).
- **Parallel processing**: Multi-threaded image processing with progress tracking.

## Getting Started

### 1. Requirements & Prerequisites
- **Python 3.10+**
- **LM Studio** (running with a vision model like `qwen3-vl-4b`) for the OCR stage.
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
1.  Place your source PDF files in the `input/` directory.
2.  Ensure your LM Studio server is running and accessible (default: `http://localhost:1234`).
3.  Check `config/missions.toml` if your mission requires specific page offsets or column boundary overrides.

---

## Usage Guide

The pipeline is operated via the `main.py` CLI. It currently supports two main commands: `process` and `info`.

### Processing a Transcript
The `process` command is the main entry point. It extracts pages, enhances images, detects layout, and performs OCR.

```bash
# Basic processing (looks for the file in the 'input/' folder)
python main.py process AS11_TEC.PDF

# Processing specific page ranges
# Format: 'start-end', 'single', or 'multiple,ranges'
python main.py process AS11_TEC.PDF --pages 1-50
python main.py process AS11_TEC.PDF --pages 10,12,14-16

# Skip OCR (useful for testing image enhancement or layout detection)
python main.py process AS11_TEC.PDF --no-ocr

# Clean previous output before starting
python main.py process AS11_TEC.PDF --clean

# Overriding OCR URL at runtime
python main.py process AS11_TEC.PDF --ocr-url http://192.168.1.50:1234

# Enable AI classification pass (text + image)
python main.py process AS11_TEC.PDF --ocr-postprocess classify

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
ocr_model = "gemma3-12b"
ocr_timeout = 120
ocr_max_tokens = 4096
ocr_prompt = "structured" # "structured" or "plain"
ocr_postprocess = "classify"  # "none" or "classify"

# Processing Settings
dpi = 300
parallel = true
workers = 4

# Image Enhancement
clahe_clip_limit = 2.0
bilateral_d = 9
unsharp_amount = 1.5
deskew_angle_threshold = 0.5

# Layout Detection
col1_end = 0.15
col2_end = 0.30
header_ratio = 0.10
```

**Mission configs**: `config/missions.toml`
```toml
[mission.11]
file_name = "AS11_TEC.PDF"
page_offset = 0
col1_end = 0.15  # Optional override
```

## Prompts

OCR and classification prompts live in `config/prompts.toml`. See `docs/PROMPTS.md` for the full reference.

## Requirements

- Python 3.10+
- pymupdf (PDF processing)
- opencv-python (image processing)
- numpy (array operations)
- click (CLI)
- tqdm (progress bars)
- loguru (logging)

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Module structure and data flow
- [Pipeline](docs/PIPELINE.md) - Processing stages in detail
- [Post-Processing](docs/POST_PROCESSING.md) - Text intelligence and structural parsing
- [Extending](docs/EXTENDING.md) - Adding missions, block types, processing steps

## Lexicon & Assets

The project includes a specialized lexicon generated from Apollo 11 ground truth data to improve OCR accuracy.
- **Lexicon**: `assets/lexicon/apollo11_lexicon.json`
- **Source**: The underlying data (`a11tec.csv`) is derived from the official Apollo 11 Technical Transcript: [Apollo Flight Journal](https://apollojournals.org/alsj/a11/a11transcript_tec.html)

## License

MIT License
