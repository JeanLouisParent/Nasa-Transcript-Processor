# NASA Transcript Processing Pipeline

Pipeline for processing scanned NASA mission transcripts. Performs page-by-page image enhancement, geometric layout detection, and OCR via LM Studio.

## Features

- **Page-by-page processing**: Processes one page at a time without loading entire PDF
- **Image enhancement**: Deskew, contrast improvement, noise removal, text sharpening
- **Geometric layout detection**: Detects text blocks using visual analysis (no OCR)
- **Block classification**: Identifies headers, annotations, footers, and COMM blocks
- **LM Studio OCR**: High-performance AI OCR (optimized JPEG payload, <5s/page)
- **Parallel processing**: Multi-threaded image processing with progress tracking

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
```

### Checking PDF Info
To verify the number of pages and basic metadata before processing:
```bash
python main.py info AS11_TEC.PDF
```

---

## CLI Reference

### `process` Arguments
| Option | Short | Description |
| :--- | :--- | :--- |
| `--pages` | `-p` | Specific pages to process (e.g., `1-10,15`). |
| `--clean` | | Deletes the output directory for this PDF before starting. |
| `--no-ocr` | | Runs the vision pipeline but skips the AI OCR stage. |
| `--ocr-url` | | Overrides the OCR server URL defined in `defaults.toml`. |
| `--verbose` | `-v` | Enables DEBUG level logs for detailed troubleshooting. |

---

## Post-Processing Intelligence
This pipeline includes advanced post-processing to ensure high accuracy (~95%):
- **Iterative Splitting**: Automatically separates dialogue from station annotations (e.g., `GRAND BAHAMA`).
- **Lexicon Protection**: Technical terms like `GUAYMAS` or `REFSMMAT` are protected from being incorrectly "fixed" into common words.
- **Visual Scoring**: Text correction prioritizes visual similarity over word frequency.
- **Smart Merging**: Multi-line dialogues are merged into clean paragraphs in the JSON output.

See the [Post-Processing Documentation](docs/POST_PROCESSING.md) for more details.

## Output Structure

```
output/
└── AS11_TEC/
    └── Page_001/
        ├── AS11_TEC_page_0001_raw.pdf       # Original page
        ├── AS11_TEC_page_0001_enhanced.png  # Enhanced image
        ├── AS11_TEC_page_0001_blocks.png    # Layout visualization
        ├── AS11_TEC_page_0001_ocr_raw.txt   # Raw OCR text
        └── AS11_TEC_page_0001.json          # Structured blocks
```

## JSON Output Format

```json
{
  "page": {
    "number": 42,
    "tape": "1/2",
    "apollo": "APOLLO 11 AIR-TO-GROUND VOICE TRANSCRIPTION"
  },
  "blocks": [
    {"type": "comm", "timestamp": "00 00 00 00", "speaker": "CDR", "text": "..."},
    {"type": "continuation", "text": "..."},
    {"type": "annotation", "text": "..."}
  ]
}
```

## CLI Reference

### process

```
python main.py process <PDF> [OPTIONS]

Options:
  -p, --pages TEXT    Page range: '1-50', '10', '10,12,14-16'
  --clean             Remove existing output first
  --no-ocr            Skip OCR step
  --ocr-url TEXT      LM Studio URL (overrides config)
  -v, --verbose       Enable debug logging
```

### info

```
python main.py info <PDF>
```

## Configuration

**Global defaults**: `config/defaults.toml`
```toml
# I/O
input_dir = "input"
output_dir = "output"

# OCR Settings
ocr_url = "http://localhost:1234"
ocr_model = "qwen3-vl-4b"
ocr_timeout = 120
ocr_max_tokens = 4096

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
- [Extending](docs/EXTENDING.md) - Adding missions, block types, processing steps

## Lexicon & Assets

The project includes a specialized lexicon generated from Apollo 11 ground truth data to improve OCR accuracy.
- **Lexicon**: `assets/lexicon/apollo11_lexicon.json`
- **Source**: The underlying data (`a11tec.csv`) is derived from the official Apollo 11 Technical Transcript: [Apollo Flight Journal](https://apollojournals.org/alsj/a11/a11transcript_tec.html)

## License

MIT License
