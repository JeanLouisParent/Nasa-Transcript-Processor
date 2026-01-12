# NASA Transcript Processing Pipeline

Pipeline for processing scanned NASA mission transcripts. Performs page-by-page image enhancement, geometric layout detection, and OCR via LM Studio.

## Features

- **Page-by-page processing**: Processes one page at a time without loading entire PDF
- **Image enhancement**: Deskew, contrast improvement, noise removal, text sharpening
- **Geometric layout detection**: Detects text blocks using visual analysis (no OCR)
- **Block classification**: Identifies headers, annotations, footers, and COMM blocks
- **LM Studio OCR**: High-performance AI OCR (optimized JPEG payload, <5s/page)
- **Parallel processing**: Multi-threaded image processing with progress tracking

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ocr_transcript_v2

# Create virtual environment (Python 3.10+ required)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Process all pages (image + OCR)
python main.py process AS11_TEC.PDF

# Process specific pages
python main.py process AS11_TEC.PDF -p 1-50
python main.py process AS11_TEC.PDF -p 10,12,14-16

# Image processing only (skip OCR)
python main.py process AS11_TEC.PDF --no-ocr

# Verbose logging
python main.py process AS11_TEC.PDF -p 1-5 -v

# Clean previous output
python main.py process AS11_TEC.PDF --clean

# Custom OCR server
python main.py process AS11_TEC.PDF --ocr-url http://localhost:8080

# Get PDF info
python main.py info AS11_TEC.PDF
```

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
input_dir = "input"
output_dir = "output"
ocr_url = "http://localhost:1234"
dpi = 300
parallel = true
workers = 4
```

**Mission configs**: `config/apollo_*.toml`
```toml
file_name = "AS11_TEC.PDF"
page_offset = 0
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

## License

MIT License
