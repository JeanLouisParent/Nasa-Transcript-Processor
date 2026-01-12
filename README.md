# NASA Transcript Image Processing Pipeline

Pipeline for processing scanned NASA mission transcripts. Performs page-by-page image enhancement and geometric layout detection, then runs OCR via LM Studio.

## Features

- **Page-by-page processing**: Strictly processes one page at a time without loading entire PDF
- **Image enhancement**: Deskew, contrast improvement, noise removal, text sharpening
- **Geometric layout detection**: Detects text blocks purely by visual/geometric analysis
- **Block classification**: Identifies headers, annotations, footers, and COMM blocks (no OCR)
- **LM Studio OCR**: Sends enhanced pages for page-level text extraction
- **Extensible**: Configurable for other NASA missions with different layouts

## Installation

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

## Quick Start

```bash
# Place PDFs in input/ (or pass a full path)
mkdir -p input

# Process all pages
python main.py process AS11_TEC.PDF

# Process specific page range
python main.py process AS11_TEC.PDF --pages 1-50

# Process multiple ranges or pages
python main.py process AS11_TEC.PDF --pages 10,12,14-16

# Clean previous outputs before processing
python main.py process AS11_TEC.PDF --pages 1-10 --clean

# Override LM Studio URL
python main.py process AS11_TEC.PDF --pages 1-5 --ocr-url http://localhost:1234

# Get PDF information
python main.py info AS11_TEC.PDF

```

## Output Structure

For each page, the pipeline generates:

```
output/
└── AS11_TEC/
    ├── Page_001/
    │   ├── AS11_TEC_page_0001_raw.pdf       # Original single-page PDF
    │   ├── AS11_TEC_page_0001_enhanced.png  # Processed/enhanced image
    │   └── AS11_TEC_page_0001_blocks.png    # Block overlays
    │   └── AS11_TEC_page_0001.json          # OCR blocks
    ├── Page_002/
    │   └── ...
```

## Input Directory

By default, the CLI looks for PDFs in `input/`. You can also pass a full path:

```bash
python main.py process input/AS11_TEC.PDF
python main.py info input/AS11_TEC.PDF
```

Defaults for input/output directories and LM Studio URL live in
`config/defaults.toml`.

## CLI Reference

### process

Process a PDF document.

```bash
python main.py process <PDF_PATH> [OPTIONS]

Options:
  -p, --pages RANGE     Page range (e.g., '1-50', '10', or '10,12,14-16')
  --clean               Remove existing output directory before processing
  --ocr-url TEXT        LM Studio base URL (overrides config/defaults.toml)
```

### info

Display PDF metadata.

```bash
python main.py info <PDF_PATH>
```

## Configuration

Global defaults live in `config/defaults.toml`. Mission-specific offsets
live in `config/*.toml` (for example `config/apollo_11.toml`).

Load in a custom script via `PipelineConfig.from_yaml()` and pass it to
`TranscriptPipeline` (the CLI does not currently accept a `--config` flag).

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Pipeline Stages

See [docs/PIPELINE.md](docs/PIPELINE.md) for pipeline stage documentation.

## Extending

See [docs/EXTENDING.md](docs/EXTENDING.md) for extension guide.

## Requirements

- Python 3.10+
- pymupdf (PDF processing)
- opencv-python (image processing)
- numpy (array operations)
- click (CLI)
- tqdm (progress bars)
- loguru (logging)

## License

MIT License
