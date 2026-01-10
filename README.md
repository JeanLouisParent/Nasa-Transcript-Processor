# NASA Transcript Image Processing Pipeline

Industrial-grade pipeline for processing scanned NASA mission transcripts. Performs page-by-page image enhancement and geometric layout detection **without OCR**.

## Features

- **Page-by-page processing**: Strictly processes one page at a time without loading entire PDF
- **Image enhancement**: Deskew, contrast improvement, noise removal, text sharpening
- **Geometric layout detection**: Detects text blocks purely by visual/geometric analysis
- **Block classification**: Identifies headers, annotations, footers, and COMM blocks (no OCR)
- **Parallel processing**: Multi-threaded for fast processing on multi-core systems
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
python main.py process AS11_TEC.PDF --output output/

# Process specific page range
python main.py process AS11_TEC.PDF --pages 1-50 --output output/

# Process multiple ranges or pages
python main.py process AS11_TEC.PDF --pages 10,12,14-16 --output output/

# Clean previous outputs before processing
python main.py process AS11_TEC.PDF --pages 1-10 --clean --output output/

# Get PDF information
python main.py info AS11_TEC.PDF

# Show configuration
python main.py config show
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
    ├── Page_002/
    │   └── ...
```

## Input Directory

By default, the CLI looks for PDFs in `input/`. You can also pass a full path:

```bash
python main.py process input/AS11_TEC.PDF
python main.py info input/AS11_TEC.PDF
```

## CLI Reference

### process

Process a PDF document.

```bash
python main.py process <PDF_PATH> [OPTIONS]

Options:
  -o, --output PATH     Output directory (default: output/)
  -p, --pages RANGE     Page range (e.g., '1-50', '10', or '10,12,14-16')
  -w, --workers INT     Number of parallel workers (default: 4)
  --no-parallel         Disable parallel processing
  --dpi INT             Output resolution (default: 300)
  --debug               Enable debug mode
  --clean               Remove existing output directory before processing
  -v, --verbose         Enable verbose output
```

### info

Display PDF metadata.

```bash
python main.py info <PDF_PATH>
```

### config

Manage configuration.

```bash
python main.py config show              # Display default config
python main.py config save config.yaml  # Save config to file
python main.py config validate config.yaml  # Validate config file
```

## Configuration

Create a custom configuration file for different missions:

```yaml
# apollo12.yaml
dpi: 300
parallel: true
max_workers: 8

# Column boundaries (adjust for different layouts)
col1_end: 0.12    # Timestamp column
col2_end: 0.28    # Speaker column
header_ratio: 0.08

# Image enhancement
clahe_clip_limit: 2.5
bilateral_d: 11
```

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
