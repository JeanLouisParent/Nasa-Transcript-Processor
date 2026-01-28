# NASA Transcript Processing Pipeline

A specialized pipeline for digitizing scanned NASA Apollo mission transcripts. Combines intelligent image enhancement with Vision-Language Model (VLM) OCR to produce structured, searchable transcripts.

```mermaid
flowchart LR
    PDF[/"PDF Scans"/] --> IMG["Image<br/>Processing"]
    IMG --> OCR["Multi-pass<br/>VLM OCR"]
    OCR --> INT["Text<br/>Intelligence"]
    INT --> OUT[/"JSON + TXT"/]
```

## Features

| Feature | Description |
|:--------|:------------|
| **Smart Image Enhancement** | Deskew, normalization, noise removal, and contrast optimization |
| **Multi-pass OCR** | Primary + raw + faint fallback passes with intelligent merge |
| **Structured Parsing** | Extracts timestamps, speakers, locations, and dialogue |
| **Timestamp Recovery** | Maintains chronological integrity across pages |
| **Speaker Correction** | Fuzzy matching against mission-specific callsigns |
| **Global Export** | Merged JSON and formatted text transcripts |

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **LM Studio** running a vision model (e.g., `qwen3-vl-4b`)

### Installation

```bash
git clone <repository-url>
cd ocr_transcript_v2

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Basic Usage

```bash
# Place your PDF in input/ then run:
python main.py process AS11_TEC.PDF

# Process specific pages
python main.py process AS11_TEC.PDF --pages 1-50

# Image processing only (no OCR)
python main.py process AS11_TEC.PDF --no-ocr
```

---

## CLI Reference

### Commands

| Command | Description |
|:--------|:------------|
| `process <PDF>` | Run the full pipeline (image + OCR + export) |
| `export <PDF>` | Regenerate merged JSON and TXT from existing page data |
| `info <PDF>` | Display PDF metadata and page count |

### Process Options

```
-p, --pages TEXT        Page range (e.g., "1-50", "10,12,14-16")
--clean                 Delete previous output before running
--no-ocr                Skip OCR stage (image processing only)
--ocr-url TEXT          Override LM Studio URL
--ocr-prompt [plain|column]  OCR prompt mode
--timing / --no-timing  Show per-page timing breakdowns
-v, --verbose           Verbose logging to pipeline.log
```

### Examples

```bash
# Process pages 100-150 with timing info
python main.py process AS11_TEC.PDF --pages 100-150 --timing

# Clean start with custom OCR server
python main.py process AS11_TEC.PDF --clean --ocr-url http://192.168.1.50:1234

# Export only (after prior processing)
python main.py export AS11_TEC.PDF
```

---

## Output Structure

```
output/
└── AS11_TEC/
    ├── AS11_TEC_merged.json      # Complete structured transcript (pages keyed as "Page 001")
    ├── AS11_TEC_transcript.txt   # Human-readable transcript
    └── pages/
        └── Page_001/
            ├── AS11_TEC_page_0001.json  # Per-page structured data
            ├── assets/
            │   ├── *_enhanced.png       # Processed image (sent to OCR)
            │   ├── *_raw.png            # Original render
            │   └── *_faint.png          # High-contrast fallback
            └── ocr/
                ├── *_ocr_raw.txt        # Primary OCR output
                └── *_ocr_*.txt          # Fallback passes

state/
└── AS11_TEC_timestamps_index.json   # Cross-page timestamp continuity
```

---

## Configuration

Configuration is layered: **defaults** → **mission overrides** → **CLI arguments**

### Global Defaults

`config/defaults.toml` — Applies to all runs:

```toml
# I/O directories
input_dir = "input"
output_dir = "output"
state_dir = "state"

# OCR
ocr_url = "http://localhost:1234"
ocr_model = "qwen/qwen3-vl-4b"
ocr_timeout = 120
ocr_max_tokens = 4096

# Multi-pass OCR (recommended)
ocr_dual_pass = true      # Raw image fallback
ocr_faint_pass = true     # High-contrast fallback
ocr_text_column_pass = true  # Right-column fill

# Processing
dpi = 300
parallel = true
workers = 4
```

### Mission Overrides

`config/missions.toml` — Per-mission settings:

```toml
[mission.11]
file_name = "AS11_TEC.PDF"
page_offset = -2
valid_speakers = ["CDR", "CC", "CMP", "LMP", "SC", "HOUSTON"]
valid_locations = ["TRANQ", "COLUMBIA", "EAGLE"]
```

### OCR Prompts

`config/prompts.toml` — Customize VLM instructions without code changes.

See [Configuration Reference](docs/CONFIGURATION.md) for complete details.

---

## Documentation

| Document | Content |
|:---------|:--------|
| [Architecture](docs/ARCHITECTURE.md) | System design, data structures, module responsibilities |
| [Pipeline](docs/PIPELINE.md) | Image processing stages, OCR strategy |
| [Post-Processing](docs/POST_PROCESSING.md) | Parsing algorithms, correction logic |
| [Configuration](docs/CONFIGURATION.md) | Complete configuration reference |
| [Schemas](docs/SCHEMAS.md) | JSON Schema definitions for output validation |

---

## License

MIT License
