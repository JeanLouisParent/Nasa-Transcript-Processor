<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg" alt="NASA Logo" width="200"/>

# NASA Transcript Processing Pipeline

*Digitizing Apollo Mission Communications for the Modern Era*

</div>

## About This Project

Passionate about space exploration and the Apollo missions, I wanted to create something meaningful while learning to work with modern AI coding agents. The ultimate goal is to digitize the transcripts of all Apollo missions with the highest possible quality, making them accessible, modern, and easily searchable for everyone.

**What drives this project:**
- Deep fascination with space exploration and Apollo mission history
- Hands-on learning with AI-powered development tools (LLMs, vision models, coding agents)
- Making historic space communications more accessible and analyzable
- Building something concrete that combines classic engineering with cutting-edge AI
- Creating a foundation for reprinting these historical documents in a modern physical format

---

## Overview

The system uses a two-stage architecture:
1.  **Image Pipeline (Parallel):** Extracts pages, processes images, and generates assets.
2.  **Intelligence Pipeline (Sequential):** Runs VLM OCR, parses text, and applies corrections.

It is designed to handle the idiosyncrasies of typewriter-era documents, including faint text, margin notes, and timestamp inconsistencies.

## Prerequisites

- **Python 3.10+**
- **LM Studio** running a vision model (e.g., `qwen/qwen3-vl-4b`)
    - Server must be accessible (default: `http://localhost:1234`)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ocr_transcript_v2
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Quick Start

### 1. Basic Processing

Place your PDF in the `input/` directory (e.g., `input/AS11_TEC.PDF`).

Run the full pipeline:
```bash
python main.py process AS11_TEC.PDF
```
*This extracts pages, processes images, runs OCR, and generates structured output.*

### 2. Fast Iteration (Reparse)

If you change configuration (e.g., speaker names, text replacements) or code logic, you do not need to re-run the slow OCR step. You can reparse from the cached OCR text in minutes.

```bash
python main.py reparse AS11_TEC.PDF
```

### 3. Post-Processing Only

If you only changed block-level merging logic (no parsing changes), run the fastest update:

```bash
python main.py postprocess AS11_TEC.PDF
```

### 4. Export

To regenerate the merged JSON and text files:

```bash
python main.py export AS11_TEC.PDF
```

---

## CLI Reference

| Command | Description |
|:--------|:------------|
| `process <PDF>` | Run the full pipeline (image + OCR + export) |
| `reparse <PDF>` | Reparse pages from stored OCR text (skip OCR) |
| `postprocess <PDF>` | Post-process existing per-page JSON |
| `export <PDF>` | Regenerate merged JSON and TXT |
| `info <PDF>` | Display PDF metadata and page count |

### Common Options

- `-p, --pages TEXT`: Page range (e.g., `1-50`, `10,12-14`).
- `--no-ocr`: Skip OCR stage (image processing only).
- `--ocr-url TEXT`: Override LM Studio URL.
- `--clean`: Delete previous output before running.

---

## Output Structure

Outputs are generated in the `output/` directory:

```
output/AS11_TEC/
├── AS11_TEC_merged.json      # Complete structured transcript
├── AS11_TEC_transcript.txt   # Human-readable transcript
└── pages/
    └── Page_001/
        ├── AS11_TEC_page_0001.json  # Per-page structured data
        ├── assets/                  # Processed images
        └── ocr/                     # Raw OCR text files
```

## Configuration

Configuration is applied in layers: **Defaults** -> **Mission Overrides** -> **CLI Arguments**.

- **Global Defaults:** `config/defaults.toml` (I/O paths, OCR model, threading)
- **Mission Config:** `config/missions.toml` (Speaker lists, text replacements, page offsets)
- **Prompts:** `config/prompts.toml` (OCR instructions)

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for details.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): System design and module responsibilities.
- [Pipeline](docs/PIPELINE.md): Deep dive into image processing and OCR algorithms.
- [Post-Processing](docs/POST_PROCESSING.md): Details on parsing, correction, and validation.
- [Configuration](docs/CONFIGURATION.md): Complete configuration reference.
- [Schemas](docs/SCHEMAS.md): JSON Schema definitions.

## License

MIT License
