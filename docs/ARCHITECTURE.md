# Architecture Documentation

This document describes the architecture of the NASA Transcript Processing
Pipeline.

## Overview

```mermaid
flowchart TD
    CLI[main.py CLI] --> ORCH[pipeline.py Orchestrator]

    subgraph Core[Core Engine]
        ORCH --> CFG[config.py]
    end

    subgraph Processors[Vision Stack]
        EXT[page_extractor.py]
        IMG[image_processor.py]
    end

    subgraph Intelligence[OCR and Post-Processing]
        CLIENT[ocr_client.py]
        PARSER[ocr_parser.py]
        CORR[correctors/*.py]
    end

    ORCH --> Processors
    ORCH --> Intelligence
    PARSER --> CORR
    Intelligence --> OUT[utils/output_generator.py]
    CLIENT --> PARSER
```

## Module Responsibilities

### config.py

**Purpose**: Centralized configuration management

**Key Classes**:

- `PipelineConfig`: Dataclass with all configurable parameters (~30 fields)

**Key Parameters**:

- `dpi`: Output resolution (default: 300)
- `parallel`: Enable parallel processing (default: True)
- `max_workers`: Number of workers (default: 4)
- `col1_end`, `col2_end`: Column boundary ratios
- `header_ratio`: Header zone height ratio

**Methods**:

- `validate()`: Returns list of validation errors

### global_config.py

**Purpose**: Load global defaults from TOML

**Key Classes**:

- `GlobalConfig`: Input/output directories, OCR URL, workers

**Usage**:

```python
config = load_global_config(Path("config/defaults.toml"))
```

### mission_config.py

**Purpose**: Load mission-specific settings from TOML

**Key Classes**:

- `MissionConfig`: Page offset, file name matching

**Usage**:

```python
config = load_mission_config(Path("config"), "AS11_TEC.PDF")
```

### page_extractor.py

**Purpose**: Extract pages from PDF without loading entire document

**Key Classes**:

- `PageExtractor`: Main extraction class

**Key Methods**:

- `extract_page_image(page_num)`: Get page as numpy array (BGR)
- `extract_page_pdf(page_num, path)`: Extract single page PDF
- `iter_pages(start, end)`: Lazy page iterator
- `get_page_info(page_num)`: Get page metadata

**Thread Safety**: Yes (each call opens/closes PDF)

### image_processor.py

**Purpose**: Image enhancement and normalization

**Key Classes**:

- `ImageProcessor`: Main processing class
- `ProcessingResult`: Result with image and metadata

**Processing Steps**:

1. Grayscale conversion
2. Deskew (rotation correction)
3. Size normalization (Letter size @ 300 DPI)
4. CLAHE contrast enhancement
5. Bilateral noise removal
6. Spot cleaning (remove small artifacts)
7. Unsharp mask sharpening

### utils/output_generator.py

**Purpose**: Generate output files for each page

**Key Classes**:

- `PageOutput`: Paths to generated files
- `OutputGenerator`: Main generator class

**Outputs per page**:

- `Page_NNN/<PDF_STEM>_page_NNNN.json`: Structured transcript
- `Page_NNN/assets/*_raw.pdf`: Single page extracted from source
- `Page_NNN/assets/*_enhanced.png`: Processed grayscale image
- `Page_NNN/ocr/*_ocr_*.txt`: OCR artifacts (raw)
- No block overlay images are generated

### ocr_client.py

**Purpose**: Send images to LM Studio for OCR

**Key Classes**:

- `LMStudioOCRClient`: OpenAI-compatible API client
- `OCRError`: Base exception
- `OCRConnectionError`: Connection failures
- `OCRResponseError`: Invalid responses

**Features**:

- JPEG payload optimization (quality 95)
- OpenAI-compatible image_url format
- Configurable timeout (default: 120s)
- No AI post-processing stage; parsing uses heuristics and lexicon correction.

### ocr_parser.py

**Purpose**: Parse OCR text into structured blocks and apply intelligent
corrections

**Key Components**:

- `parse_ocr_text()`: Advanced iterative parser for layout recovery.
- `TextCorrector`: Lexicon-based spelling and context engine.
- `TimestampCorrector`: Timecode recovery and chronological validation.
- `SpeakerCorrector`: Standardizes callers based on mission roster.
- Location extraction for parenthesized station identifiers.

**Post-Processing Details**: See [POST_PROCESSING.md](./POST_PROCESSING.md)
for logic and formulas.

**Output JSON Structure**:

```json
{
  "header": {"page": 42, "tape": "1/2", "is_apollo_title": true},
  "blocks": [
    {"type": "comm", "timestamp": "00 00 00 00", "speaker": "CDR", "text": "..."},
    {"type": "continuation", "text": "..."},
    {"type": "annotation", "text": "..."},
    {"type": "meta", "text": "END OF TAPE"}
  ]
}
```

### pipeline.py

**Purpose**: Orchestrate complete processing

**Key Classes**:

- `PageResult`: Single page processing result
- `PipelineResult`: Full document result with statistics
- `TranscriptPipeline`: Main orchestrator

**Key Methods**:

- `process_page(page_num)`: Process single page
- `process_pages(page_numbers)`: Process specific pages
- `process_range(start, end)`: Process page range
- `process_all()`: Process entire document

**Features**:

- Sequential and parallel processing
- Progress callbacks
- Per-page error handling

## Data Flow

```text
PDF File
    │
    ▼ PageExtractor.extract_page_image()
numpy.ndarray (BGR, 300 DPI)
    │
    ▼ ImageProcessor.process()
ProcessingResult (grayscale, normalized)
    │
    ├──────────────────────────────────────┐
    │                                      │
    ▼ OutputGenerator.generate()           ▼ LMStudioOCRClient.ocr_image()
PageOutput (PNG, PDF files)                Raw OCR text
                                           ▼ correct_image_text() (optional)
                                           ▼ parse_ocr_text() + build_page_json()
                                           JSON file
```

## Threading Model

```text
Main Thread
    │
    ├── ThreadPoolExecutor (max_workers from config)
    │       │
    │       ├── Worker 1: process_page(0)
    │       ├── Worker 2: process_page(1)
    │       ├── Worker 3: process_page(2)
    │       └── Worker N: process_page(N)
    │
    └── tqdm progress bar (sequential OCR follows)
```

**Thread Safety**:

- PDF extraction: Each worker opens its own file handle
- OpenCV operations: Thread-safe
- Output directories: Created atomically with `exist_ok=True`
- OCR: Sequential (not parallelized)

## Error Handling

- **Per-page isolation**: Errors don't stop the pipeline
- **Logging**: Uses loguru at WARNING level (DEBUG with -v)
- **Result tracking**: `PageResult.success` and `.error` fields
- **Exit codes**: Non-zero if any pages fail
- **OCR errors**: Caught and written to JSON with error field

## Configuration Hierarchy

```text
PipelineConfig defaults (hardcoded in dataclass)
       │
       ▼
GlobalConfig (config/defaults.toml)
       │
       ▼
MissionConfig (config/missions.toml)
       │
       ▼
CLI arguments (--pages, --no-ocr, --ocr-url, -v)
```

## File Structure

```text
ocr_transcript_v2/
├── main.py                 # CLI entry point
├── config/
│   ├── defaults.toml       # Global defaults
│   └── missions.toml       # Mission configs (consolidated)
├── src/
│   ├── __init__.py
│   ├── config.py           # PipelineConfig
│   ├── global_config.py    # GlobalConfig loader
│   ├── mission_config.py   # MissionConfig loader
│   ├── page_extractor.py   # PDF → numpy
│   ├── image_processor.py  # Image enhancement
│   ├── output_generator.py # File generation
│   ├── ocr_client.py       # LM Studio client
│   ├── ocr_parser.py       # OCR text parsing
│   └── pipeline.py         # Orchestrator
├── docs/
│   ├── ARCHITECTURE.md     # This file
│   ├── PIPELINE.md         # Stage details
│   └── EXTENDING.md        # Extension guide
├── input/                  # Default PDF location
└── output/                 # Generated files
```
