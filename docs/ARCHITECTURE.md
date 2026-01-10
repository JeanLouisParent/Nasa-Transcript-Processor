# Architecture Documentation

This document describes the architecture of the NASA Transcript Processing Pipeline. It is designed for both human developers and AI code agents.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                            │
│                     Click-based interface                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     pipeline.py (Orchestrator)                  │
│            Coordinates all processing, parallel execution       │
└─────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   page_     │ │   image_    │ │   layout_   │ │   block_    │
│ extractor   │ │ processor   │ │  detector   │ │ classifier  │
│    .py      │ │    .py      │ │    .py      │ │    .py      │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────┐
                                              │   output_   │
                                              │ generator   │
                                              │    .py      │
                                              └─────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────┐
                                              │  config.py  │
                                              │ (shared)    │
                                              └─────────────┘
```

## Module Responsibilities

### config.py
**Purpose**: Centralized configuration management

**Key Classes**:
- `PipelineConfig`: Dataclass with all configurable parameters

**Extension Points**:
- Add new parameters to `PipelineConfig`
- Use `from_yaml()` for mission-specific configs

### page_extractor.py
**Purpose**: Extract pages from PDF without loading entire document

**Key Classes**:
- `PageExtractor`: Main extraction class

**Key Methods**:
- `extract_page_image(page_num)`: Get page as numpy array
- `extract_page_pdf(page_num, path)`: Extract single page PDF
- `iter_pages(start, end)`: Lazy page iterator

**Thread Safety**: Yes (uses mutex for PDF access)

### image_processor.py
**Purpose**: Image enhancement and normalization

**Key Classes**:
- `ImageProcessor`: Main processing class
- `ProcessingResult`: Result dataclass

**Pipeline Steps**:
1. Grayscale conversion
2. Deskew (rotation correction)
3. Size normalization
4. CLAHE contrast enhancement
5. Bilateral noise removal
6. Morphological spot cleaning
7. Unsharp mask sharpening

**Extension Points**:
- Add processing steps in `process()` method
- Override individual step methods

### layout_detector.py
**Purpose**: Detect text blocks geometrically

**Key Classes**:
- `Block`: Bounding box with metadata
- `LayoutResult`: Detection result
- `LayoutDetector`: Main detector

**Algorithm**:
1. Binarize + horizontal dilation (connect characters)
2. Line region detection + row clustering (merge overlaps)
3. Column boundary detection from ink projection
4. Header/footer/annotation/COMM classification (geometric heuristics)
5. COMM grouping with continuation rows

**Extension Points**:
- Adjust kernel sizes for different fonts
- Add column detection heuristics

### block_classifier.py
**Purpose**: Legacy classifier (not used by the pipeline)

**Key Classes**:
- `BlockType`: Enum of block types
- `ClassifiedBlock`: Block with classification
- `BlockClassifier`: Main classifier

**Notes**:
- The current pipeline uses `layout_detector.py` for block typing.
- This module is kept for experimentation and may be removed later.

**Extension Points**:
- Add new `BlockType` values
- Modify classification heuristics

### output_generator.py
**Purpose**: Generate output files

**Key Classes**:
- `PageOutput`: Output file paths
- `OutputGenerator`: Main generator

**Outputs**:
- `<PDF>_page_XXXX_raw.pdf`: Single page PDF
- `<PDF>_page_XXXX_enhanced.png`: Processed image
- `<PDF>_page_XXXX_blocks.png`: Image with block overlays

**Extension Points**:
- Add new output formats
- Modify blocks visualization

### pipeline.py
**Purpose**: Orchestrate complete processing

**Key Classes**:
- `PageResult`: Single page result
- `PipelineResult`: Full document result
- `TranscriptPipeline`: Main orchestrator

**Features**:
- Sequential and parallel processing
- Progress callbacks
- Error handling per page

## Data Flow

```
PDF File
    │
    ▼ PageExtractor.extract_page_image()
numpy.ndarray (BGR, 300 DPI)
    │
    ▼ ImageProcessor.process()
ProcessingResult (grayscale, normalized)
    │
    ▼ LayoutDetector.detect()
LayoutResult (list of Block)
    │
    ▼ BlockClassifier.classify()
ClassificationResult (list of ClassifiedBlock)
    │
    ▼ OutputGenerator.generate()
PageOutput (file paths)
```

## Threading Model

```
Main Thread
    │
    ├── ThreadPoolExecutor (max_workers=4)
    │       │
    │       ├── Worker 1: process_page(0)
    │       ├── Worker 2: process_page(1)
    │       ├── Worker 3: process_page(2)
    │       └── Worker 4: process_page(3)
    │
    └── tqdm progress bar
```

**Thread Safety**:
- PDF extraction uses mutex (`_extract_lock`)
- OpenCV operations are thread-safe
- Output directories created atomically

## Error Handling

Each page is processed independently:
- Errors are caught and logged
- Failed pages don't stop pipeline
- `PageResult.success` and `.error` track failures

## Configuration Hierarchy

```
DEFAULT_CONFIG (hardcoded defaults)
       │
       ▼
YAML file (via `PipelineConfig.from_yaml` in custom scripts) [optional]
       │
       ▼
CLI arguments (--dpi, --workers, etc.)
```
