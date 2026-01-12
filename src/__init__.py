"""
NASA Transcript Image Processing Pipeline

Industrial-grade pipeline for processing scanned NASA mission transcripts.
Performs page-by-page image enhancement and geometric layout detection,
with optional OCR via LM Studio.

Modules:
    config: Pipeline configuration
    page_extractor: PDF to image extraction
    image_processor: Deskew, enhancement, normalization
    layout_detector: Geometric block detection
    output_generator: Output file generation
    pipeline: Main orchestrator
    ocr_client: LM Studio OCR client
    ocr_parser: OCR output parsing
"""

__version__ = "1.0.0"
__author__ = "NASA Transcript Processing Team"

from .config import PipelineConfig
from .pipeline import TranscriptPipeline

__all__ = [
    "PipelineConfig",
    "TranscriptPipeline",
    "__version__",
]
