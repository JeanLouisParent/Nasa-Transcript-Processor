"""
OCR Output Parser.

This module provides the public API for parsing OCR text into structured blocks.
The implementation is split across multiple modules in the parsing subpackage.
"""

# Re-export public API
from .parsing import parse_ocr_text, build_page_json

__all__ = ["parse_ocr_text", "build_page_json"]
