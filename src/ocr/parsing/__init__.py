"""
OCR parsing submodule.
"""

from .state_machine import parse_ocr_text
from .block_builder import build_page_json, PageBuilderConfig

__all__ = ["parse_ocr_text", "build_page_json", "PageBuilderConfig"]
