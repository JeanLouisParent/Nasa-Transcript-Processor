"""
State machine for parsing OCR text into structured rows.
This module now acts as a facade for the TranscriptParser class.
"""

from .parser import TranscriptParser

def parse_ocr_text(
    text: str,
    page_num: int,
    mission_keywords: list[str] | None = None,
    valid_speakers: list[str] | None = None
) -> tuple[list[dict], bool]:
    """
    Parse plain OCR output into structured rows using TranscriptParser.

    Returns:
        (rows, has_footer): List of parsed blocks and footer presence flag
    """
    parser = TranscriptParser(page_num, mission_keywords, valid_speakers)
    return parser.parse(text)
