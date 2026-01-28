"""
Utility functions for OCR parsing.
"""

import difflib
import re


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_find(text: str, keyword: str, threshold: float = 0.6) -> bool:
    """
    Check if keyword appears in text with fuzzy matching.
    """
    if not text or not keyword:
        return False
    text_upper = text.upper()
    keyword_upper = keyword.upper()
    if len(keyword_upper) <= 3:
        return bool(re.search(rf"\b{re.escape(keyword_upper)}\b", text_upper))
    if keyword_upper in text_upper:
        return True
    words = text_upper.split()
    for word in words:
        if difflib.SequenceMatcher(None, word, keyword_upper).ratio() >= threshold:
            return True
    return False


def is_transcription_header(text: str) -> bool:
    """Check if text is the standard transcription header."""
    if not text:
        return False
    text = text.strip()
    if len(text) > 90:
        return False
    upper = text.upper().replace("0", "O")
    if "VOICE" not in upper or "TRANS" not in upper:
        return False
    norm_text = re.sub(r"[^A-Z]", "", upper)
    target = "AIRTOGROUNDVOICETRANSCRIPTION"
    return difflib.SequenceMatcher(None, norm_text, target).ratio() >= 0.6


def clean_trailing_footer(text: str) -> str:
    """Remove trailing tape/page footers from text."""
    # Double pattern: "Tape XX Page YY" or "Page YY Tape XX"
    text = re.sub(
        r"[\s\.\n\r]+(?:T[A-Z0-9]{2,})\s*[\d/IX]+\s+(?:P[A-Z0-9]{2,})\s*[\d/IX]+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[\s\.\n\r]+(?:P[A-Z0-9]{2,})\s*[\d/IX]+\s+(?:T[A-Z0-9]{2,})\s*[\d/IX]+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Single pattern: "Tape XX" or "Page YY" at the end
    text = re.sub(
        r"[\s\.\n\r]+(?:TAPS|TAPC|TAPE|TYPE|TANE|PAGE|PAGS|PACE|PAXE|PAGO|Paze|Page|Pags)\s*[\d/IX]+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def extract_header_metadata(lines: list[str], page_num: int, page_offset: int = 0) -> dict:
    """Extract page/tape metadata from header lines."""
    from .patterns import TIMESTAMP_STRICT_RE, TIMESTAMP_PREFIX_RE

    result = {"page": page_num + 1 + page_offset, "tape": None, "is_apollo_title": False}
    first_ts_idx = next(
        (i for i, ln in enumerate(lines) if TIMESTAMP_STRICT_RE.match(ln) or TIMESTAMP_PREFIX_RE.match(ln)),
        None,
    )
    header_lines = lines[:first_ts_idx] if first_ts_idx is not None else lines[:10]
    for line in header_lines:
        norm = normalize_whitespace(line).upper()
        if fuzzy_find(norm, "APOLLO") and fuzzy_find(norm, "TRANSCRIPTION"):
            result["is_apollo_title"] = True
    return result
