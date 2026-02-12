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


def is_goss_net_noise(text: str) -> bool:
    """
    Detect OCR variants of "(GOSS NET 1)" so they can be filtered reliably.
    Examples: "GOSS NET 1", "GO.05 NOT 1)", "G0SS N3T 1",
    "(60.000 NI'1 1)", "(000, NI\" 1)", "(00:5 HOT 1)".
    """
    if not text:
        return False
    raw = text.strip().upper()
    if len(raw) > 40:
        return False

    # Pattern 1: Heavily corrupted variants with lots of digits/noise
    # Look for "NET/NOT/NI/HOT" followed by "1"
    if re.search(r"\b(NET|NOT|NI|HOT|N[IE][T1])\s*['\"]?\s*1\b", raw):
        # Also has lots of zeros/digits (characteristic of corrupted GOSS NET)
        if raw.count("0") + raw.count("O") >= 2:
            return True

    # Pattern 2: Just a fragment like "(00" or "(000" with nothing else
    # But NOT timestamps like "(00:08:00)"
    if re.match(r"^\(?\d{2,4}\)?$", text.strip()) and ":" not in text:
        return True

    norm = re.sub(r"[^A-Z0-9]", "", raw)
    if not norm:
        return False

    # Pattern 3: Original fuzzy matching logic
    mapped = (
        norm
        .replace("0", "O")
        .replace("5", "S")
        .replace("3", "E")
        .replace("7", "T")
        .replace("I", "1")
        .replace("L", "1")
    )
    target = "GOSSNET1"
    if target in mapped:
        return True
    return difflib.SequenceMatcher(None, mapped, target).ratio() >= 0.72


def is_not1_footer_noise(text: str) -> bool:
    """
    Detect OCR garbage variants of footer/header fragments like "(... NOT 1)".
    """
    if not text:
        return False
    raw = text.strip().upper()
    if len(raw) > 36:
        return False
    if "NOT" not in raw or "1" not in raw:
        return False
    # Restrict to short, mostly punctuation/uppercase/digits snippets.
    if re.match(r"^\(?\s*[A-Z0-9\.\:\-_%\"' ]{0,20}\s*NOT\s*1\)?\s*$", raw):
        return True
    return False


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


def clean_leading_footer_noise(text: str) -> str:
    """Remove OCR garbage prefixes like '(00% NOT 1)' from line starts."""
    if not text:
        return text
    # Strip leading smart/curly quotes (OCR artifacts)
    cleaned = re.sub(r"^[\u2018\u2019\u201B\u2032\u0027\u0060\u00B4]+", "", text.strip())
    cleaned = re.sub(
        r"^\(?\s*[A-Z0-9\.\:\-_%\"']{0,12}\s*NOT\s*1\)\s*",
        "",
        cleaned.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def is_garbled_page_tape_label(text: str) -> bool:
    """
    Detect garbled Page/Tape labels in continuations.
    OCR misreads like "Porch 107" (Page), "Type 65/H" (Tape), "Ptc 18/3" (Tape),
    "Low 111" (Page), "Lack 48" (Tape), "Up 83/2" (Tape), "Pnrc 107" (Porch),
    or with noise: "(00% NOT 1) \"Type 10/1 page 360"
    """
    if not text or len(text) > 60:
        return False

    # Pattern 1: Short text (< 15 chars) matching corrupted Page/Tape patterns
    # Catches OCR corruptions before spell-check: "Pnrc 107", "Lnck 48", "Typo 65", etc.
    # Starts with P (Page), T/L (Tape), or similar corruption patterns
    if len(text.strip()) < 15:
        if re.match(r"^[PTL][a-z]{2,4}\s+\d+[\d/]*$", text.strip(), re.IGNORECASE):
            return True
        # Also catch 2-letter prefixes: "Up 83", "Tc 18"
        if re.match(r"^[PTLU][a-z]\s+\d+[\d/]*$", text.strip(), re.IGNORECASE):
            return True

    # Pattern 2: Known Page/Tape keywords (after spell-check)
    if re.search(
        r"\b(?:Porch|Page|Type|Tape|Ptc|Low|Lack|Up|Pyro|P\s*\dc)\s+[\d/]+",
        text,
        re.IGNORECASE,
    ):
        return True

    # Pattern 3: "page" or "tape" with numbers
    if re.search(r"\b(?:page|tape)\s+\d+", text, re.IGNORECASE):
        return True

    return False


def extract_header_metadata(lines: list[str], page_num: int, page_offset: int = 0) -> dict:
    """Extract page/tape metadata from header lines."""
    from .patterns import TIMESTAMP_STRICT_RE, TIMESTAMP_PREFIX_RE, HEADER_TAPE_RE

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

        # Extract tape number from OCR if present (e.g., "Tape 1/8")
        tape_match = HEADER_TAPE_RE.search(line)
        if tape_match:
            # Normalize the tape format (remove extra spaces)
            tape_str = tape_match.group(1).replace(" ", "")
            result["tape"] = tape_str

    return result
