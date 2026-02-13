"""
Helper utilities for merge operations.
Contains timestamp, text, and speaker extraction functions.
"""

from __future__ import annotations
import re


def parse_timestamp(ts: str) -> tuple[int, int, int, int] | None:
    """Parse timestamp string into (day, hour, minute, second) tuple."""
    parts = ts.split()
    if len(parts) != 4:
        return None
    try:
        return tuple(int(p) for p in parts)
    except (ValueError, TypeError):
        return None


def format_timestamp(ts: tuple[int, int, int, int]) -> str:
    """Format timestamp tuple as 'DD HH MM SS' string."""
    return f"{ts[0]:02d} {ts[1]:02d} {ts[2]:02d} {ts[3]:02d}"


def bump_timestamp(ts: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Increment timestamp by 1 second.
    Apollo format: (dd, hh, mm, ss) = (day, hour, minute, second)
    """
    dd, hh, mm, ss = ts
    ss += 1
    if ss >= 60:
        ss = 0
        mm += 1
        if mm >= 60:
            mm = 0
            hh += 1
            if hh >= 24:
                hh = 0
                dd += 1
    return dd, hh, mm, ss


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_missing_speaker(value: str | None) -> bool:
    """Check if speaker value is missing or invalid."""
    return not value or not value.strip()


def extract_speaker_from_text(text: str, valid_speakers: list[str]) -> tuple[str | None, str]:
    """
    Extracts a speaker callsign if it appears at the immediate start of dialogue text.

    Args:
        text: The dialogue string to evaluate.
        valid_speakers: List of allowed callsigns.

    Returns:
        Tuple of (extracted_speaker_code, cleaned_dialogue_text).
    """
    if not text:
        return None, text

    for speaker in valid_speakers:
        # Check if text starts with "SPEAKER " or "SPEAKER:"
        if text.startswith(speaker + ' ') or text.startswith(speaker + ':'):
            # Extract speaker and clean text
            prefix_len = len(speaker) + 1
            cleaned_text = text[prefix_len:].strip()
            return speaker, cleaned_text

    return None, text


def starts_continuation(text: str) -> bool:
    """Check if text starts like a continuation (lowercase or ellipsis)."""
    return bool(text and (text[0].islower() or text.startswith("...")))


def looks_truncated(text: str) -> bool:
    """
    Detect likely mid-word truncation at the end of a line.
    Returns True if the text seems cut off mid-sentence.
    """
    if not text:
        return False

    text = text.rstrip()
    if len(text) < 3:
        return False

    # Ends with hyphen or lowercase letter (no punctuation)
    if text.endswith("-"):
        return True

    # Last char is lowercase and no sentence-ending punctuation
    if text[-1].islower():
        # Allow certain exceptions (abbreviations, etc.)
        last_word = text.split()[-1] if text.split() else ""
        # If last word is very short (like "a", "i"), it might be ok
        if len(last_word) <= 2:
            return False
        return True

    return False


def apply_text_replacements(text: str, replacements: dict[str, str]) -> str:
    """Apply a dictionary of text replacements."""
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def strip_end_of_tape_residue(text: str) -> str:
    """Remove END OF TAPE markers and surrounding noise."""
    if not text:
        return text
    text = re.sub(r"\*\*\s*END\s+OF\s+TAPE\s*\*\*", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_footer_text(text: str) -> str:
    """
    Clean footer-related noise from text.
    Removes common footer patterns like page numbers, tape markers, etc.
    """
    if not text:
        return text

    # Remove patterns like "Page 123", "Tape 45/6"
    text = re.sub(r"\b(PAGE|TAPE)\s+\d+(/\d+)?", "", text, flags=re.IGNORECASE)

    # Remove patterns like "123/456" (page/total)
    text = re.sub(r"\b\d{1,3}/\d{1,3}\b", "", text)

    # Remove standalone numbers at start/end that look like page numbers
    text = re.sub(r"^\d{1,4}\s+", "", text)
    text = re.sub(r"\s+\d{1,4}$", "", text)

    return text.strip()
