"""
Text preprocessing for OCR parsing.
"""

import re
from .patterns import (
    LINE_TAG_RE,
    TIMESTAMP_EMBEDDED_RE,
    TIMESTAMP_PREFIX_RE,
    REV_EMBEDDED_RE,
    SPEAKER_TOKEN_RE,
)


def should_split_embedded_timestamp(line: str, match: re.Match) -> bool:
    """
    Split only when the embedded timestamp is followed by a plausible speaker token.
    """
    remainder = line[match.end() :].lstrip()
    if not remainder:
        return False
    token = remainder.split()[0]
    if SPEAKER_TOKEN_RE.match(token):
        return True
    if remainder.startswith("("):
        close_idx = remainder.find(")")
        if close_idx != -1:
            after = remainder[close_idx + 1 :].lstrip()
            if after:
                token = after.split()[0]
                if SPEAKER_TOKEN_RE.match(token):
                    return True
    return False


def preprocess_lines(text: str, mission_keywords: list[str] | None = None) -> list[dict]:
    """
    Preprocess OCR text into cleaned lines with forced types.
    Returns list of dicts with 'text' and 'forced' keys.
    """
    raw_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Normalize fractional seconds like "03 10 47 .1" -> "03 10 47 11"
        line = re.sub(r"\b(\d{2}\s+\d{2}\s+\d{2})\s+\.(\d)\b", r"\1 1\2", line)
        raw_lines.append(line)

    lines = []
    for raw in raw_lines:
        tag_match = LINE_TAG_RE.match(raw)
        forced_type = None
        if tag_match:
            forced_type = tag_match.group(1).lower()
            raw = raw[tag_match.end() :].strip()
        if raw:
            lines.append({"text": raw, "forced": forced_type})

    # Iteratively split embedded components
    current_processing = lines
    final_lines = []

    while current_processing:
        entry = current_processing.pop(0)
        line = entry["text"]
        forced_type = entry["forced"]

        # If line starts with a timestamp, search for splits after the initial timestamp zone
        search_start = 0
        if TIMESTAMP_PREFIX_RE.match(line):
            search_start = 12

        # Check for embedded timestamp
        ts_match = TIMESTAMP_EMBEDDED_RE.search(line, search_start)
        if ts_match and not should_split_embedded_timestamp(line, ts_match):
            ts_match = None

        # Check for embedded REV marker
        rev_match = REV_EMBEDDED_RE.search(line, search_start)

        # Check for trailing mission keyword
        kw_match = None
        if mission_keywords:
            for kw in mission_keywords:
                if len(kw) > 3:
                    kw_re = re.compile(rf"\s+({re.escape(kw)})", re.IGNORECASE)
                    m = kw_re.search(line, search_start)
                    if m:
                        if not kw_match or m.start() < kw_match.start():
                            kw_match = m

        matches = [m for m in [ts_match, rev_match, kw_match] if m]
        if matches:
            match = min(matches, key=lambda x: x.start())
            start = match.start()
            part1 = line[:start].strip()
            part2 = line[start:].strip()
            if part1:
                final_lines.append({"text": part1, "forced": forced_type})
            if part2:
                current_processing.insert(0, {"text": part2, "forced": forced_type})
        else:
            final_lines.append({"text": line, "forced": forced_type})

    return final_lines
