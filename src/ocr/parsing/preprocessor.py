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

TIMESTAMP_ANY_SEP_RE = re.compile(
    r"(?<!\d)"
    r"((?=[0-9OIilCSB]*\d)[0-9OIilCSB]{1,2})"
    r"[^0-9OIilCSB]+"
    r"((?=[0-9OIilCSB]*\d)[0-9OIilCSB]{1,2})"
    r"[^0-9OIilCSB]+"
    r"((?=[0-9OIilCSB]*\d)[0-9OIilCSB]{1,2})"
    r"[^0-9OIilCSB]+"
    r"((?=[0-9OIilCSB]*\d)[0-9OIilCSB]{1,2})"
    r"(?!\d)"
)


def _normalize_ts_token(token: str) -> str:
    token = token.replace("O", "0").replace("o", "0")
    token = token.replace("I", "1").replace("i", "1").replace("l", "1")
    token = token.replace("C", "0").replace("c", "0")
    token = token.replace("S", "5").replace("s", "5")
    token = token.replace("B", "8").replace("b", "8")
    token = token.replace("°", "0")
    if len(token) == 1:
        token = f"0{token}"
    return token


def normalize_timestamp_noise(line: str) -> str:
    """
    Normalize timestamp-like chunks so the parser can recognize them.
    """
    def repl(match: re.Match) -> str:
        a = _normalize_ts_token(match.group(1))
        b = _normalize_ts_token(match.group(2))
        c = _normalize_ts_token(match.group(3))
        d = _normalize_ts_token(match.group(4))
        return f"{a} {b} {c} {d}"

    return TIMESTAMP_ANY_SEP_RE.sub(repl, line)


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
        line = normalize_timestamp_noise(line)
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
