"""
OCR Output Parser.

Parses plain text OCR output into structured blocks for NASA transcripts.
"""

import re

# Regex patterns for transcript parsing
TIMESTAMP_LINE_RE = re.compile(r"^\d{2}\s+\d{2}\s+\d{2}\s+\d{1,2}$")
TIMESTAMP_PREFIX_RE = re.compile(r"^(\d{2}\s+\d{2}\s+\d{2}\s+\d{1,2})\b")
SPEAKER_LINE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,6}(?:\s*\([A-Z0-9]+\))?$")
SPEAKER_PAREN_RE = re.compile(r"^\([A-Z0-9]+\)$")
HEADER_PAGE_RE = re.compile(r"\bPAGE\s*(\d{1,4})\b", re.IGNORECASE)
HEADER_TAPE_RE = re.compile(r"\bTAPE\s*([0-9]{1,2}\s*/\s*[0-9]{1,2})\b", re.IGNORECASE)
HEADER_KEYWORDS = ("GOSS", "NET", "TAPE", "PAGE", "APOLLO", "AIR-TO-GROUND")
HEADER_APOLLO_KEYS = ("APOLLO", "AIR", "GROUND", "VOICE", "TRANSCRIPTION")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", text).strip()


def parse_ocr_text(text: str, page_num: int) -> list[dict]:
    """
    Parse plain OCR output into structured rows.

    Args:
        text: Raw OCR text output
        page_num: 0-indexed page number

    Returns:
        List of row dictionaries with keys: page, line, type, timestamp, speaker, text
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # Find first timestamp to identify header zone
    first_ts_idx = next(
        (i for i, line in enumerate(lines)
         if TIMESTAMP_LINE_RE.match(line) or TIMESTAMP_PREFIX_RE.match(line)),
        None
    )

    rows = []
    line_index = 0
    pending_ts = ""
    pending_speaker = ""
    pending_text: list[str] = []

    def flush_pending():
        nonlocal pending_ts, pending_speaker, pending_text, line_index
        if not pending_ts and not pending_speaker and not pending_text:
            return
        line_index += 1
        rows.append({
            "page": page_num + 1,
            "line": line_index,
            "type": "comm" if pending_ts else "text",
            "timestamp": pending_ts,
            "speaker": pending_speaker,
            "text": " ".join(pending_text).strip(),
        })
        pending_ts = ""
        pending_speaker = ""
        pending_text = []

    for idx, line in enumerate(lines):
        upper = line.upper()

        # Check for header/footer/annotation
        is_header = (
            first_ts_idx is not None
            and idx <= first_ts_idx
            and not TIMESTAMP_PREFIX_RE.match(line)
            and any(kw in upper for kw in HEADER_KEYWORDS)
        )
        is_footer = "***" in line or "ASTERISK" in upper
        is_annotation = "(REV" in upper

        if is_header or is_footer or is_annotation:
            flush_pending()
            line_index += 1
            block_type = "header" if is_header else ("footer" if is_footer else "annotation")
            rows.append({
                "page": page_num + 1,
                "line": line_index,
                "type": block_type,
                "timestamp": "",
                "speaker": "",
                "text": line,
            })
            continue

        # Standalone timestamp line
        if TIMESTAMP_LINE_RE.match(line):
            flush_pending()
            pending_ts = line
            continue

        # Line starting with timestamp
        prefix_match = TIMESTAMP_PREFIX_RE.match(line)
        if prefix_match:
            flush_pending()
            pending_ts = prefix_match.group(1)
            remainder = line[len(pending_ts):].strip()
            if remainder:
                tokens = remainder.split()
                if tokens and SPEAKER_LINE_RE.match(tokens[0]):
                    pending_speaker = tokens.pop(0)
                    if tokens and SPEAKER_PAREN_RE.match(tokens[0]):
                        pending_speaker += " " + tokens.pop(0)
                if tokens:
                    pending_text.append(" ".join(tokens))
            continue

        # Continuation of a timestamped block
        if pending_ts:
            if SPEAKER_LINE_RE.match(line):
                pending_speaker = f"{pending_speaker} {line}".strip() if pending_speaker else line
                continue
            if SPEAKER_PAREN_RE.match(line):
                pending_speaker = f"{pending_speaker} {line}".strip() if pending_speaker else line
                continue
            pending_text.append(line)
            continue

        # Standalone text line
        line_index += 1
        rows.append({
            "page": page_num + 1,
            "line": line_index,
            "type": "text",
            "timestamp": "",
            "speaker": "",
            "text": line,
        })

    flush_pending()
    return rows


def extract_header_metadata(lines: list[str], page_num: int, page_offset: int = 0) -> dict:
    """
    Extract page metadata from header lines.

    Args:
        lines: Non-empty text lines from OCR
        page_num: 0-indexed page number
        page_offset: Page number offset from mission config

    Returns:
        Dictionary with keys: number, tape, apollo
    """
    result = {"number": page_num + 1 + page_offset, "tape": "", "apollo": ""}

    # Find header zone (before first timestamp)
    first_ts_idx = next(
        (i for i, line in enumerate(lines)
         if TIMESTAMP_LINE_RE.match(line) or TIMESTAMP_PREFIX_RE.match(line)),
        None
    )
    header_lines = lines[:first_ts_idx + 1] if first_ts_idx is not None else lines[:10]

    apollo_parts = []
    for line in header_lines:
        if TIMESTAMP_PREFIX_RE.match(line):
            continue
        normalized = normalize_whitespace(line)
        if not normalized:
            continue

        # Extract page number
        if match := HEADER_PAGE_RE.search(normalized):
            result["number"] = int(match.group(1))

        # Extract tape number
        if match := HEADER_TAPE_RE.search(normalized):
            result["tape"] = match.group(1).replace(" ", "")

        # Collect Apollo header parts
        if any(key in normalized.upper() for key in HEADER_APOLLO_KEYS):
            apollo_parts.append(normalized)

    if apollo_parts:
        apollo_text = normalize_whitespace(" ".join(apollo_parts))
        if "APOLLO" in apollo_text.upper():
            result["apollo"] = apollo_text

    return result


def build_page_json(rows: list[dict], lines: list[str], page_num: int, page_offset: int = 0) -> dict:
    """
    Build structured JSON output for a page.

    Args:
        rows: Parsed rows from parse_ocr_text
        lines: Raw non-empty lines
        page_num: 0-indexed page number
        page_offset: Page number offset

    Returns:
        Dictionary with page info and blocks
    """
    page_info = extract_header_metadata(lines, page_num, page_offset)
    blocks = []

    for row in rows:
        if row["type"] == "header":
            # Update page info from header rows
            normalized = normalize_whitespace(row["text"])
            if match := HEADER_PAGE_RE.search(normalized):
                page_info["number"] = int(match.group(1))
            if match := HEADER_TAPE_RE.search(normalized):
                page_info["tape"] = match.group(1).replace(" ", "")
            upper = normalized.upper()
            if "APOLLO" in upper and "AIR" in upper:
                page_info["apollo"] = normalized
            continue

        block_type = "continuation" if row["type"] == "text" else row["type"]
        block = {"type": block_type}

        if block_type == "comm":
            if row["timestamp"]:
                block["timestamp"] = row["timestamp"]
            if row["speaker"]:
                block["speaker"] = row["speaker"]

        if row["text"]:
            block["text"] = row["text"]

        blocks.append(block)

    return {"page": page_info, "blocks": blocks}
