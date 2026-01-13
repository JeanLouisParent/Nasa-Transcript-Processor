"""
OCR Output Parser.

Parses plain text OCR output into structured blocks for NASA transcripts.
"""

import difflib
import re
from pathlib import Path

from .speaker_corrector import SpeakerCorrector
from .text_corrector import TextCorrector
from .timestamp_corrector import TimestampCorrector

# Regex patterns for transcript parsing
TIMESTAMP_LINE_RE = re.compile(r"^\d{2}\s+\d{2}\s+\d{2}\s+\d{1,2}$")
TIMESTAMP_PREFIX_RE = re.compile(r"^(\d{2}\s+\d{2}\s+\d{2}\s+\d{1,2})\b")
SPEAKER_LINE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,6}(?:\s*\([A-Z0-9]+\))?$")
SPEAKER_PAREN_RE = re.compile(r"^\([A-Z0-9]+\)$")
HEADER_PAGE_RE = re.compile(r"\bPAGE\s*(\d{1,4})\b", re.IGNORECASE)
HEADER_TAPE_RE = re.compile(r"\bTAPE\s*([0-9]{1,2}\s*/\s*[0-9]{1,2})\b", re.IGNORECASE)
HEADER_KEYWORDS = ("GOSS", "NET", "TAPE", "PAGE", "APOLLO", "AIR-TO-GROUND")
HEADER_APOLLO_KEYS = ("APOLLO", "AIR", "GROUND", "VOICE", "TRANSCRIPTION")
END_OF_TAPE_KEYWORD = "END OF TAPE"


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
            and any(fuzzy_find(line, kw) for kw in HEADER_KEYWORDS)
        )
        is_footer = "***" in line or "ASTERISK" in upper
        is_annotation = "(REV" in upper
        is_end_of_tape = fuzzy_find(line, END_OF_TAPE_KEYWORD)

        if is_header or is_footer or is_annotation or is_end_of_tape:
            flush_pending()
            line_index += 1
            if is_end_of_tape:
                block_type = "meta"
            elif is_header:
                block_type = "header"
            elif is_footer:
                block_type = "footer"
            else:
                block_type = "annotation"
            
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


def fuzzy_find(text: str, keyword: str, threshold: float = 0.6) -> bool:
    """Check if keyword is fuzzily present in text."""
    if not text or not keyword:
        return False
    text_upper = text.upper()
    keyword_upper = keyword.upper()
    
    if keyword_upper in text_upper:
        return True
        
    # Check words for similarity
    words = text_upper.split()
    for word in words:
        if difflib.SequenceMatcher(None, word, keyword_upper).ratio() >= threshold:
            return True
    return False


def extract_header_metadata(lines: list[str], page_num: int, page_offset: int = 0) -> dict:
    """
    Extract page metadata from header lines with fuzzy matching.

    Args:
        lines: Non-empty text lines from OCR
        page_num: 0-indexed page number
        page_offset: Page number offset from mission config

    Returns:
        Dictionary with keys: page, tape, is_apollo_title
    """
    result = {
        "page": page_num + 1 + page_offset, # Default fallback
        "tape": None,
        "is_apollo_title": False
    }

    # Find header zone (before first timestamp)
    first_ts_idx = next(
        (i for i, line in enumerate(lines) 
         if TIMESTAMP_LINE_RE.match(line) or TIMESTAMP_PREFIX_RE.match(line)),
        None
    )
    # Limit search to first 10 lines if no timestamp found
    search_limit = first_ts_idx if first_ts_idx is not None else min(len(lines), 10)
    header_lines = lines[:search_limit]

    # Specific extractors
    for line in header_lines:
        normalized = normalize_whitespace(line)
        upper = normalized.upper()

        # 1. Page Number (PAGE XXX)
        if match := HEADER_PAGE_RE.search(normalized):
            try:
                result["page"] = int(match.group(1))
            except ValueError:
                pass
        elif fuzzy_find(upper, "PAGE"):
            digits = re.findall(r"\d+", normalized)
            if digits:
                result["page"] = int(digits[-1])

        # 2. Tape Number (TAPE XX/XX)
        if match := HEADER_TAPE_RE.search(normalized):
            result["tape"] = match.group(1).replace(" ", "")
        elif fuzzy_find(upper, "TAPE"):
            tape_match = re.search(r"(\d+[\s/-]+\d+)", normalized)
            if tape_match:
                result["tape"] = tape_match.group(1).replace(" ", "")

        # 3. Apollo Title (Boolean)
        if fuzzy_find(upper, "APOLLO") and fuzzy_find(upper, "TRANSCRIPTION"):
            result["is_apollo_title"] = True

    return result


def clean_trailing_footer(text: str) -> str:
    """Remove trailing footer info (tape/page) from text."""
    # Pattern for "tape XX/XX Page XX" at end of string
    pattern = re.compile(r"\s+tape\s+[\d/]+\s+Page\s+\d+\s*$", re.IGNORECASE)
    return pattern.sub("", text)


def build_page_json(
    rows: list[dict], 
    lines: list[str], 
    page_num: int, 
    page_offset: int = 0,
    valid_speakers: list[str] = None,
    text_replacements: dict[str, str] = None
) -> dict:
    """
    Build structured JSON output for a page.

    Args:
        rows: Parsed rows from parse_ocr_text
        lines: Raw non-empty lines
        page_num: 0-indexed page number
        page_offset: Page number offset
        valid_speakers: Optional list of valid speaker codes for correction
        text_replacements: Optional dictionary of regex replacements for text

    Returns:
        Dictionary with header info and blocks
    """
    header_info = extract_header_metadata(lines, page_num, page_offset)
    blocks = []

    for row in rows:
        # Skip header rows in the main block list, as they are aggregated in header_info
        if row["type"] == "header":
            continue

        block_type = "continuation" if row["type"] == "text" else row["type"]
        block = {"type": block_type}

        if block_type == "comm":
            if row["timestamp"]:
                block["timestamp"] = row["timestamp"]
            if row["speaker"]:
                block["speaker"] = row["speaker"]

        if row["text"]:
            block["text"] = clean_trailing_footer(row["text"])

        blocks.append(block)

    # Post-process timestamps
    ts_corrector = TimestampCorrector()
    blocks = ts_corrector.process_blocks(blocks)

    # Post-process speakers
    if valid_speakers:
        sp_corrector = SpeakerCorrector(valid_speakers)
        blocks = sp_corrector.process_blocks(blocks)

    # Post-process text (spelling and noise)
    # TODO: Make lexicon path configurable via mission config if needed
    lexicon_path = Path("assets/lexicon/apollo11_lexicon.json")
    if lexicon_path.exists():
        txt_corrector = TextCorrector(lexicon_path, text_replacements)
        blocks = txt_corrector.process_blocks(blocks)

    return {"header": header_info, "blocks": blocks}
