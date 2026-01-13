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
TS_CHARS = r"[\dOI'I\)\(\]\[]"
TIMESTAMP_STRICT_RE = re.compile(rf"^{TS_CHARS}{{2}}\s+{TS_CHARS}{{2}}\s+{TS_CHARS}{{2}}\s+{TS_CHARS}{{1,2}}$")
TIMESTAMP_PREFIX_RE = re.compile(rf"^({TS_CHARS}{{2}}\s+{TS_CHARS}{{2}}\s+{TS_CHARS}{{2}}\s+{TS_CHARS}{{1,2}})\b")
TIMESTAMP_EMBEDDED_RE = re.compile(rf"\s+({TS_CHARS}{{2}}\s+{TS_CHARS}{{2}}\s+{TS_CHARS}{{2}}\s+{TS_CHARS}{{1,2}})\b")
REV_EMBEDDED_RE = re.compile(r"(\([RE][FV]\s+\d+\))", re.IGNORECASE)

SPEAKER_LINE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,6}(?:\s*\([A-Z0-9]+\))?$")
SPEAKER_PAREN_RE = re.compile(r"^\([A-Z0-9]+\)$")
HEADER_PAGE_RE = re.compile(r"\bPAGE\s*(\d{1,4})\b", re.IGNORECASE)
HEADER_TAPE_RE = re.compile(r"\bTAPE\s*([0-9]{1,2}\s*/\s*[0-9]{1,2})\b", re.IGNORECASE)
HEADER_KEYWORDS = ("GOSS", "NET", "TAPE", "PAGE", "APOLLO", "AIR-TO-GROUND")
END_OF_TAPE_KEYWORD = "END OF TAPE"


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", text).strip()


def parse_ocr_text(text: str, page_num: int, mission_keywords: list[str] = None) -> list[dict]:
    """
    Parse plain OCR output into structured rows.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Pre-process lines to split embedded timestamps or REV markers
    split_lines = []
    for line in lines:
        match = None
        ts_match = TIMESTAMP_EMBEDDED_RE.search(line)
        rev_match = REV_EMBEDDED_RE.search(line)
        
        if ts_match and rev_match:
            match = ts_match if ts_match.start() < rev_match.start() else rev_match
        else:
            match = ts_match or rev_match

        if match:
            start = match.start()
            part1 = line[:start].strip()
            part2 = line[start:].strip()
            if part1: split_lines.append(part1)
            if part2: split_lines.append(part2)
        else:
            split_lines.append(line)
    
    lines = split_lines
    if not lines: return []

    first_ts_idx = next((i for i, l in enumerate(lines) if TIMESTAMP_STRICT_RE.match(l) or TIMESTAMP_PREFIX_RE.match(l)), None)

    rows = []
    line_index = 0
    pending_ts = ""
    pending_speaker = ""
    pending_text = []

    def flush_pending():
        nonlocal pending_ts, pending_speaker, pending_text, line_index
        if not pending_ts and not pending_speaker and not pending_text: return
        line_index += 1
        rows.append({
            "page": page_num + 1, "line": line_index,
            "type": "comm" if pending_ts else "text",
            "timestamp": pending_ts, "speaker": pending_speaker,
            "text": " ".join(pending_text).strip(),
        })
        pending_ts = ""; pending_speaker = ""; pending_text = []

    for idx, line in enumerate(lines):
        upper = line.upper()

        # Standalone timestamp
        if TIMESTAMP_STRICT_RE.match(line):
            flush_pending()
            pending_ts = line
            continue

        # Prefix timestamp
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
                if tokens: pending_text.append(" ".join(tokens))
            continue

        # Annotations
        is_header = not pending_ts and first_ts_idx is not None and idx <= first_ts_idx and any(fuzzy_find(line, kw) for kw in HEADER_KEYWORDS)
        is_footer = not pending_ts and ("***" in line or "ASTERISK" in upper)
        is_annotation = "(REV" in upper or "(RFV" in upper or (mission_keywords and not pending_ts and any(fuzzy_find(line, kw) for kw in mission_keywords))
        is_end_of_tape = fuzzy_find(line, END_OF_TAPE_KEYWORD)

        if is_header or is_footer or is_annotation or is_end_of_tape:
            flush_pending()
            line_index += 1
            rows.append({
                "page": page_num + 1, "line": line_index,
                "type": "meta" if is_end_of_tape else ("header" if is_header else ("footer" if is_footer else "annotation")),
                "timestamp": "", "speaker": "", "text": line,
            })
            continue

        if pending_ts:
            if SPEAKER_LINE_RE.match(line) or SPEAKER_PAREN_RE.match(line):
                pending_speaker = f"{pending_speaker} {line}".strip() if pending_speaker else line
                continue
            pending_text.append(line)
            continue

        pending_text.append(line)

    flush_pending()
    return rows


def fuzzy_find(text: str, keyword: str, threshold: float = 0.6) -> bool:
    if not text or not keyword: return False
    text_upper = text.upper(); keyword_upper = keyword.upper()
    if len(keyword_upper) <= 3:
        return bool(re.search(rf"\b{re.escape(keyword_upper)}\b", text_upper))
    if keyword_upper in text_upper: return True
    words = text_upper.split()
    for word in words:
        if difflib.SequenceMatcher(None, word, keyword_upper).ratio() >= threshold: return True
    return False


def extract_header_metadata(lines: list[str], page_num: int, page_offset: int = 0) -> dict:
    result = {"page": page_num + 1 + page_offset, "tape": None, "is_apollo_title": False}
    first_ts_idx = next((i for i, l in enumerate(lines) if TIMESTAMP_STRICT_RE.match(l) or TIMESTAMP_PREFIX_RE.match(l)), None)
    header_lines = lines[:first_ts_idx] if first_ts_idx is not None else lines[:10]
    for line in header_lines:
        norm = normalize_whitespace(line).upper()
        if match := HEADER_PAGE_RE.search(norm): result["page"] = int(match.group(1))
        if match := HEADER_TAPE_RE.search(norm): result["tape"] = match.group(1).replace(" ", "")
        if fuzzy_find(norm, "APOLLO") and fuzzy_find(norm, "TRANSCRIPTION"): result["is_apollo_title"] = True
    return result


def clean_trailing_footer(text: str) -> str:
    return re.sub(r"\s+tape\s+[\d/]+\s+Page\s+\d+\s*$", "", text, flags=re.IGNORECASE)


def build_page_json(rows: list[dict], lines: list[str], page_num: int, page_offset: int = 0, valid_speakers: list[str] = None, text_replacements: dict[str, str] = None, mission_keywords: list[str] = None) -> dict:
    header_info = extract_header_metadata(lines, page_num, page_offset)
    blocks = []
    for row in rows:
        if row["type"] == "header": continue
        block_type = "continuation" if row["type"] == "text" else row["type"]
        block = {"type": block_type}
        if block_type == "comm":
            if row["timestamp"]: block["timestamp"] = row["timestamp"]
            if row["speaker"]: block["speaker"] = row["speaker"]
        if row["text"]: block["text"] = clean_trailing_footer(row["text"])
        if blocks and block_type == "continuation" and blocks[-1]["type"] == "continuation":
            blocks[-1]["text"] = (blocks[-1]["text"] + " " + block["text"]).strip()
        else:
            blocks.append(block)
    
    ts_corrector = TimestampCorrector()
    blocks = ts_corrector.process_blocks(blocks)
    if valid_speakers:
        blocks = SpeakerCorrector(valid_speakers).process_blocks(blocks)
    lexicon_path = Path("assets/lexicon/apollo11_lexicon.json")
    if lexicon_path.exists():
        blocks = TextCorrector(lexicon_path, text_replacements, mission_keywords).process_blocks(blocks)
    return {"header": header_info, "blocks": blocks}
