"""
OCR Output Parser.

Parses plain text OCR output into structured blocks for NASA transcripts.
"""

import difflib
import re
from pathlib import Path

from src.correctors.speaker_corrector import SpeakerCorrector
from src.correctors.text_corrector import TextCorrector
from src.correctors.timestamp_corrector import TimestampCorrector

# Regex patterns for transcript parsing
TS_CHARS = r"[\dOI'()]"
TIMESTAMP_STRICT_RE = re.compile(rf"^{TS_CHARS}{{2}}(?:\s+{TS_CHARS}{{2}}){{2,3}}$")
TIMESTAMP_PREFIX_RE = re.compile(rf"^({TS_CHARS}{{2}}(?:\s+{TS_CHARS}{{2}}){{2,3}})\b")
TIMESTAMP_EMBEDDED_RE = re.compile(rf"\s+({TS_CHARS}{{2}}(?:\s+{TS_CHARS}{{2}}){{2,3}})\b")
REV_EMBEDDED_RE = re.compile(r"(\([RE][FV]\s+\d+\))", re.IGNORECASE)

SPEAKER_LINE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,6}(?:\s*\([A-Z0-9]+\))?$")
SPEAKER_PAREN_RE = re.compile(r"^\(([A-Z0-9]+)\)$")
LOCATION_PAREN_RE = re.compile(r"^\s*\(([A-Z0-9\s]+)\)\s*$")
HEADER_PAGE_RE = re.compile(r"\bPAGE\s*(\d{1,4})\b", re.IGNORECASE)
HEADER_TAPE_RE = re.compile(r"\bTAPE\s*([0-9]{1,2}\s*/\s*[0-9]{1,2})\b", re.IGNORECASE)
HEADER_KEYWORDS = ("GOSS", "NET", "TAPE", "PAGE", "APOLLO", "AIR-TO-GROUND")
END_OF_TAPE_KEYWORD = "END OF TAPE"
TRANSITION_KEYWORDS = (
    "REST PERIOD",
    "NO COMMUNICATIONS",
    "NO COMMUNICATION",
    "LOSS OF SIGNAL",
    "AOS",
    "LOS",
    "SILENCE",
)
LINE_TAG_RE = re.compile(r"^\[(HEADER|FOOTER|ANNOTATION|COMM|META)\]\s*", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", text).strip()


def parse_ocr_text(text: str, page_num: int, mission_keywords: list[str] = None) -> list[dict]:
    """
    Parse plain OCR output into structured rows.
    """
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = []
    for raw in raw_lines:
        tag_match = LINE_TAG_RE.match(raw)
        forced_type = None
        if tag_match:
            forced_type = tag_match.group(1).lower()
            raw = raw[tag_match.end():].strip()
        if raw:
            lines.append({"text": raw, "forced": forced_type})
    
    # Pre-process lines iteratively to split embedded components
    current_processing = lines
    final_lines = []
    
    while current_processing:
        entry = current_processing.pop(0)
        line = entry["text"]
        forced_type = entry["forced"]
        match = None
        
        # If line starts with a timestamp, search for splits after the initial timestamp zone
        search_start = 0
        if TIMESTAMP_PREFIX_RE.match(line):
            search_start = 12 

        # 1. Check for embedded timestamp
        ts_match = TIMESTAMP_EMBEDDED_RE.search(line, search_start)
        # 2. Check for embedded REV marker
        rev_match = REV_EMBEDDED_RE.search(line, search_start)
        # 3. Check for trailing mission keyword
        kw_match = None
        if mission_keywords:
            for kw in mission_keywords:
                if len(kw) > 3:
                    kw_re = re.compile(rf"\s+({{re.escape(kw)}})", re.IGNORECASE)
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
            if part1: final_lines.append({"text": part1, "forced": forced_type})
            if part2: current_processing.insert(0, {"text": part2, "forced": forced_type})
        else:
            final_lines.append({"text": line, "forced": forced_type})
    
    lines = final_lines
    if not lines: return []

    # Find first timestamp to identify header zone
    first_ts_idx = next(
        (i for i, entry in enumerate(lines)
         if TIMESTAMP_STRICT_RE.match(entry["text"]) or TIMESTAMP_PREFIX_RE.match(entry["text"])),
        None
    )

    rows = []
    line_index = 0
    pending_ts = ""
    pending_speaker = ""
    pending_location = ""
    pending_text = []
    pending_force_comm = False

    def flush_pending():
        nonlocal pending_ts, pending_speaker, pending_location, pending_text, line_index, pending_force_comm
        if not pending_ts and not pending_speaker and not pending_text: return
        line_index += 1
        rows.append({
            "page": page_num + 1, "line": line_index,
            "type": "comm" if (pending_ts or pending_force_comm) else "text",
            "timestamp": pending_ts, "speaker": pending_speaker,
            "location": pending_location,
            "text": " ".join(pending_text).strip(),
        })
        pending_ts = ""; pending_speaker = ""; pending_location = ""; pending_text = []
        pending_force_comm = False

    prev_comm_like = False
    saw_comm_or_ts = False
    for idx, entry in enumerate(lines):
        line = entry["text"]
        forced_type = entry["forced"]
        upper = line.upper()
        has_lower = any(c.islower() for c in line)
        location_only = LOCATION_PAREN_RE.match(line)

        if forced_type == "annotation":
            if TIMESTAMP_PREFIX_RE.match(line) or TIMESTAMP_STRICT_RE.match(line):
                forced_type = "comm"
            elif prev_comm_like and has_lower and "(REV" not in upper and "(RFV" not in upper:
                forced_type = "comm"

        if forced_type == "comm" and not saw_comm_or_ts:
            if not TIMESTAMP_PREFIX_RE.match(line) and not TIMESTAMP_STRICT_RE.match(line):
                pending_text.append(line)
                continue

        if location_only:
            location_value = location_only.group(1).strip()
            if pending_ts:
                pending_location = location_value
                continue
            if rows and rows[-1].get("type") == "comm" and not rows[-1].get("location"):
                rows[-1]["location"] = location_value
                continue

        if forced_type is None and line.lstrip().startswith("***"):
            flush_pending()
            line_index += 1
            rows.append({
                "page": page_num + 1, "line": line_index,
                "type": "footer",
                "timestamp": "", "speaker": "", "location": "", "text": line,
            })
            prev_comm_like = False
            continue

        if forced_type is None and fuzzy_find(line, END_OF_TAPE_KEYWORD):
            flush_pending()
            line_index += 1
            rows.append({
                "page": page_num + 1, "line": line_index,
                "type": "meta",
                "timestamp": "", "speaker": "", "location": "", "text": line,
            })
            prev_comm_like = False
            continue

        if forced_type == "header" and ("(REV" in upper or "(RFV" in upper):
            forced_type = "annotation"

        if forced_type in ("header", "footer", "annotation", "meta"):
            flush_pending()
            line_index += 1
            rows.append({
                "page": page_num + 1, "line": line_index,
                "type": "meta" if forced_type == "meta" else forced_type,
                "timestamp": "", "speaker": "", "location": "", "text": line,
            })
            prev_comm_like = forced_type == "annotation" and has_lower
            continue

        if forced_type == "comm":
            pending_force_comm = True
            saw_comm_or_ts = True

        # Standalone timestamp
        if TIMESTAMP_STRICT_RE.match(line):
            flush_pending()
            pending_ts = line
            prev_comm_like = True
            saw_comm_or_ts = True
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
                    # Check if location is attached to speaker
                    loc_match = re.search(r"\(([^)]+)\)", pending_speaker)
                    if loc_match:
                        pending_location = loc_match.group(1)
                        pending_speaker = pending_speaker[:loc_match.start()].strip()
                    elif tokens and re.match(r"^\([^)]+\)$", tokens[0]):
                        pending_location = tokens.pop(0).strip("()")
                if tokens: pending_text.append(" ".join(tokens))
            prev_comm_like = True
            saw_comm_or_ts = True
            continue

        if forced_type is None:
            # Annotations
            is_header = not pending_ts and first_ts_idx is not None and idx <= first_ts_idx and any(fuzzy_find(line, kw) for kw in HEADER_KEYWORDS)
            is_footer = not pending_ts and (
                "***" in line
                or "ASTERISK" in upper
                or (idx >= len(lines) - 3 and (HEADER_PAGE_RE.search(upper) or HEADER_TAPE_RE.search(upper)))
            )
            is_annotation = "(REV" in upper or "(RFV" in upper or (mission_keywords and not pending_ts and any(fuzzy_find(line, kw) for kw in mission_keywords))
            is_end_of_tape = fuzzy_find(line, END_OF_TAPE_KEYWORD)
            is_transition = any(fuzzy_find(line, kw) for kw in TRANSITION_KEYWORDS)

            if is_header or is_footer or is_annotation or is_end_of_tape or is_transition:
                flush_pending()
                line_index += 1
                rows.append({
                    "page": page_num + 1, "line": line_index,
                    "type": "meta" if (is_end_of_tape or is_transition) else ("header" if is_header else ("footer" if is_footer else "annotation")),
                    "timestamp": "", "speaker": "", "location": "", "text": line,
                })
                continue

        if pending_ts:
            # Check for location tag at the start of the line
            loc_at_start_match = re.match(r"^\s*\(([A-Z0-9\s]+)\)\s*", line)
            if loc_at_start_match:
                pending_location = loc_at_start_match.group(1).strip()
                line = line[loc_at_start_match.end():].strip()
                if not line: continue
            
            if SPEAKER_LINE_RE.match(line) or SPEAKER_PAREN_RE.match(line):
                pending_speaker = f"{pending_speaker} {line}".strip() if pending_speaker else line
                continue
            pending_text.append(line)
            prev_comm_like = True
            saw_comm_or_ts = True
            continue

        pending_text.append(line)
        prev_comm_like = prev_comm_like or pending_force_comm

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
    # 1. Double pattern: "Tape XX Page YY" or "Page YY Tape XX"
    text = re.sub(r"[\s\.\n\r]+(?:T[A-Z0-9]{2,})\s*[\d/IX]+\s+(?:P[A-Z0-9]{2,})\s*[\d/IX]+\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s\.\n\r]+(?:P[A-Z0-9]{2,})\s*[\d/IX]+\s+(?:T[A-Z0-9]{2,})\s*[\d/IX]+\s*$", "", text, flags=re.IGNORECASE)
    
    # 2. Single pattern: "Tape XX" or "Page YY" at the very end
    text = re.sub(r"[\s\.\n\r]+(?:TAPS|TAPC|TAPE|TYPE|TANE|PAGE|PAGS|PACE|PAXE|PAGO|Paze|Page|Pags)\s*[\d/IX]+\s*$", "", text, flags=re.IGNORECASE)
    
    return text.strip()


def build_page_json(
    rows: list[dict],
    lines: list[str],
    page_num: int,
    page_offset: int = 0,
    valid_speakers: list[str] = None,
    text_replacements: dict[str, str] = None,
    mission_keywords: list[str] = None,
    valid_locations: list[str] = None,
    initial_ts: str = None,
    previous_block_type: str = None
) -> dict:
    header_info = extract_header_metadata(lines, page_num, page_offset)
    blocks = []
    for row in rows:
        if row["type"] == "header": continue
        block_type = "continuation" if row["type"] == "text" else row["type"]
        block = {"type": block_type}
        
        if block_type == "comm":
            if row["timestamp"]: block["timestamp"] = row["timestamp"]
            if row["speaker"]: block["speaker"] = row["speaker"]
            if row["location"]: block["location"] = row["location"]
            
        if row["text"]:
            block["text"] = row["text"]

        if blocks and block_type == "continuation" and blocks[-1]["type"] == "continuation":
            blocks[-1]["text"] = (blocks[-1]["text"] + " " + block["text"]).strip()
        else:
            blocks.append(block)

    if blocks and blocks[0]["type"] == "continuation" and previous_block_type in ("comm", "continuation"):
        blocks[0]["continuation_from_prev"] = True
    
    # Final cleanup of footers and locations on merged text
    for block in blocks:
        if block.get("text"):
            text = clean_trailing_footer(block["text"])
            # Remove any residual location tag at the start: "(TRANQ) ..."
            if valid_locations:
                for loc in valid_locations:
                    text = re.sub(rf"^\({re.escape(loc)}\)\s*", "", text, flags=re.IGNORECASE)
            block["text"] = text.strip()
    
    # Post-process timestamps
    ts_corrector = TimestampCorrector(initial_ts)
    blocks = ts_corrector.process_blocks(blocks)
    
    # Fuzzy correct locations and speakers
    if valid_locations:
        for block in blocks:
            loc = block.get("location")
            if loc:
                best_loc = difflib.get_close_matches(loc.upper(), valid_locations, n=1, cutoff=0.5)
                if best_loc:
                    block["location"] = best_loc[0]

    if valid_speakers:
        blocks = SpeakerCorrector(valid_speakers).process_blocks(blocks)

    # Post-process text (spelling and noise)
    lexicon_path = Path("assets/lexicon/apollo11_lexicon.json")
    if lexicon_path.exists():
        blocks = TextCorrector(lexicon_path, text_replacements, mission_keywords).process_blocks(blocks)
    return {"header": header_info, "blocks": blocks}
