"""
OCR Output Parser V3.1 - Enhanced Robustness.

Parses plain text OCR output into structured NASA transcript blocks.
Handles flexible timestamps, location extraction, and aggressive line splitting.
"""

import difflib
import re
from pathlib import Path

from src.correctors.speaker_corrector import SpeakerCorrector
from src.correctors.text_corrector import TextCorrector
from src.correctors.timestamp_corrector import TimestampCorrector

# Flexible Timestamp: 4 groups, last one can be 1 or 2 digits
TS_CHARS = r"[\dOI'I\)\(C]"
TIMESTAMP_PATTERN = rf"{TS_CHARS}{{2}}\s+{TS_CHARS}{{2}}\s+{TS_CHARS}{{2}}\s+{TS_CHARS}{{1,2}}"
TIMESTAMP_RE = re.compile(rf"({TIMESTAMP_PATTERN})")

# Known Speakers pattern
SPEAKER_RE = re.compile(r"\b(CC|CDR|LMP|CMP|MS|SC)\b")

HEADER_KEYWORDS = ("GOSS", "NET", "TAPE", "PAGE", "APOLLO", "AIR-TO-GROUND")

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def parse_ocr_text(text: str, page_num: int, mission_keywords: list[str] = None) -> list[dict]:
    # Preliminary cleanup
    text = text.replace('|', ' ')
    
    raw_lines = text.splitlines()
    processed_lines = []
    
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        
        # Split line if timestamp is embedded
        ts_match = TIMESTAMP_RE.search(line)
        if ts_match and ts_match.start() > 0:
            prefix = line[:ts_match.start()].strip()
            remainder = line[ts_match.start():].strip()
            if prefix: processed_lines.append(prefix)
            if remainder: processed_lines.append(remainder)
        else:
            processed_lines.append(line)

    rows = []
    current_ts = None
    current_speaker = None
    current_location = None
    current_text_parts = []

    def flush_comm():
        nonlocal current_ts, current_speaker, current_location, current_text_parts
        if current_ts:
            text = " ".join(current_text_parts).strip()
            
            # Extract location from text if present at the start: "(TRANQ) Roger."
            loc_at_start = re.match(r"^\s*\(([^)]+)\)\s*", text)
            if loc_at_start:
                extracted_loc = loc_at_start.group(1).strip()
                # If we don't already have a location, use this one
                if not current_location:
                    current_location = extracted_loc
                # Remove location from text
                text = text[loc_at_start.end():].strip()

            rows.append({
                "type": "comm",
                "timestamp": current_ts,
                "speaker": current_speaker or "",
                "location": current_location or "",
                "text": text
            })
            current_ts = None
            current_speaker = None
            current_location = None
            current_text_parts = []

    for line in processed_lines:
        ts_start_match = TIMESTAMP_RE.match(line)
        
        if ts_start_match:
            flush_comm()
            current_ts = ts_start_match.group(1)
            remainder = line[len(current_ts):].strip()
            
            # Look for Speaker and Location in the remainder
            spk_match = SPEAKER_RE.search(remainder)
            if spk_match:
                current_speaker = spk_match.group(1)
                text_after_spk = remainder[spk_match.end():].strip()
                
                # Check for location in parentheses near speaker
                loc_match = re.search(r"\(([^)]+)\)", remainder)
                if loc_match:
                    current_location = loc_match.group(1)
                    # Clean the specific location tag from the text part
                    text_after_spk = text_after_spk.replace(f"({current_location})", "").strip()
                
                current_text_parts = [text_after_spk]
            else:
                current_text_parts = [remainder]
            continue

        is_header = any(kw in line.upper() for kw in HEADER_KEYWORDS)
        if is_header and not current_ts:
            rows.append({"type": "header", "text": line})
            continue
            
        if "(REV" in line.upper() or "(RFV" in line.upper():
            flush_comm()
            rows.append({"type": "annotation", "text": line})
            continue

        if current_ts:
            current_text_parts.append(line)
        else:
            rows.append({"type": "annotation", "text": line})

    flush_comm()
    return rows

def extract_header_metadata(lines: list[str], page_num: int, page_offset: int = 0) -> dict:
    result = {"page": page_num + 1 + page_offset, "tape": None, "is_apollo_title": False}
    text = " ".join(lines[:10]).upper()
    pg_match = re.search(r"PAGE\s*(\d+)", text)
    if pg_match: result["page"] = int(pg_match.group(1))
    tp_match = re.search(r"TAPE\s*(\d+/\d+)", text)
    if tp_match: result["tape"] = tp_match.group(1)
    if "APOLLO" in text and "TRANSCRIPTION" in text:
        result["is_apollo_title"] = True
    return result

def build_page_json(rows: list[dict], lines: list[str], page_num: int, page_offset: int = 0, valid_speakers: list[str] = None, text_replacements: dict[str, str] = None, mission_keywords: list[str] = None, valid_locations: list[str] = None, initial_ts: str = None) -> dict:
    header_info = extract_header_metadata(lines, page_num, page_offset)
    final_blocks = []
    for row in rows:
        if row["type"] == "header": continue
        if "text" in row:
            row["text"] = re.sub(r"\s+", " ", row["text"]).strip()
        final_blocks.append(row)
        
    ts_corrector = TimestampCorrector(initial_ts)
    final_blocks = ts_corrector.process_blocks(final_blocks)
    
    if valid_speakers:
        final_blocks = SpeakerCorrector(valid_speakers).process_blocks(final_blocks)
        
    if valid_locations:
        for block in final_blocks:
            loc = block.get("location")
            if loc:
                best_loc = difflib.get_close_matches(loc.upper(), valid_locations, n=1, cutoff=0.5)
                if best_loc:
                    block["location"] = best_loc[0]

    return {"header": header_info, "blocks": final_blocks}
