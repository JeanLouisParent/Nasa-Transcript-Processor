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

SPEAKER_TOKEN_RE = re.compile(r"^[A-Z0-9]{1,8}(?:/[A-Z0-9]{1,8})?$")
SPEAKER_LINE_RE = re.compile(
    r"^[A-Z0-9]{1,8}(?:/[A-Z0-9]{1,8})?"
    r"(?:\s+[A-Z0-9]{1,8}(?:/[A-Z0-9]{1,8})?){0,2}"
    r"(?:\s*\([A-Z0-9]+\))?$"
)
SPEAKER_PAREN_RE = re.compile(r"^\(([A-Z0-9]+)\)$")
LOCATION_PAREN_RE = re.compile(r"^\s*\(([A-Z0-9\s]+)\)\s*$")
HEADER_PAGE_RE = re.compile(r"\b(?:PAGE|PLAY|LAY)\s*(\d{1,4})\b", re.IGNORECASE)
HEADER_TAPE_RE = re.compile(r"\bTAPE\s*([0-9]{1,2}\s*/\s*[0-9]{1,2})\b", re.IGNORECASE)
HEADER_PAGE_ONLY_RE = re.compile(r"^\s*(?:PAGE|PLAY|LAY)\s*\d{1,4}\s*$", re.IGNORECASE)
HEADER_TAPE_ONLY_RE = re.compile(r"^\s*TAPE\s*\d{1,2}\s*/\s*\d{1,2}\s*$", re.IGNORECASE)
HEADER_TAPE_SIMPLE_RE = re.compile(r"^\s*TAPE\s*\d{1,4}\s*$", re.IGNORECASE)
HEADER_TAPE_PAGE_ONLY_RE = re.compile(
    r"^\s*(?:TAPE\s*\d{1,2}\s*/\s*\d{1,2}\s+PAGE\s*\d{1,4}"
    r"|(?:PAGE|PLAY|LAY)\s*\d{1,4}\s+TAPE\s*\d{1,2}\s*/\s*\d{1,2})\s*$",
    re.IGNORECASE,
)
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
LUNAR_REV_RE = re.compile(r"^--\s*(BEGIN|END)\s+LUNAR\s+REV\s+(\d+)\b", re.IGNORECASE)
REST_PERIOD_RE = re.compile(r"\bREST\s+PERIOD\b", re.IGNORECASE)
NO_COMM_RE = re.compile(r"\bNO\s+COMMUNICATIONS?\b", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", text).strip()


def should_split_embedded_timestamp(line: str, match: re.Match) -> bool:
    """
    Split only when the embedded timestamp is followed by a plausible speaker token.
    """
    remainder = line[match.end():].lstrip()
    if not remainder:
        return False
    token = remainder.split()[0]
    if SPEAKER_TOKEN_RE.match(token):
        return True
    if remainder.startswith("("):
        close_idx = remainder.find(")")
        if close_idx != -1:
            after = remainder[close_idx + 1:].lstrip()
            if after:
                token = after.split()[0]
                if SPEAKER_TOKEN_RE.match(token):
                    return True
    return False


def parse_ocr_text(text: str, page_num: int, mission_keywords: list[str] | None = None) -> list[dict]:
    """
    Parse plain OCR output into structured rows.
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
        if ts_match and not should_split_embedded_timestamp(line, ts_match):
            ts_match = None
        # 2. Check for embedded REV marker
        rev_match = REV_EMBEDDED_RE.search(line, search_start)
        # 3. Check for trailing mission keyword
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

    lines = final_lines
    if not lines:
        return []

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
    pending_ts_hint: str | None = None
    timestamp_only_run = 0
    timestamp_run_start_idx: int | None = None
    timestamp_list_mode = False
    timestamp_list_start_idx: int | None = None
    timestamp_list_row_idx: int | None = None

    def flush_pending():
        nonlocal pending_ts, pending_speaker, pending_location, pending_text, line_index, pending_force_comm, pending_ts_hint
        if not pending_ts and not pending_speaker and not pending_text:
            return
        line_index += 1
        row = {
            "page": page_num + 1, "line": line_index,
            "type": "comm" if (pending_ts or pending_force_comm) else "text",
            "timestamp": pending_ts, "speaker": pending_speaker,
            "location": pending_location,
            "text": " ".join(pending_text).strip(),
        }
        if pending_ts_hint:
            row["timestamp_suffix_hint"] = pending_ts_hint
        rows.append(row)
        pending_ts = ""
        pending_speaker = ""
        pending_location = ""
        pending_text = []
        pending_force_comm = False
        pending_ts_hint = None

    prev_comm_like = False
    saw_comm_or_ts = False
    def take_speaker_tokens(tokens: list[str]) -> tuple[str, list[str]]:
        speaker_tokens = []
        while tokens and len(speaker_tokens) < 3:
            token = tokens[0]
            if len(token) == 1:
                break
            if SPEAKER_TOKEN_RE.match(token):
                speaker_tokens.append(tokens.pop(0))
                continue
            break
        return " ".join(speaker_tokens), tokens

    for idx, entry in enumerate(lines):
        line = entry["text"]
        forced_type = entry["forced"]
        upper = line.upper()
        has_lower = any(c.islower() for c in line)
        location_only = LOCATION_PAREN_RE.match(line)
        is_header_only = (
            HEADER_PAGE_ONLY_RE.match(line)
            or HEADER_TAPE_ONLY_RE.match(line)
            or HEADER_TAPE_PAGE_ONLY_RE.match(line)
        )

        has_timestamp = TIMESTAMP_PREFIX_RE.match(line) or TIMESTAMP_STRICT_RE.match(line)

        if timestamp_list_mode and forced_type in ("header", "footer", "annotation", "meta"):
            if not (
                (forced_type == "footer" and line.lstrip().startswith("***"))
                or (forced_type == "meta" and fuzzy_find(line, END_OF_TAPE_KEYWORD))
                or (
                    forced_type == "header"
                    and (
                        HEADER_PAGE_ONLY_RE.match(line)
                        or HEADER_TAPE_ONLY_RE.match(line)
                        or HEADER_TAPE_PAGE_ONLY_RE.match(line)
                    )
                )
            ):
                forced_type = None

        if forced_type in ("header", "footer", "annotation", "meta") and has_timestamp:
            forced_type = "comm"

        if is_header_only and not has_timestamp:
            continue

        if forced_type == "meta":
            if idx <= (first_ts_idx or -1) and (HEADER_PAGE_RE.search(upper) or HEADER_TAPE_RE.search(upper)):
                forced_type = "header"
            elif pending_ts or prev_comm_like or has_lower:
                forced_type = "comm"

        if (
            forced_type is None
            and prev_comm_like
            and not has_lower
            and len(line.strip()) <= 8
            and not line.lstrip().startswith("***")
        ):
            forced_type = "comm"

        if forced_type == "annotation":
            if TIMESTAMP_PREFIX_RE.match(line) or TIMESTAMP_STRICT_RE.match(line):
                forced_type = "comm"
            elif "(REV" not in upper and "(RFV" not in upper:
                if has_lower:
                    forced_type = "comm"
                elif len(line.strip()) <= 4 and not has_lower:
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

        if SPEAKER_LINE_RE.match(line) and not pending_ts and not timestamp_list_mode:
            if rows and rows[-1].get("type") == "comm" and not rows[-1].get("speaker"):
                rows[-1]["speaker"] = line.strip()
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

        if forced_type == "footer":
            if fuzzy_find(line, END_OF_TAPE_KEYWORD):
                forced_type = "meta"
            elif not line.lstrip().startswith("***"):
                forced_type = "comm"

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
            if timestamp_only_run == 0:
                timestamp_run_start_idx = len(rows)
            pending_ts = line
            timestamp_only_run += 1
            if timestamp_only_run >= 5:
                timestamp_list_mode = True
                if timestamp_list_start_idx is None:
                    timestamp_list_start_idx = timestamp_run_start_idx
                    timestamp_list_row_idx = timestamp_list_start_idx
            prev_comm_like = True
            saw_comm_or_ts = True
            continue

        # Prefix timestamp
        prefix_match = TIMESTAMP_PREFIX_RE.match(line)
        if prefix_match:
            flush_pending()
            if timestamp_only_run == 0:
                timestamp_run_start_idx = len(rows)
            pending_ts = prefix_match.group(1)
            timestamp_only_run += 1
            if timestamp_only_run >= 5:
                timestamp_list_mode = True
                if timestamp_list_start_idx is None:
                    timestamp_list_start_idx = timestamp_run_start_idx
                    timestamp_list_row_idx = timestamp_list_start_idx
            remainder = line[len(pending_ts):].strip()
            if remainder:
                tokens = remainder.split()
                if (
                    len(tokens) >= 2
                    and tokens[0].isdigit()
                    and len(tokens[0]) <= 2
                    and SPEAKER_TOKEN_RE.match(tokens[1])
                ):
                    pending_ts_hint = tokens.pop(0)
                speaker, tokens = take_speaker_tokens(tokens)
                if speaker:
                    pending_speaker = speaker
                    # Check if location is attached to speaker
                    loc_match = re.search(r"\(([^)]+)\)", pending_speaker)
                    if loc_match:
                        pending_location = loc_match.group(1)
                        pending_speaker = pending_speaker[:loc_match.start()].strip()
                    elif tokens and re.match(r"^\([^)]+\)$", tokens[0]):
                        pending_location = tokens.pop(0).strip("()")
                if tokens:
                    pending_text.append(" ".join(tokens))
            prev_comm_like = True
            saw_comm_or_ts = True
            continue

        if timestamp_list_mode and pending_ts and not pending_speaker and not pending_text:
            flush_pending()
        if not timestamp_list_mode and timestamp_only_run:
            timestamp_only_run = 0
            timestamp_run_start_idx = None

        if not pending_ts and not timestamp_list_mode and (SPEAKER_LINE_RE.match(line) or LOCATION_PAREN_RE.match(line)):
            continue
        if timestamp_list_mode and not pending_ts and forced_type is None and timestamp_list_row_idx is not None:
            if timestamp_list_row_idx >= len(rows):
                timestamp_list_row_idx = len(rows) - 1 if rows else None
            if timestamp_list_row_idx is not None and timestamp_list_row_idx >= 0:
                row = rows[timestamp_list_row_idx]
                if SPEAKER_LINE_RE.match(line):
                    if row.get("speaker"):
                        timestamp_list_row_idx += 1
                        if timestamp_list_row_idx >= len(rows):
                            timestamp_list_row_idx = len(rows) - 1
                        row = rows[timestamp_list_row_idx]
                    row["speaker"] = line.strip()
                elif LOCATION_PAREN_RE.match(line):
                    if row.get("location"):
                        timestamp_list_row_idx += 1
                        if timestamp_list_row_idx >= len(rows):
                            timestamp_list_row_idx = len(rows) - 1
                        row = rows[timestamp_list_row_idx]
                    row["location"] = line.strip("()").strip()
                else:
                    if row.get("text") and (row.get("speaker") or row.get("location")):
                        timestamp_list_row_idx += 1
                        if timestamp_list_row_idx >= len(rows):
                            timestamp_list_row_idx = len(rows) - 1
                        row = rows[timestamp_list_row_idx]
                    row["text"] = (row.get("text", "") + " " + line).strip()
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
                if not line:
                    continue

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

    if timestamp_list_mode:
        ts_row_indices = [i for i, row in enumerate(rows) if row.get("timestamp")]
        start_idx = timestamp_list_start_idx
        if start_idx is None and ts_row_indices:
            start_idx = ts_row_indices[0]
        if start_idx is not None and ts_row_indices:
            has_speaker = any(rows[i].get("speaker") for i in ts_row_indices)
            if not has_speaker:
                ts_line_indices = [
                    i for i, entry in enumerate(lines)
                    if TIMESTAMP_STRICT_RE.match(entry["text"]) or TIMESTAMP_PREFIX_RE.match(entry["text"])
                ]
                if ts_line_indices:
                    row_idx = start_idx
                    for i in range(start_idx, len(rows)):
                        rows[i]["speaker"] = ""
                        rows[i]["location"] = ""
                        rows[i]["text"] = ""
                    for entry in lines[ts_line_indices[-1] + 1:]:
                        if row_idx >= len(rows):
                            break
                        line = entry["text"]
                        if (
                            HEADER_PAGE_ONLY_RE.match(line)
                            or HEADER_TAPE_ONLY_RE.match(line)
                            or HEADER_TAPE_PAGE_ONLY_RE.match(line)
                        ):
                            continue
                        row = rows[row_idx]
                        if SPEAKER_LINE_RE.match(line):
                            if row.get("speaker"):
                                row_idx += 1
                                if row_idx >= len(rows):
                                    break
                                row = rows[row_idx]
                            row["speaker"] = line.strip()
                        elif LOCATION_PAREN_RE.match(line):
                            if row.get("location"):
                                row_idx += 1
                                if row_idx >= len(rows):
                                    break
                                row = rows[row_idx]
                            row["location"] = line.strip("()").strip()
                        else:
                            if row.get("text") and (row.get("speaker") or row.get("location")):
                                row_idx += 1
                                if row_idx >= len(rows):
                                    break
                                row = rows[row_idx]
                            row["text"] = (row.get("text", "") + " " + line).strip()
                    while rows and not (
                        rows[-1].get("timestamp")
                        or rows[-1].get("speaker")
                        or rows[-1].get("location")
                        or rows[-1].get("text")
                    ):
                        rows.pop()
    return rows


def fuzzy_find(text: str, keyword: str, threshold: float = 0.6) -> bool:
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


def extract_header_metadata(lines: list[str], page_num: int, page_offset: int = 0) -> dict:
    result = {"page": page_num + 1 + page_offset, "tape": None, "is_apollo_title": False}
    first_ts_idx = next(
        (i for i, ln in enumerate(lines) if TIMESTAMP_STRICT_RE.match(ln) or TIMESTAMP_PREFIX_RE.match(ln)),
        None
    )
    header_lines = lines[:first_ts_idx] if first_ts_idx is not None else lines[:10]
    for line in header_lines:
        norm = normalize_whitespace(line).upper()
        if fuzzy_find(norm, "APOLLO") and fuzzy_find(norm, "TRANSCRIPTION"):
            result["is_apollo_title"] = True
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
    valid_speakers: list[str] | None = None,
    text_replacements: dict[str, str] | None = None,
    mission_keywords: list[str] | None = None,
    valid_locations: list[str] | None = None,
    initial_ts: str | None = None,
    previous_block_type: str | None = None
) -> dict:
    header_info = extract_header_metadata(lines, page_num, page_offset)
    blocks = []
    for row in rows:
        if row["type"] == "header":
            continue
        block_type = "continuation" if row["type"] == "text" else row["type"]
        block = {"type": block_type}

        if block_type == "comm":
            if row["timestamp"]:
                block["timestamp"] = row["timestamp"]
            if row["speaker"]:
                block["speaker"] = row["speaker"]
            if row["location"]:
                block["location"] = row["location"]
            if row.get("timestamp_suffix_hint"):
                block["timestamp_suffix_hint"] = row["timestamp_suffix_hint"]

        if row["text"]:
            text_value = row["text"]
            if (
                block_type == "comm"
                and (
                    HEADER_PAGE_ONLY_RE.match(text_value)
                    or HEADER_TAPE_ONLY_RE.match(text_value)
                    or HEADER_TAPE_SIMPLE_RE.match(text_value)
                    or HEADER_TAPE_PAGE_ONLY_RE.match(text_value)
                )
            ):
                continue
            block["text"] = text_value
            if block_type == "meta" and fuzzy_find(row["text"], END_OF_TAPE_KEYWORD):
                block["meta_type"] = "end_of_tape"
            if block_type == "footer" and row["text"].lstrip().startswith("***"):
                block["text"] = "*** Three asterisks denote clipping of words and phrases."

        if block_type == "continuation" and blocks:
            if blocks[-1].get("text") and block.get("text"):
                blocks[-1]["text"] = (blocks[-1]["text"] + " " + block["text"]).strip()
            elif block.get("text"):
                blocks[-1]["text"] = block["text"]
            continue
        if (
            block_type == "meta"
            and block.get("text")
            and blocks
            and blocks[-1].get("type") == "meta"
            and not blocks[-1].get("meta_type")
            and not block.get("meta_type")
        ):
            blocks[-1]["text"] = (blocks[-1].get("text", "") + " " + block["text"]).strip()
            continue
        if blocks and block_type == "continuation" and blocks[-1]["type"] == "continuation":
            blocks[-1]["text"] = (blocks[-1]["text"] + " " + block["text"]).strip()
        else:
            blocks.append(block)

    merged_blocks = []
    for block in blocks:
        if block.get("type") == "comm" and block.get("text"):
            text_value = block["text"].strip()
            match = LUNAR_REV_RE.match(text_value)
            if match:
                action = match.group(1).upper()
                rev_num = match.group(2)
                block["text"] = f"{action} LUNAR REV {rev_num}"
                ts = block.get("timestamp", "")
                if ts:
                    parts = ts.split()
                    if len(parts) == 4:
                        parts[-1] = "--"
                        block["timestamp"] = " ".join(parts)
                    elif len(parts) == 3:
                        block["timestamp"] = " ".join(parts + ["--"])
                block["type"] = "meta"
                block["meta_type"] = "lunar_rev"
                block.pop("speaker", None)
                block.pop("location", None)

    for block in blocks:
        if (
            block.get("type") == "continuation"
            and block.get("text")
            and merged_blocks
            and merged_blocks[-1].get("type") == "comm"
        ):
            text = block["text"]
            if text[:1] in ";,.)" or (text[:1].islower()):
                merged_blocks[-1]["text"] = (merged_blocks[-1].get("text", "") + " " + text).strip()
                continue
        if (
            block.get("type") == "comm"
            and not block.get("speaker")
            and block.get("text")
            and merged_blocks
            and merged_blocks[-1].get("type") == "comm"
        ):
            text = block["text"]
            if text[:1] in ";,.)" or (text[:1].islower()):
                merged_blocks[-1]["text"] = (merged_blocks[-1].get("text", "") + " " + text).strip()
                continue
        if (
            block.get("type") in ("meta", "continuation", "annotation")
            and block.get("text")
            and merged_blocks
            and merged_blocks[-1].get("type") == "comm"
        ):
            text = block["text"]
            if text[:1] in ";,.)" or (text[:1].islower()):
                merged_blocks[-1]["text"] = (merged_blocks[-1].get("text", "") + " " + text).strip()
                continue
        merged_blocks.append(block)
    blocks = merged_blocks

    # Canonicalize REST PERIOD pages
    rest_period_found = False
    for block in blocks:
        text_val = block.get("text", "")
        if text_val and REST_PERIOD_RE.search(text_val) and NO_COMM_RE.search(text_val):
            block["type"] = "meta"
            block["meta_type"] = "rest_period"
            block["text"] = "REST PERIOD - NO COMMUNICATIONS"
            rest_period_found = True
        if text_val and is_transcription_header(text_val):
            block["type"] = "meta"
            block["meta_type"] = "transcript_header"
            block["text"] = "AIR-TO-GROUND VOICE TRANSCRIPTION"

    if blocks and blocks[0]["type"] == "continuation" and previous_block_type in ("comm", "continuation"):
        blocks[0]["continuation_from_prev"] = True

    if rest_period_found or any(b.get("meta_type") == "rest_period" for b in blocks):
        header_info["page_type"] = "rest_period"

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
