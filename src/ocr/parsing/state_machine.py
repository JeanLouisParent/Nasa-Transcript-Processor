"""
State machine for parsing OCR text into structured rows.
"""

import re
from .patterns import (
    TIMESTAMP_STRICT_RE,
    TIMESTAMP_PREFIX_RE,
    SPEAKER_TOKEN_RE,
    SPEAKER_LINE_RE,
    SPEAKER_PAREN_RE,
    LOCATION_PAREN_RE,
    HEADER_PAGE_RE,
    HEADER_TAPE_RE,
    HEADER_PAGE_ONLY_RE,
    HEADER_TAPE_ONLY_RE,
    HEADER_TAPE_PAGE_ONLY_RE,
    HEADER_KEYWORDS,
    END_OF_TAPE_KEYWORD,
    TRANSITION_KEYWORDS,
)
from .utils import fuzzy_find
from .preprocessor import preprocess_lines


def parse_ocr_text(text: str, page_num: int, mission_keywords: list[str] | None = None) -> list[dict]:
    """
    Parse plain OCR output into structured rows.
    """
    lines = preprocess_lines(text, mission_keywords)
    if not lines:
        return []

    # Find first timestamp to identify header zone
    first_ts_idx = next(
        (
            i
            for i, entry in enumerate(lines)
            if TIMESTAMP_STRICT_RE.match(entry["text"]) or TIMESTAMP_PREFIX_RE.match(entry["text"])
        ),
        None,
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
            "page": page_num + 1,
            "line": line_index,
            "type": "comm" if (pending_ts or pending_force_comm) else "text",
            "timestamp": pending_ts,
            "speaker": pending_speaker,
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
            rows.append(
                {
                    "page": page_num + 1,
                    "line": line_index,
                    "type": "footer",
                    "timestamp": "",
                    "speaker": "",
                    "location": "",
                    "text": line,
                }
            )
            prev_comm_like = False
            continue

        if forced_type is None and fuzzy_find(line, END_OF_TAPE_KEYWORD):
            flush_pending()
            line_index += 1
            rows.append(
                {
                    "page": page_num + 1,
                    "line": line_index,
                    "type": "meta",
                    "timestamp": "",
                    "speaker": "",
                    "location": "",
                    "text": line,
                }
            )
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
            rows.append(
                {
                    "page": page_num + 1,
                    "line": line_index,
                    "type": "meta" if forced_type == "meta" else forced_type,
                    "timestamp": "",
                    "speaker": "",
                    "location": "",
                    "text": line,
                }
            )
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
            remainder = line[len(pending_ts) :].strip()
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
                        pending_speaker = pending_speaker[: loc_match.start()].strip()
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

        if (
            not pending_ts
            and not timestamp_list_mode
            and (SPEAKER_LINE_RE.match(line) or LOCATION_PAREN_RE.match(line))
        ):
            continue
        if (
            timestamp_list_mode
            and not pending_ts
            and forced_type is None
            and timestamp_list_row_idx is not None
        ):
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
            is_header = (
                not pending_ts
                and first_ts_idx is not None
                and idx <= first_ts_idx
                and any(fuzzy_find(line, kw) for kw in HEADER_KEYWORDS)
            )
            is_footer = not pending_ts and (
                "***" in line
                or "ASTERISK" in upper
                or (
                    idx >= len(lines) - 3
                    and (HEADER_PAGE_RE.search(upper) or HEADER_TAPE_RE.search(upper))
                )
            )
            is_annotation = (
                "(REV" in upper
                or "(RFV" in upper
                or (
                    mission_keywords
                    and not pending_ts
                    and any(fuzzy_find(line, kw) for kw in mission_keywords)
                )
            )
            is_end_of_tape = fuzzy_find(line, END_OF_TAPE_KEYWORD)
            is_transition = any(fuzzy_find(line, kw) for kw in TRANSITION_KEYWORDS)

            if is_header or is_footer or is_annotation or is_end_of_tape or is_transition:
                flush_pending()
                line_index += 1
                rows.append(
                    {
                        "page": page_num + 1,
                        "line": line_index,
                        "type": "meta"
                        if (is_end_of_tape or is_transition)
                        else ("header" if is_header else ("footer" if is_footer else "annotation")),
                        "timestamp": "",
                        "speaker": "",
                        "location": "",
                        "text": line,
                    }
                )
                continue

        if pending_ts:
            # Check for location tag at the start of the line
            loc_at_start_match = re.match(r"^\s*\(([A-Z0-9\s]+)\)\s*", line)
            if loc_at_start_match:
                pending_location = loc_at_start_match.group(1).strip()
                line = line[loc_at_start_match.end() :].strip()
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

    # Timestamp list mode post-processing
    if timestamp_list_mode:
        ts_row_indices = [i for i, row in enumerate(rows) if row.get("timestamp")]
        start_idx = timestamp_list_start_idx
        if start_idx is None and ts_row_indices:
            start_idx = ts_row_indices[0]
        if start_idx is not None and ts_row_indices:
            has_speaker = any(rows[i].get("speaker") for i in ts_row_indices)
            if not has_speaker:
                ts_line_indices = [
                    i
                    for i, entry in enumerate(lines)
                    if TIMESTAMP_STRICT_RE.match(entry["text"]) or TIMESTAMP_PREFIX_RE.match(entry["text"])
                ]
                if ts_line_indices:
                    row_idx = start_idx
                    for i in range(start_idx, len(rows)):
                        rows[i]["speaker"] = ""
                        rows[i]["location"] = ""
                        rows[i]["text"] = ""
                    for entry in lines[ts_line_indices[-1] + 1 :]:
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
