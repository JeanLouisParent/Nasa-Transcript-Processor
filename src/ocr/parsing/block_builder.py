"""
Build final page JSON from parsed rows.
"""

import difflib
import re
from pathlib import Path

from src.correctors.speaker_corrector import SpeakerCorrector
from src.correctors.text_corrector import TextCorrector
from src.correctors.timestamp_corrector import TimestampCorrector

from .patterns import (
    HEADER_PAGE_ONLY_RE,
    HEADER_TAPE_ONLY_RE,
    HEADER_TAPE_SIMPLE_RE,
    HEADER_TAPE_PAGE_ONLY_RE,
    LUNAR_REV_RE,
    REST_PERIOD_RE,
    NO_COMM_RE,
    END_OF_TAPE_KEYWORD,
    GOSS_NET_RE,
)
from .utils import fuzzy_find, is_transcription_header, clean_trailing_footer, extract_header_metadata


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
    previous_block_type: str | None = None,
) -> dict:
    """
    Build final page JSON from parsed rows with all corrections applied.
    """
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
            if (
                block.get("text")
                and GOSS_NET_RE.match(block["text"])
                and block_type in ("continuation", "meta", "annotation")
            ):
                continue

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

    # Transform LUNAR REV blocks
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

    # Smart stitching of continuation blocks
    merged_blocks = []
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

    # Apply correctors
    ts_corrector = TimestampCorrector(initial_ts)
    blocks = ts_corrector.process_blocks(blocks)

    if valid_locations:
        for block in blocks:
            loc = block.get("location")
            if loc:
                best_loc = difflib.get_close_matches(loc.upper(), valid_locations, n=1, cutoff=0.5)
                if best_loc:
                    block["location"] = best_loc[0]

    if valid_speakers:
        blocks = SpeakerCorrector(valid_speakers).process_blocks(blocks)

    # Text correction
    lexicon_path = Path("assets/lexicon/apollo11_lexicon.json")
    if lexicon_path.exists():
        blocks = TextCorrector(lexicon_path, text_replacements, mission_keywords).process_blocks(blocks)
        blocks = [b for b in blocks if not (b.get("text") and GOSS_NET_RE.match(str(b.get("text"))))]

    return {"header": header_info, "blocks": blocks}
