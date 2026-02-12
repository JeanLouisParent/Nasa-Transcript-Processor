"""
Build final page JSON from parsed rows.
"""

import re
from pathlib import Path

from src.utils.station_normalization import match_station_name
from .cleaning import (
    split_embedded_timestamp_blocks,
    merge_duplicate_comm_timestamps,
    merge_nearby_duplicate_timestamps,
    clean_or_merge_continuations,
    merge_inline_annotations,
    merge_fragment_annotations,
    normalize_parenthesized_radio_calls,
    remove_repeated_phrases,
)
from .patterns import (
    HEADER_PAGE_ONLY_RE,
    HEADER_TAPE_ONLY_RE,
    HEADER_TAPE_SIMPLE_RE,
    HEADER_TAPE_PAGE_ONLY_RE,
    LUNAR_REV_RE,
    REST_PERIOD_RE,
    END_OF_TAPE_KEYWORD,
    GOSS_NET_RE,
)
from .utils import (
    fuzzy_find,
    is_transcription_header,
    is_goss_net_noise,
    is_not1_footer_noise,
    clean_trailing_footer,
    clean_leading_footer_noise,
    extract_header_metadata,
)


def normalize_timestamp(ts: str) -> str:
    """
    Normalize timestamp to ensure valid ranges (seconds < 60, minutes < 60, hours < 24).
    Handles timestamps like "01 06 59 60" → "01 07 00 00".
    """
    if not ts:
        return ts

    parts = ts.split()
    if len(parts) != 4:
        return ts

    try:
        d, h, m, s = [int(p) for p in parts]

        # Normalize seconds
        if s >= 60:
            m += s // 60
            s = s % 60

        # Normalize minutes
        if m >= 60:
            h += m // 60
            m = m % 60

        # Normalize hours
        if h >= 24:
            d += h // 24
            h = h % 24

        return f"{d:02d} {h:02d} {m:02d} {s:02d}"
    except (ValueError, IndexError):
        return ts


def build_page_json(
    rows: list[dict],
    lines: list[str],
    page_num: int,
    page_offset: int = 0,
    valid_speakers: list[str] | None = None,
    text_replacements: dict[str, str] | None = None,
    mission_keywords: list[str] | None = None,
    valid_locations: list[str] | None = None,
    inline_annotation_terms: list[str] | None = None,
    initial_ts: str | None = None,
    previous_block_type: str | None = None,
    lexicon_path: Path | None = None,
    footer_text_overrides: dict[int, str] | None = None,
    speaker_ocr_fixes: dict[str, str] | None = None,
    has_footer: bool = False,
) -> dict:
    """
    Build final page JSON from parsed rows with all corrections applied.
    """
    header_info = extract_header_metadata(lines, page_num, page_offset)
    header_info["footer"] = has_footer
    blocks = []

    for row in rows:
        if row["type"] == "header":
            continue
        block_type = "continuation" if row["type"] == "text" else row["type"]
        block = {"type": block_type}

        if block_type == "comm":
            if row["timestamp"]:
                block["timestamp"] = normalize_timestamp(row["timestamp"])
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
            # Apply footer text overrides for pages with corrupted footers
            display_page = page_num + 1 + page_offset
            if block_type == "footer" and footer_text_overrides and display_page in footer_text_overrides:
                block["text"] = footer_text_overrides[display_page]
            if (
                block.get("text")
                and (
                    GOSS_NET_RE.match(block["text"])
                    or is_goss_net_noise(block["text"])
                    or is_not1_footer_noise(block["text"])
                )
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

    # Split blocks that embed multiple timestamps in their text
    blocks = split_embedded_timestamp_blocks(blocks)
    # Merge consecutive comm blocks that share the same timestamp
    blocks = merge_duplicate_comm_timestamps(blocks)
    # Merge nearby duplicates within a small window
    blocks = merge_nearby_duplicate_timestamps(blocks)

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
                        block["timestamp"] = normalize_timestamp(" ".join(parts))
                    elif len(parts) == 3:
                        block["timestamp"] = normalize_timestamp(" ".join(parts + ["--"]))
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
        # Merge speakerless comm blocks starting with punctuation/lowercase into previous comm
        if (
            block.get("type") == "comm"
            and not block.get("speaker")
            and block.get("text")
            and merged_blocks
            and merged_blocks[-1].get("type") == "comm"
        ):
            text = block["text"]
            if is_not1_footer_noise(text):
                continue
            if text[:1] in ";,.)\"'" or text[:1].islower() or text.startswith("..."):
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
    blocks = clean_or_merge_continuations(blocks)
    blocks = merge_inline_annotations(blocks, inline_annotation_terms)
    blocks = merge_fragment_annotations(blocks)

    # Canonicalize REST PERIOD pages
    rest_period_found = any(
        (b.get("text") and REST_PERIOD_RE.search(b.get("text", "")))
        for b in blocks
    )
    normalized_blocks: list[dict] = []
    for block in blocks:
        text_val = block.get("text", "")
        # Strip trailing page numbers before checking (e.g. "...Page 270")
        text_check = re.sub(r"\s+Page\s+\d+\s*$", "", text_val, flags=re.IGNORECASE)
        if text_check and REST_PERIOD_RE.search(text_check):
            block["type"] = "meta"
            block["meta_type"] = "rest_period"
            block["text"] = "REST PERIOD - NO COMMUNICATIONS"
            block.pop("speaker", None)
            block.pop("location", None)
            block.pop("timestamp_correction", None)
        if text_val and is_transcription_header(text_val):
            if rest_period_found:
                continue
            block["type"] = "meta"
            block["meta_type"] = "transcript_header"
            block["text"] = "AIR-TO-GROUND VOICE TRANSCRIPTION"
        normalized_blocks.append(block)
    blocks = normalized_blocks

    if blocks and blocks[0]["type"] == "continuation" and previous_block_type in ("comm", "continuation"):
        blocks[0]["continuation_from_prev"] = True

    if rest_period_found or any(b.get("meta_type") == "rest_period" for b in blocks):
        header_info["page_type"] = "rest_period"

    # Final cleanup of footers and locations on merged text
    for block in blocks:
        if block.get("text"):
            text = block["text"]
            text = clean_trailing_footer(text)
            text = clean_leading_footer_noise(text)

            # Remove any residual location tags (from anywhere in text): "(TRANQ) ..."
            if valid_locations:
                for loc in valid_locations:
                    # Remove location tags from anywhere in the text
                    text = re.sub(rf"\(\s*{re.escape(loc)}\s*\)\s*", " ", text, flags=re.IGNORECASE)
                    text = text.strip()
            block["text"] = text.strip()

    return {"header": header_info, "blocks": blocks}
