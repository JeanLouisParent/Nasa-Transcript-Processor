"""
Merge per-page JSON outputs into a single global JSON and formatted TXT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import re
from pathlib import Path

from src.config.global_config import load_global_config
from src.config.mission_config import load_mission_config
from src.utils.station_normalization import match_station_name
from src.utils.merge_helpers import (
    parse_timestamp as _parse_ts,
    format_timestamp as _format_ts,
    bump_timestamp as _bump_ts,
    normalize_text as _normalize_text,
    is_missing_speaker as _is_missing_speaker,
    extract_speaker_from_text as _extract_speaker_from_text,
    starts_continuation as _starts_continuation,
    looks_truncated as _looks_truncated,
    apply_text_replacements as _apply_text_replacements,
    strip_end_of_tape_residue as _strip_end_of_tape_residue,
    clean_footer_text as _clean_footer_text,
)

# Constants
MIN_FOOTER_LENGTH = 15  # Minimum character length for a valid footer


@dataclass
class PageBundle:
    """Data container for a single page's structured content and metadata."""
    page_num: int
    header: dict
    blocks: list[dict]
    source_path: Path


CREW_SPEAKERS = {
    "CDR",
    "CMP",
    "LMP",
    "SC",
    "MS",
    "SWIM 1",
    "HORNET",
    "PAO",
    "PRESIDENT NIXON",
}


def _load_post_merge_settings(pdf_stem: str) -> dict:
    """
    Retrieves mission-specific settings required during the final merge/export phase.
    """
    global_cfg = load_global_config(Path("config/defaults.toml"))
    pdf_name = pdf_stem if pdf_stem.lower().endswith(".pdf") else f"{pdf_stem}.PDF"
    mission_cfg = load_mission_config(Path("config"), pdf_name)
    overrides = mission_cfg.layout_overrides or {}
    post_merge = overrides.get("post_merge", {})
    if not isinstance(post_merge, dict):
        post_merge = {}

    global_keywords = (
        global_cfg.pipeline_defaults.get("lexicon", {}).get("mission_keywords", [])
        if global_cfg.pipeline_defaults
        else []
    )
    mission_keywords = overrides.get("mission_keywords", [])
    keywords: list[str] = []
    for kw in list(global_keywords or []) + list(mission_keywords or []):
        if kw and kw not in keywords:
            keywords.append(kw)

    post_merge["mission_keywords"] = keywords

    # Load valid speakers from mission config
    valid_speakers = overrides.get("valid_speakers", [])
    if valid_speakers:
        post_merge["valid_speakers"] = valid_speakers

    post_merge["pdf_stem"] = pdf_stem
    return post_merge


def _is_meta_line(text: str) -> bool:
    upper = text.upper()
    return (
        upper.startswith("BEGIN LUNAR REV")
        or upper.startswith("END LUNAR REV")
        or "REST PERIOD - NO COMMUNICATIONS" in upper
    )


def _split_meta_from_comm(text: str) -> tuple[str, str | None]:
    upper = text.upper()
    markers = ["BEGIN LUNAR REV", "END LUNAR REV", "REST PERIOD - NO COMMUNICATIONS"]
    for marker in markers:
        idx = upper.find(marker)
        if idx != -1:
            before = text[:idx].rstrip(" -\t")
            after = text[idx:].strip()
            return before, after
    return text, None


def _match_station_annotation(text: str, mission_keywords: list[str]) -> list[str]:
    """Identifies tracking station annotations within text using mission keywords."""
    annotations: list[str] = []
    if not text:
        return annotations
    pattern = re.compile(
        r"\b([A-Z][A-Z0-9 ]{2,40}?)\s*\.?\s*\((REV|PASS)\s*(\d+)\)\s*",
        re.IGNORECASE,
    )
    upper_keywords = [kw.upper() for kw in mission_keywords]

    def repl(match: re.Match) -> str:
        station_raw = match.group(1).strip()
        station_raw = re.split(r"[.:;]", station_raw)[-1].strip()
        marker = match.group(2).upper()
        number = match.group(3)
        matched = match_station_name(station_raw.upper(), upper_keywords) if upper_keywords else None
        if not matched:
            return match.group(0)
        annotations.append(f"{matched} ({marker} {number})")
        return ""

    _ = re.sub(pattern, repl, text)
    return annotations


def _extract_station_annotations(text: str, mission_keywords: list[str]) -> tuple[str, list[str]]:
    """Removes station annotations from text and returns them as a separate list."""
    annotations: list[str] = []
    if not text:
        return text, annotations
    pattern = re.compile(
        r"\b([A-Z][A-Z0-9 ]{2,40}?)\s*\.?\s*\((REV|PASS)\s*(\d+)\)\s*",
        re.IGNORECASE,
    )
    upper_keywords = [kw.upper() for kw in mission_keywords]

    def repl(match: re.Match) -> str:
        station_raw = match.group(1).strip()
        station_raw = re.split(r"[.:;]", station_raw)[-1].strip()
        marker = match.group(2).upper()
        number = match.group(3)
        matched = match_station_name(station_raw.upper(), upper_keywords) if upper_keywords else None
        if not matched:
            return match.group(0)
        annotations.append(f"{matched} ({marker} {number})")
        return ""

    cleaned = re.sub(pattern, repl, text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -;,.")
    return cleaned.strip(), annotations


def _is_station_annotation_line(text: str, mission_keywords: list[str]) -> bool:
    """Returns True if the entire line consists of station annotations."""
    return bool(_match_station_annotation(text, mission_keywords))


def _should_merge_continuation(prev_text: str | None, cont_text: str, flagged: bool) -> bool:
    """Determines if a continuation fragment should be merged with the previous block."""
    if not prev_text or not cont_text:
        return False
    if flagged:
        return True
    if _starts_continuation(cont_text):
        return True
    last_char = prev_text.strip()[-1] if prev_text.strip() else ""
    if last_char in ",;:-":
        return True
    if last_char and last_char not in ".?!":
        return True
    return False


def _infer_missing_speaker(
    text: str,
    prev_comm: dict | None,
    last_crew: str | None
) -> str | None:
    """
    Attempts to predict the missing speaker based on dialogue flow and keywords.

    Args:
        text: Dialogue content.
        prev_comm: Preceding communication block.
        last_crew: Last identified crew member.

    Returns:
        Inferred speaker code or None.
    """
    # Pattern 1: "ROGER" or "Roger" at start
    if re.match(r"^(ROGER|Roger)\b", text):
        if prev_comm and prev_comm.get("speaker") in CREW_SPEAKERS:
            return "CC"
        elif prev_comm and prev_comm.get("speaker") == "CC" and last_crew:
            return last_crew
        elif "11" in text or "APOLLO" in text.upper():
            return "CC"

    # Pattern 2: "Go ahead" at start
    elif re.match(r"^(Go ahead|GO AHEAD)\b", text):
        return "CC"

    # Pattern 3: Starts with "(" and previous was CC
    elif (
        prev_comm
        and prev_comm.get("speaker") == "CC"
        and text.startswith("(")
        and len(text) > 1
        and text[1].isalpha()
        and last_crew
    ):
        return last_crew

    return None


def _extract_station_annotations_from_pages(pages: list[PageBundle], mission_keywords: list[str]) -> None:
    """Extract station annotations from comm block text and create separate annotation blocks."""
    if not mission_keywords:
        return

    for page in pages:
        extracted: list[dict] = []
        for block in page.blocks:
            if block.get("type") != "comm" or not block.get("text"):
                extracted.append(block)
                continue

            text, annotations = _extract_station_annotations(block.get("text", ""), mission_keywords)
            if annotations:
                if text:
                    updated = dict(block)
                    updated["text"] = text
                    extracted.append(updated)
                for ann in annotations:
                    extracted.append({"type": "annotation", "text": ann})
            else:
                extracted.append(block)
        page.blocks = extracted


def _deduplicate_adjacent_annotations(pages: list[PageBundle]) -> None:
    """Remove duplicate adjacent annotation blocks."""
    for page in pages:
        deduped: list[dict] = []
        last_ann_norm: str | None = None
        for block in page.blocks:
            if block.get("type") == "annotation" and block.get("text"):
                norm = _normalize_text(block.get("text", ""))
                if last_ann_norm == norm:
                    continue
                last_ann_norm = norm
            else:
                last_ann_norm = None
            deduped.append(block)
        page.blocks = deduped


def _cleanup_pages(pages: list[PageBundle], post_merge: dict | None = None) -> None:
    """
    Fix common OCR artifacts in merged exports:
    - drop empty comm blocks
    - merge speakerless continuations into previous comm (including across pages)
    - infer speakers for simple call-and-response cases
    - convert LUNAR REV / REST PERIOD lines into meta blocks
    """
    post_merge = post_merge or {}
    mission_keywords = post_merge.get("mission_keywords", [])
    text_replacements = post_merge.get("text_replacements", {}) or {}
    valid_speakers = post_merge.get("valid_speakers", [])

    # Phase 1: Clean and normalize blocks
    _clean_and_normalize_blocks(pages, text_replacements, valid_speakers)

    # Phase 2: Extract station annotations
    _extract_station_annotations_from_pages(pages, mission_keywords)

    # Phase 3: Deduplicate adjacent annotations
    _deduplicate_adjacent_annotations(pages)

    # Phase 4: Fix timestamp monotonicity
    _fix_timestamp_monotonicity(pages)

    # Phase 5: Merge duplicate timestamps
    _merge_duplicate_timestamps(pages)


def _clean_and_normalize_blocks(
    pages: list[PageBundle],
    text_replacements: dict,
    valid_speakers: list[str],
) -> None:
    """
    Cleans blocks by removing empty ones, applying text replacements,
    extracting/inferring speakers, and converting meta lines.
    """
    prev_comm: dict | None = None
    last_crew: str | None = None

    for page in pages:
        cleaned: list[dict] = []

        for block in page.blocks:
            btype = block.get("type")

            # Handle footer blocks
            if btype == "footer":
                text = _clean_footer_text(block.get("text", ""))
                if text:
                    cleaned.append({"type": "footer", "text": text})
                continue

            # Handle annotation blocks
            if btype == "annotation":
                text = (block.get("text", "") or "")
                cleaned.append({"type": "annotation", "text": text} if text else block)
                continue

            # Handle continuation blocks
            if btype == "continuation":
                text = (block.get("text") or "").strip()
                if text:
                    text = _apply_text_replacements(text, text_replacements)
                    updated = dict(block)
                    updated["text"] = text
                    cleaned.append(updated)
                continue

            # Only process comm blocks beyond this point
            if btype != "comm":
                cleaned.append(block)
                continue

            # Process comm block
            processed_block = _process_comm_block(
                block, text_replacements, valid_speakers, cleaned, prev_comm, last_crew
            )

            if processed_block is None:
                # Block was merged into previous or dropped
                continue

            cleaned.append(processed_block)

            # Update context for next iteration
            if processed_block.get("type") == "comm" and not _is_missing_speaker(processed_block.get("speaker")):
                prev_comm = processed_block
                if processed_block.get("speaker") in CREW_SPEAKERS:
                    last_crew = processed_block.get("speaker")

        page.blocks = cleaned


def _process_comm_block(
    block: dict,
    text_replacements: dict,
    valid_speakers: list[str],
    cleaned: list[dict],
    prev_comm: dict | None,
    last_crew: str | None,
) -> dict | None:
    """
    Processes a single comm block: cleans text, extracts speaker, infers missing speaker.

    Returns:
        Updated block dict, or None if block should be dropped/merged.
    """
    text = (block.get("text") or "").strip()
    speaker = block.get("speaker")

    # Apply text transformations
    if text:
        text = _apply_text_replacements(text, text_replacements)
        text = _strip_end_of_tape_residue(text)

    # Drop empty blocks
    if not text:
        return None

    # Extract speaker from text if available
    if valid_speakers:
        extracted_speaker, cleaned_text = _extract_speaker_from_text(text, valid_speakers)
        if extracted_speaker:
            if not speaker:
                # Speaker missing - extract it
                speaker = extracted_speaker
                text = cleaned_text
            elif speaker == extracted_speaker:
                # Speaker redundant - clean it
                text = cleaned_text

    # Convert meta lines (LUNAR REV, REST PERIOD) to meta blocks
    if _is_meta_line(text) or any(
        m in text.upper()
        for m in ("BEGIN LUNAR REV", "END LUNAR REV", "REST PERIOD - NO COMMUNICATIONS")
    ):
        before, meta_text = _split_meta_from_comm(text)
        if meta_text:
            meta_type = "rest_period" if "REST PERIOD - NO COMMUNICATIONS" in meta_text.upper() else "lunar_rev"
            cleaned.append({"type": "meta", "text": meta_text, "meta_type": meta_type})
        if not before:
            return None
        text = before

    missing = _is_missing_speaker(speaker)

    # Merge continuation text into previous block
    if missing and _starts_continuation(text):
        target = cleaned[-1] if cleaned and cleaned[-1].get("type") == "comm" else prev_comm
        if target is not None:
            target["text"] = (target.get("text", "") + " " + text).strip()
            return None

    # Infer missing speaker from context
    if missing:
        inferred = _infer_missing_speaker(text, prev_comm, last_crew)
        if inferred:
            speaker = inferred

    # Merge digit-starting text with previous CC block
    if missing and prev_comm and prev_comm.get("speaker") == "CC" and text[0].isdigit():
        prev_comm["text"] = (prev_comm.get("text", "") + " " + text).strip()
        return None

    # Build updated block
    updated = dict(block)
    updated["text"] = text
    if speaker is not None and speaker != block.get("speaker"):
        updated["speaker"] = speaker

    return updated


def _fix_timestamp_monotonicity(pages: list[PageBundle]) -> None:
    """
    Ensures timestamps are monotonically increasing across all pages.
    Corrects OCR errors in day/hour/minute components.
    """
    last_ts: tuple[int, int, int, int] | None = None

    for page in pages:
        for idx, block in enumerate(page.blocks):
            if block.get("type") != "comm":
                continue

            ts = block.get("timestamp")
            if not ts:
                continue

            parsed = _parse_ts(ts)
            if parsed is None:
                continue

            # Check if timestamp is non-monotonic
            if last_ts is not None and parsed <= last_ts:
                candidate = _correct_non_monotonic_timestamp(parsed, last_ts)

                # Update block if correction was made
                if candidate != parsed:
                    updated = dict(block)
                    updated["timestamp"] = _format_ts(candidate)
                    page.blocks[idx] = updated
                    parsed = candidate

            last_ts = parsed


def _correct_non_monotonic_timestamp(
    parsed: tuple[int, int, int, int],
    last_ts: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Attempts to correct a non-monotonic timestamp by fixing OCR errors.

    Args:
        parsed: Current timestamp (dd, hh, mm, ss)
        last_ts: Previous valid timestamp

    Returns:
        Corrected timestamp tuple
    """
    candidate = parsed

    # Fix day if it looks wrong
    if parsed[0] == 0 and last_ts[0] > 0:
        candidate = (last_ts[0], parsed[1], parsed[2], parsed[3])
    elif parsed[0] < last_ts[0]:
        candidate = (last_ts[0], parsed[1], parsed[2], parsed[3])

    # Fix hour if it looks wrong (e.g., 05 -> 00)
    if candidate[1] < last_ts[1] and candidate[0] == last_ts[0]:
        candidate = (candidate[0], last_ts[1], candidate[2], candidate[3])

    # Fix minute if it looks wrong
    if candidate[2] < last_ts[2] and candidate[:2] == last_ts[:2]:
        candidate = (candidate[0], candidate[1], last_ts[2], candidate[3])

    # If still not monotonic, just bump from last
    if candidate <= last_ts:
        candidate = _bump_ts(last_ts)

    return candidate


def _merge_duplicate_timestamps(pages: list[PageBundle]) -> None:
    """
    Merges consecutive comm blocks with identical timestamp and speaker.
    Concatenates text and preserves location if present.
    """
    for page in pages:
        merged: list[dict] = []

        for block in page.blocks:
            # Check if we can merge with previous block
            if (
                block.get("type") == "comm"
                and merged
                and merged[-1].get("type") == "comm"
                and block.get("timestamp")
                and block.get("timestamp") == merged[-1].get("timestamp")
                and block.get("speaker")
                and block.get("speaker") == merged[-1].get("speaker")
            ):
                # Merge text
                prev_text = (merged[-1].get("text") or "").strip()
                curr_text = (block.get("text") or "").strip()
                if curr_text:
                    merged[-1]["text"] = f"{prev_text} {curr_text}".strip() if prev_text else curr_text

                # Preserve location if present
                if not merged[-1].get("location") and block.get("location"):
                    merged[-1]["location"] = block.get("location")

                continue

            merged.append(block)

        page.blocks = merged


def collect_page_jsons(output_dir: Path, pdf_stem: str) -> list[PageBundle]:
    """
    Crawls the output directory to find and load all per-page JSON files.

    Args:
        output_dir: Root directory where mission outputs are stored.
        pdf_stem: The identifier of the mission (PDF filename without extension).

    Returns:
        A sorted list of PageBundle objects representing all valid pages.
    """
    base_dir = output_dir / pdf_stem
    bundles: list[PageBundle] = []
    if not base_dir.exists():
        return bundles
    pages_root = base_dir / "pages"
    if not pages_root.exists():
        return bundles
    for page_dir in sorted(pages_root.glob("Page_*")):
        if not page_dir.is_dir():
            continue
        json_files = list(page_dir.glob(f"{pdf_stem}_page_*.json"))
        if not json_files:
            continue
        json_path = json_files[0]
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        header = data.get("header", {})
        blocks = data.get("blocks", [])
        try:
            page_num = int(header.get("page", 0))
        except (TypeError, ValueError):
            page_num = 0
        if isinstance(page_num, int) and page_num <= 0:
            continue
        bundles.append(PageBundle(page_num=page_num, header=header, blocks=blocks, source_path=json_path))
    bundles.sort(key=lambda b: (b.page_num, b.source_path.name))
    return bundles


def build_global_json(pdf_stem: str, pages: list[PageBundle]) -> dict:
    """
    Aggregates individual page data into a single master JSON structure.

    Args:
        pdf_stem: Mission identifier.
        pages: List of loaded page bundles.

    Returns:
        A dictionary containing the full document hierarchy.
    """
    post_merge = _load_post_merge_settings(pdf_stem)
    _cleanup_pages(pages, post_merge)
    page_list: list[dict] = []
    for page in pages:
        header = dict(page.header or {})
        # Tape numbers are already validated and correct in per-page JSONs
        # from main.py process/reparse, so just use them as-is

        cleaned_blocks: list[dict] = []
        for block in page.blocks:
            if not isinstance(block, dict):
                cleaned_blocks.append(block)
                continue
            
            # Define fixed key order for a professional look
            btype = block.get("type")
            ordered = {"type": btype}
            
            # Add fields in logical order based on block type
            if btype == "comm":
                if "timestamp" in block: ordered["timestamp"] = block["timestamp"]
                if "speaker" in block: ordered["speaker"] = block["speaker"]
                if "location" in block: ordered["location"] = block["location"]
            elif btype == "meta":
                if "meta_type" in block: ordered["meta_type"] = block["meta_type"]
                if "timestamp" in block: ordered["timestamp"] = block["timestamp"]
            
            # Text always comes last or near-last
            if "text" in block: ordered["text"] = block["text"]
            
            # Append any remaining fields while stripping debug/internal flags
            for k, v in block.items():
                if k not in ordered and k not in (
                    "timestamp_correction", "timestamp_warning", 
                    "timestamp_suffix_hint", "_column_fill"
                ):
                    ordered[k] = v
            
            cleaned_blocks.append(ordered)

        page_list.append({
            "header": header,
            "blocks": cleaned_blocks,
            "source": str(page.source_path)
        })

    return {
        "document": pdf_stem,
        "page_count": len(page_list),
        "pages": page_list,
    }


def render_transcript_text(pages: list[PageBundle]) -> str:
    """
    Generates a human-readable plain text version of the entire transcript.

    Args:
        pages: List of page bundles.

    Returns:
        A formatted multi-line string.
    """
    lines: list[str] = []
    ts_width = 11  # "00 00 00 00"
    speaker_width = 5
    text_column = ts_width + 2 + speaker_width + 2
    for page in pages:
        header = page.header or {}
        page_label = header.get("page") or page.page_num or "?"
        tape = header.get("tape") or ""
        tape_str = f" | TAPE {tape}" if tape else ""
        lines.append(f"==== PAGE {page_label}{tape_str} ====")
        for block in page.blocks:
            btype = block.get("type")
            text = (block.get("text") or "").strip()
            if btype == "comm":
                ts = block.get("timestamp", "")
                speaker = block.get("speaker", "")
                location = block.get("location", "")
                loc_part = f"({location}) " if location else ""
                prefix = f"{ts:<{ts_width}}  {speaker:<{speaker_width}}  {loc_part}"
                if text:
                    lines.append(prefix + text)
                else:
                    lines.append(prefix.rstrip())
            elif btype == "continuation":
                if text:
                    lines.append(" " * text_column + text)
            elif btype in ("meta", "annotation", "footer"):
                if text:
                    lines.append(text)
            else:
                if text:
                    lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_transcript_markdown(pages: list[PageBundle], pdf_stem: str) -> str:
    """
    Generates a Markdown version of the transcript with tables for dialogue.

    Args:
        pages: List of page bundles.
        pdf_stem: Mission identifier.

    Returns:
        A Markdown-formatted string.
    """
    lines: list[str] = []

    # Title
    lines.append(f"# {pdf_stem} Transcript\n")
    lines.append("---\n")

    for page in pages:
        header = page.header or {}
        page_label = header.get("page") or page.page_num or "?"
        tape = header.get("tape") or ""
        page_type = header.get("page_type")

        # Page header
        tape_str = f" · Tape {tape}" if tape else ""
        lines.append(f"## Page {page_label}{tape_str}\n")

        if page_type == "rest_period":
            lines.append("_Rest period - No communications_\n")
            continue

        # Collect comm blocks for table
        comm_blocks = []
        meta_blocks = []

        for block in page.blocks:
            btype = block.get("type")
            if btype == "comm":
                comm_blocks.append(block)
            elif btype in ("meta", "annotation", "header", "footer"):
                meta_blocks.append(block)

        # Render meta blocks first
        for block in meta_blocks:
            text = (block.get("text") or "").strip()
            if text:
                meta_type = block.get("meta_type")
                if meta_type == "end_of_tape":
                    lines.append(f"**{text}**\n")
                elif meta_type:
                    lines.append(f"_{text}_\n")
                else:
                    lines.append(f"> {text}\n")

        # Render comm table
        if comm_blocks:
            lines.append("| Time | Speaker | Location | Dialogue |")
            lines.append("|:-----|:--------|:---------|:---------|")

            for block in comm_blocks:
                ts = block.get("timestamp", "").replace(" ", ":") or "—"
                speaker = block.get("speaker", "") or "—"
                location = block.get("location", "") or ""
                text = (block.get("text") or "").strip()

                # Escape pipe characters in text
                text = text.replace("|", "\\|")

                loc_cell = location if location else ""
                lines.append(f"| {ts} | {speaker} | {loc_cell} | {text} |")

            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


def write_global_outputs(output_dir: Path, pdf_stem: str) -> Path:
    """
    Orchestrates the creation of all final documents (JSON, Text, etc.).

    Args:
        output_dir: Destination root directory.
        pdf_stem: Mission identifier.

    Returns:
        Path to the generated global JSON file.
    """
    pages = collect_page_jsons(output_dir, pdf_stem)
    global_json = build_global_json(pdf_stem, pages)
    output_root = output_dir / pdf_stem
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / f"{pdf_stem}_merged.json"
    json_path.write_text(json.dumps(global_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return json_path
