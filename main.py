#!/usr/bin/env python3
"""
NASA Transcript Processing Pipeline - CLI Interface.

Usage:
    python main.py process AS11_TEC.PDF
    python main.py process AS11_TEC.PDF --pages 1-50
    python main.py process AS11_TEC.PDF --no-ocr
    python main.py info AS11_TEC.PDF
"""

import json
import re
import shutil
import sys
import time
from pathlib import Path

import click
import cv2
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.config.global_config import GlobalConfig, load_global_config, load_prompt_config
from src.config.mission_config import load_mission_config
from src.core.pipeline import PageResult, TranscriptPipeline
from src.core.post_processing import PostProcessor
from src.correctors.timestamp_index import GlobalTimestampIndex
from src.ocr.ocr_client import (
    COLUMN_OCR_PROMPT,
    PLAIN_OCR_PROMPT,
    TEXT_COLUMN_OCR_PROMPT,
    LMStudioOCRClient,
)
from src.ocr.ocr_parser import build_page_json, parse_ocr_text
from src.ocr.parsing.merger import merge_payloads
from src.ocr.parsing.patterns import (
    HEADER_PAGE_ONLY_RE,
    HEADER_TAPE_ONLY_RE,
    HEADER_TAPE_PAGE_ONLY_RE,
    LOCATION_PAREN_RE,
    SPEAKER_LINE_RE,
)
from src.processors.page_extractor import get_pdf_info, PageExtractor
from src.processors.image_processor import ImageProcessor
from src.utils.console import PipelineConsole
from src.utils.console import console as rich_console
from src.utils.merge_export import write_global_outputs
from src.utils.tape_validator import validate_and_correct_tape, format_tape

# Initialize console manager
console = PipelineConsole()

def setup_logging(verbose: bool = False) -> None:
    """
    Configures the logger to write to a rotating file.
    
    Logging to stdout is disabled to prevent interference with the 
    Rich Live console dashboard.

    Args:
        verbose: If True, sets log level to DEBUG; otherwise INFO.
    """
    logger.remove()
    # Log strictly to file to avoid breaking the Rich Live dashboard
    logger.add("pipeline.log", rotation="10 MB", level="DEBUG" if verbose else "INFO")


def resolve_pdf_path(pdf_arg: str, input_dir: Path) -> Path:
    """
    Resolves a PDF filename or path against the configured input directory.

    Args:
        pdf_arg: Filename or full path provided by the user.
        input_dir: The directory to search if a relative name is provided.

    Returns:
        A Path object pointing to the source PDF.
    """
    path = Path(pdf_arg)
    if path.exists():
        return path
    name = pdf_arg if path.suffix else f"{pdf_arg}.pdf"
    return input_dir / name


def parse_pages(page_spec: str, max_pages: int) -> list[int]:
    """
    Converts a user-provided page range string into a list of indices.

    Supports comma-separated values and ranges (e.g., "1-50,60,70-80").

    Args:
        page_spec: The raw range string from CLI.
        max_pages: Total page count of the document for bounds checking.

    Returns:
        Sorted list of zero-indexed page numbers.
    """
    if not page_spec:
        return list(range(max_pages))

    pages = set()
    for token in page_spec.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start = int(parts[0]) - 1
                end = int(parts[1])
            except ValueError:
                raise click.BadParameter(f"Invalid range: {token}") from None
            if start < 0:
                raise click.BadParameter("Page numbers must be >= 1")
            pages.update(range(start, min(end, max_pages)))
        else:
            try:
                page = int(token) - 1
            except ValueError:
                raise click.BadParameter(f"Invalid page: {token}") from None
            if page < 0:
                raise click.BadParameter("Page numbers must be >= 1")
            if page < max_pages:
                pages.add(page)

    return sorted(pages)


def run_ocr_pipeline(
    pdf_path: Path,
    config: GlobalConfig,
    page_results: list[PageResult],
    page_offset: int = 0,
    valid_speakers: list[str] | None = None,
    text_replacements: dict[str, str] | None = None,
    mission_keywords: list[str] | None = None,
    valid_locations: list[str] | None = None,
    mission_overrides: dict[str, object] | None = None,
    lexicon_path: Path | None = None,
) -> tuple[int, dict[int, dict[str, float]]]:
    """
    Executes the OCR and intelligence stage of the pipeline.

    This function handles:
    1. Multi-pass OCR (Plain, Raw Fallback, Faint Enhancement, Column Fill).
    2. Payload merging and structural assembly.
    3. Post-processing cleanup and corrections.
    4. Tape number and metadata validation.
    5. Progress tracking and timing metrics.

    Args:
        pdf_path: Source PDF file.
        config: Global application configuration.
        page_results: Image processing results from the first stage.
        page_offset: Mission-specific page numbering offset.
        valid_speakers: List of allowed callsigns.
        text_replacements: Mission-specific regex fixes.
        mission_keywords: Terms for disambiguation.
        valid_locations: List of allowed location codes.
        mission_overrides: Raw mission configuration dictionary.
        lexicon_path: Path to the vocabulary for spell checking.

    Returns:
        Tuple of (total_failures_count, per_page_timing_dict).
    """
    prompts_cfg = load_prompt_config(Path("config/prompts.toml"))
    plain_prompt = prompts_cfg.get("plain_ocr_prompt", PLAIN_OCR_PROMPT)
    column_prompt = prompts_cfg.get("column_ocr_prompt", COLUMN_OCR_PROMPT)
    text_column_prompt = prompts_cfg.get("text_column_prompt", TEXT_COLUMN_OCR_PROMPT)

    prompt = plain_prompt
    if config.ocr_prompt == "plain":
        prompt = plain_prompt
    elif config.ocr_prompt == "column":
        prompt = column_prompt

    client = LMStudioOCRClient(
        base_url=config.ocr_url,
        model=config.ocr_model,
        timeout_s=120,
        max_tokens=4096,
        prompt=prompt,
    )

    failures = 0
    timings: dict[int, dict[str, float]] = {}
    total_to_ocr = sum(1 for pr in page_results if pr.success)
    console.start_ocr(total_to_ocr)

    # Load Global Timestamp Index
    index_path = config.state_dir / f"{pdf_path.stem}_timestamps_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    ts_index = GlobalTimestampIndex.load(index_path)

    if mission_overrides is None:
        mission_overrides = {}

    # Get invalid location annotations from config
    global_correctors = config.pipeline_defaults.get("correctors", {})
    invalid_location_annotations = global_correctors.get("invalid_location_annotations", {}).get("annotations", [])
    raw_footer_overrides = mission_overrides.get("footer_text_overrides", {})
    footer_text_overrides = {int(k): v for k, v in raw_footer_overrides.items()} if raw_footer_overrides else None

    global_speaker_ocr_fixes = global_correctors.get("speaker_ocr_fixes", {})
    mission_speaker_ocr_fixes = mission_overrides.get("speaker_ocr_fixes", {})
    speaker_ocr_fixes = {**global_speaker_ocr_fixes, **mission_speaker_ocr_fixes}

    ocr_text_column_pass = bool(
        mission_overrides.get("ocr_text_column_pass")
        or config.pipeline_defaults.get("ocr_text_column_pass")
        or False
    )
    ocr_dual_pass = bool(
        mission_overrides.get("ocr_dual_pass")
        or config.pipeline_defaults.get("ocr_dual_pass")
        or False
    )
    ocr_faint_pass = bool(
        mission_overrides.get("ocr_faint_pass")
        or config.pipeline_defaults.get("ocr_faint_pass")
        or False
    )
    col2_end_val = (
        mission_overrides.get("col2_end")
        or config.pipeline_defaults.get("col2_end")
        or 0.30
    )
    col2_end = float(str(col2_end_val)) if col2_end_val else 0.30

    # Initialize components
    pipeline_config = config
    image_processor = ImageProcessor(pipeline_config)
    page_extractor = PageExtractor(pdf_path, pipeline_config)

    post_processor = PostProcessor(
        valid_speakers=valid_speakers,
        valid_locations=valid_locations,
        mission_keywords=mission_keywords,
        text_replacements=text_replacements,
        speaker_ocr_fixes=speaker_ocr_fixes,
        invalid_location_annotations=invalid_location_annotations,
        manual_speaker_corrections=mission_overrides.get("manual_speaker_corrections", {}),
        lexicon_path=lexicon_path,
    )

    previous_block_type = None
    tape_x = 1
    tape_y = 1
    tape_started = False
    prev_has_end_of_tape = False

    for pr in page_results:
        if not pr.success or not pr.output:
            continue

        page_num = pr.page_num
        page_start = time.perf_counter()
        page_dir = pr.output.page_dir
        ocr_dir = pr.output.ocr_dir
        page_id = f"{pdf_path.stem}_page_{page_num + 1:04d}"

        try:
            stage_t = {}
            # Load enhanced image from disk
            t0 = time.perf_counter()
            enhanced_path = pr.output.enhanced_image
            enhanced = cv2.imread(str(enhanced_path))
            if enhanced is None:
                raise FileNotFoundError(f"Enhanced image not found: {enhanced_path}")
            stage_t["load_enhanced"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            raw_text = client.ocr_image(enhanced)
            stage_t["ocr_plain"] = time.perf_counter() - t0
            # Preserve raw OCR response for diagnostics.
            raw_text_output = raw_text
            if raw_text_output and not raw_text_output.endswith("\n"):
                raw_text_output += "\n"
            # Normalize newlines for parsing.
            text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

            t0 = time.perf_counter()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            rows, has_footer = parse_ocr_text(text, page_num, mission_keywords, valid_speakers)
            stage_t["parse_plain"] = time.perf_counter() - t0

            # Get last timestamp from previous pages
            initial_ts = ts_index.get_last_timestamp_before(page_num)

            t0 = time.perf_counter()
            payload = build_page_json(
                rows, lines, page_num, page_offset,
                valid_speakers, text_replacements,
                mission_keywords, valid_locations,
                inline_annotation_terms=invalid_location_annotations,
                initial_ts=initial_ts,
                previous_block_type=previous_block_type,
                lexicon_path=lexicon_path,
                footer_text_overrides=footer_text_overrides,
                speaker_ocr_fixes=speaker_ocr_fixes,
                has_footer=has_footer
            )

            # Run Post-Processing
            payload["blocks"] = post_processor.process_blocks(payload.get("blocks", []), initial_ts)

            stage_t["build_plain"] = time.perf_counter() - t0

            raw_image = None
            if ocr_dual_pass or ocr_faint_pass:
                t0 = time.perf_counter()
                # Use PageExtractor for consistent rendering
                raw_image = page_extractor.extract_page_image(page_num)
                raw_image_path = pr.output.assets_dir / f"{page_id}_raw.png"
                cv2.imwrite(str(raw_image_path), raw_image)
                stage_t["render_raw"] = time.perf_counter() - t0

            if ocr_dual_pass:
                t0 = time.perf_counter()
                raw_text_fallback = client.ocr_image(raw_image)
                stage_t["ocr_raw"] = time.perf_counter() - t0
                raw_text_fallback_output = raw_text_fallback
                if raw_text_fallback_output and not raw_text_fallback_output.endswith("\n"):
                    raw_text_fallback_output += "\n"
                raw_text_fallback = raw_text_fallback.replace("\r\n", "\n").replace("\r", "\n")
                t0 = time.perf_counter()
                fallback_lines = [line.strip() for line in raw_text_fallback.splitlines() if line.strip()]
                fallback_rows, fallback_has_footer = parse_ocr_text(raw_text_fallback, page_num, mission_keywords, valid_speakers)
                stage_t["parse_raw"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                fallback_payload = build_page_json(
                    fallback_rows, fallback_lines, page_num, page_offset,
                    valid_speakers, text_replacements,
                    mission_keywords, valid_locations,
                    inline_annotation_terms=invalid_location_annotations,
                    initial_ts=initial_ts,
                    previous_block_type=previous_block_type,
                    lexicon_path=lexicon_path,
                    speaker_ocr_fixes=speaker_ocr_fixes,
                    has_footer=fallback_has_footer
                )
                fallback_payload["blocks"] = post_processor.process_blocks(fallback_payload.get("blocks", []), initial_ts)
                stage_t["build_raw"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                payload = merge_payloads(payload, fallback_payload)
                stage_t["merge_raw"] = time.perf_counter() - t0
                (ocr_dir / f"{page_id}_ocr_raw_fallback.txt").write_text(
                    raw_text_fallback_output, encoding="utf-8"
                )
            if ocr_faint_pass:
                t0 = time.perf_counter()
                # Use ImageProcessor for faint enhancement
                faint_image = image_processor.enhance_contrast_heavy(raw_image)
                faint_image_path = pr.output.assets_dir / f"{page_id}_faint.png"
                cv2.imwrite(str(faint_image_path), faint_image)
                stage_t["enhance_faint"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                faint_text = client.ocr_image(faint_image)
                stage_t["ocr_faint"] = time.perf_counter() - t0
                faint_text_output = faint_text
                if faint_text_output and not faint_text_output.endswith("\n"):
                    faint_text_output += "\n"
                faint_text = faint_text.replace("\r\n", "\n").replace("\r", "\n")
                t0 = time.perf_counter()
                faint_lines = [line.strip() for line in faint_text.splitlines() if line.strip()]
                faint_rows, faint_has_footer = parse_ocr_text(faint_text, page_num, mission_keywords, valid_speakers)
                stage_t["parse_faint"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                faint_payload = build_page_json(
                    faint_rows, faint_lines, page_num, page_offset,
                    valid_speakers, text_replacements,
                    mission_keywords, valid_locations,
                    inline_annotation_terms=invalid_location_annotations,
                    initial_ts=initial_ts,
                    previous_block_type=previous_block_type,
                    lexicon_path=lexicon_path,
                    speaker_ocr_fixes=speaker_ocr_fixes,
                    has_footer=faint_has_footer
                )
                faint_payload["blocks"] = post_processor.process_blocks(faint_payload.get("blocks", []), initial_ts)
                stage_t["build_faint"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                payload = merge_payloads(payload, faint_payload)
                stage_t["merge_faint"] = time.perf_counter() - t0
                (ocr_dir / f"{page_id}_ocr_faint_fallback.txt").write_text(
                    faint_text_output, encoding="utf-8"
                )

            if ocr_text_column_pass:
                missing_text_blocks: list[dict] = [
                    b for b in payload.get("blocks", [])
                    if b.get("type") == "comm" and not b.get("text")
                ]
                column_lines: list[str] = []
                if missing_text_blocks:
                    h, w = enhanced.shape[:2]
                    start_x = min(w - 1, max(0, int(w * col2_end) + 5))
                    column_crop = enhanced[:, start_x:w]
                    t0 = time.perf_counter()
                    column_text = client.ocr_image_with_prompt(column_crop, text_column_prompt)
                    stage_t["ocr_textcol"] = stage_t.get("ocr_textcol", 0.0) + (time.perf_counter() - t0)
                    column_text = column_text.replace("\r\n", "\n").replace("\r", "\n")
                    column_lines_raw = column_text.splitlines()
                    column_text = "\n".join(column_lines_raw)
                    raw_lines = [line.strip() for line in column_lines_raw if line.strip()]
                    column_lines = [
                        line for line in raw_lines
                        if not SPEAKER_LINE_RE.match(line)
                        and not LOCATION_PAREN_RE.match(line)
                        and not HEADER_PAGE_ONLY_RE.match(line)
                        and not HEADER_TAPE_ONLY_RE.match(line)
                        and not HEADER_TAPE_PAGE_ONLY_RE.match(line)
                    ]
                    if column_lines:
                        for block, line in zip(missing_text_blocks, column_lines, strict=False):
                            block["text"] = line
                            block["_column_fill"] = True
                        (ocr_dir / f"{page_id}_ocr_textcol.txt").write_text(
                            column_text + "\n", encoding="utf-8"
                        )
                # Merge adjacent column_ocr continuations
                merged_blocks = []
                for block in payload.get("blocks", []):
                    if (
                        block.get("type") == "comm"
                        and block.get("text")
                        and merged_blocks
                        and merged_blocks[-1].get("type") == "comm"
                        and merged_blocks[-1].get("text")
                    ):
                        text = block["text"]
                        if block.get("_column_fill") and (text[:1] in ";,.)" or text[:1].islower()):
                            merged_blocks[-1]["text"] = (
                                merged_blocks[-1].get("text", "") + " " + text
                            ).strip()
                            continue
                    merged_blocks.append(block)
                payload["blocks"] = merged_blocks
                if missing_text_blocks and column_lines:
                    payload["blocks"] = [
                        b for b in payload.get("blocks", [])
                        if not (
                            b.get("type") == "comm"
                            and b.get("timestamp_correction") == "inferred_missing"
                            and not b.get("speaker")
                            and not b.get("text")
                        )
                    ]
                for block in payload.get("blocks", []):
                    block.pop("_column_fill", None)

            # Recompute page/tape metadata (ignore OCR header Page/Tape lines)
            header = payload.get("header", {})
            logical_page = page_num + 1 + page_offset
            header["page"] = logical_page

            # Start tape numbering when logical page reaches 1
            if not tape_started and logical_page >= 1:
                tape_started = True
                tape_y = 1

            # Validate and correct tape using OCR + logic
            if tape_started:
                ocr_tape = header.get("tape")
                tape_x, tape_y, was_corrected = validate_and_correct_tape(
                    ocr_tape, tape_x, tape_y, prev_has_end_of_tape
                )
                header["tape"] = format_tape(tape_x, tape_y)
            else:
                header["tape"] = None

            payload["header"] = header

            # Detect END OF TAPE marker for next page's validation
            blocks = payload.get("blocks", [])
            prev_has_end_of_tape = any(
                b.get("type") == "meta"
                and isinstance(b.get("text"), str)
                and "END OF TAPE" in b.get("text").upper()
                for b in blocks
            )

            # Update index
            t0 = time.perf_counter()
            page_timestamps = [
                b.get("timestamp") for b in payload.get("blocks", [])
                if b.get("type") == "comm" and b.get("timestamp")
            ]
            ts_index.add_timestamps(page_num, page_timestamps)
            stage_t["ts_index"] = time.perf_counter() - t0

            # Prune empty blocks
            payload["blocks"] = post_processor.prune_empty_blocks(payload.get("blocks", []))

            t0 = time.perf_counter()
            (ocr_dir / f"{page_id}_ocr_raw.txt").write_text(raw_text_output, encoding="utf-8")
            (page_dir / f"{page_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            stage_t["write_out"] = time.perf_counter() - t0

            page_duration = time.perf_counter() - page_start
            postprocess_time = sum(
                stage_t.get(k, 0.0) for k in (
                    "parse_plain", "build_plain", "parse_raw", "build_raw", "merge_raw",
                    "enhance_faint", "parse_faint", "build_faint", "merge_faint",
                    "ts_index", "write_out"
                )
            )
            timings[page_num] = {
                "ocr_s": page_duration - postprocess_time,
                "postprocess_s": postprocess_time,
                "ocr_total_s": page_duration,
                **stage_t
            }
            timing_info = (
                f"extract={pr.extract_s or 0:.3f}s "
                f"process={pr.process_s or 0:.3f}s "
                f"output={pr.output_s or 0:.3f}s "
                f"ocr_plain={stage_t.get('ocr_plain', 0):.3f}s "
                f"ocr_raw={stage_t.get('ocr_raw', 0):.3f}s "
                f"ocr_faint={stage_t.get('ocr_faint', 0):.3f}s "
                f"ocr_textcol={stage_t.get('ocr_textcol', 0):.3f}s "
                f"ocr={timings[page_num]['ocr_s']:.3f}s "
                f"postprocess={timings[page_num]['postprocess_s']:.3f}s "
                f"ocr_total={timings[page_num]['ocr_total_s']:.3f}s"
            )
            console.update_ocr_progress(page_num, page_duration, timing_info=timing_info)

            # Update previous_block_type for continuation handling
            if payload.get("blocks"):
                previous_block_type = payload["blocks"][-1].get("type")
            else:
                previous_block_type = None

            # Small pause to let the OCR server breathe/cleanup VRAM
            time.sleep(0.5)

        except Exception as e:
            failures += 1
            console.fail_ocr(page_num, str(e))
            logger.error(f"OCR failed for page {page_num + 1}: {e}")

    ts_index.save()
    return failures, timings


@click.group()
@click.version_option(version="1.0.0", prog_name="nasa-transcript")
def cli():
    """NASA Transcript Processing Pipeline."""
    pass


@cli.command()
@click.argument("pdf_name")
def export(pdf_name: str):
    """Merge per-page JSON into a global JSON and formatted TXT."""
    setup_logging()
    global_cfg = load_global_config(Path("config/defaults.toml"))
    pdf_path = resolve_pdf_path(pdf_name, global_cfg.input_dir)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        raise SystemExit(1)

    json_path = write_global_outputs(global_cfg.output_dir, pdf_path.stem)
    print(f"Wrote global JSON: {json_path}")


@cli.command("postprocess")
@click.argument("pdf_name")
def postprocess_json(pdf_name: str):
    """Post-process existing per-page JSON without rerunning OCR."""
    setup_logging()
    global_cfg = load_global_config(Path("config/defaults.toml"))
    pdf_path = resolve_pdf_path(pdf_name, global_cfg.input_dir)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        raise SystemExit(1)

    mission_cfg = load_mission_config(Path("config"), pdf_path.name)
    mission_overrides = mission_cfg.layout_overrides or {}

    valid_speakers = mission_overrides.get("valid_speakers")
    valid_locations = mission_overrides.get("valid_locations")
    lexicon_cfg = global_cfg.pipeline_defaults.get("lexicon", {})
    global_mission_keywords = (
        lexicon_cfg.get("mission_keywords")
        if isinstance(lexicon_cfg, dict)
        else []
    )
    mission_keyword_overrides = mission_overrides.get("mission_keywords", [])
    mission_keywords: list[str] = []
    for kw in (global_mission_keywords or []):
        if kw not in mission_keywords:
            mission_keywords.append(kw)
    for kw in (mission_keyword_overrides or []):
        if kw not in mission_keywords:
            mission_keywords.append(kw)

    global_parser = global_cfg.pipeline_defaults.get("parser", {})
    if isinstance(global_parser, dict):
        global_replacements = global_parser.get("text_replacements", {})
    else:
        global_replacements = {}
    mission_replacements = mission_overrides.get("text_replacements", {})
    text_replacements = {**global_replacements, **mission_replacements}

    # Load corrector configs
    global_correctors = global_cfg.pipeline_defaults.get("correctors", {})
    global_speaker_ocr_fixes = global_correctors.get("speaker_ocr_fixes", {})
    mission_speaker_ocr_fixes = mission_overrides.get("speaker_ocr_fixes", {})
    speaker_ocr_fixes = {**global_speaker_ocr_fixes, **mission_speaker_ocr_fixes}
    invalid_location_annotations = global_correctors.get("invalid_location_annotations", {}).get("annotations", [])
    manual_speaker_corrections = mission_overrides.get("manual_speaker_corrections", {})

    output_dir = global_cfg.output_dir / pdf_path.stem / "pages"
    if not output_dir.exists():
        print(f"Error: output not found: {output_dir}")
        raise SystemExit(1)

    # Rebuild timestamp index from updated JSON
    index_path = global_cfg.state_dir / f"{pdf_path.stem}_timestamps_index.json"
    ts_index = GlobalTimestampIndex(index_path)

    # Get lexicon path from config with fallback to default
    lexicon_path_str = lexicon_cfg.get("path", "resources/lexicon/apollo11_lexicon.json")
    lexicon_path = Path(lexicon_path_str)

    page_files = sorted(output_dir.glob("Page_*/**/*.json"))
    updated = 0

    # Initialize PostProcessor
    logger.info(f"Initializing PostProcessor with {len(valid_speakers) if valid_speakers else 0} valid speakers")
    post_processor = PostProcessor(
        valid_speakers=valid_speakers,
        valid_locations=valid_locations,
        mission_keywords=mission_keywords,
        text_replacements=text_replacements,
        speaker_ocr_fixes=speaker_ocr_fixes,
        invalid_location_annotations=invalid_location_annotations,
        manual_speaker_corrections=manual_speaker_corrections,
        lexicon_path=lexicon_path,
    )

    # Initialize tape tracking for validation
    page_offset = mission_overrides.get("page_offset", 0)
    tape_x = 1
    tape_y = 1
    tape_started = False
    prev_has_end_of_tape = False

    for page_file in page_files:
        try:
            payload = json.loads(page_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        blocks = payload.get("blocks", [])
        if not blocks:
            continue

        # Determine page number for index lookup
        match = re.search(r"_page_(\d{4})\\.json$", page_file.name)
        page_num = 0
        if match:
            page_num = int(match.group(1)) - 1

        initial_ts = ts_index.get_last_timestamp_before(page_num)

        # Run Post-Processing
        new_blocks = post_processor.process_blocks(blocks, initial_ts)

        payload["blocks"] = new_blocks

        # Update tape validation
        if match:
            header = payload.get("header", {})
            logical_page = page_num + 1 + page_offset
            header["page"] = logical_page

            # Start tape numbering when logical page reaches 1
            if not tape_started and logical_page >= 1:
                tape_started = True
                tape_y = 1

            # Validate and correct tape using OCR + logic
            if tape_started:
                ocr_tape = header.get("tape")
                tape_x, tape_y, was_corrected = validate_and_correct_tape(
                    ocr_tape, tape_x, tape_y, prev_has_end_of_tape
                )
                header["tape"] = format_tape(tape_x, tape_y)
            else:
                header["tape"] = None

            payload["header"] = header

            # Detect END OF TAPE marker for next page's validation
            prev_has_end_of_tape = any(
                b.get("type") == "meta"
                and isinstance(b.get("text"), str)
                and "END OF TAPE" in b.get("text").upper()
                for b in new_blocks
            )

            page_timestamps = [
                b.get("timestamp") for b in new_blocks
                if b.get("type") == "comm" and b.get("timestamp")
            ]
            ts_index.add_timestamps(page_num, page_timestamps)

        # Prune empty blocks
        payload["blocks"] = post_processor.prune_empty_blocks(payload.get("blocks", []))

        page_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        updated += 1

    ts_index.save()
    print(f"Post-processed pages: {updated}")

@cli.command("reparse")
@click.argument("pdf_name")
@click.option("-p", "--pages", help="Page range (e.g., '1-50', '10,12-14')")
def reparse_from_ocr(pdf_name: str, pages: str):
    """Reparse pages from stored OCR text files without rerunning OCR."""
    setup_logging()
    global_cfg = load_global_config(Path("config/defaults.toml"))
    pdf_path = resolve_pdf_path(pdf_name, global_cfg.input_dir)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        raise SystemExit(1)

    mission_cfg = load_mission_config(Path("config"), pdf_path.name)
    mission_overrides = mission_cfg.layout_overrides or {}

    # Configs
    valid_speakers = mission_overrides.get("valid_speakers")
    valid_locations = mission_overrides.get("valid_locations")
    lexicon_cfg = global_cfg.pipeline_defaults.get("lexicon", {})
    # Get lexicon path from config with fallback to default
    lexicon_path_str = lexicon_cfg.get("path", "resources/lexicon/apollo11_lexicon.json")
    lexicon_path = Path(lexicon_path_str)
    global_mission_keywords = (
        lexicon_cfg.get("mission_keywords")
        if isinstance(lexicon_cfg, dict)
        else []
    )
    mission_keyword_overrides = mission_overrides.get("mission_keywords", [])
    mission_keywords: list[str] = []
    for kw in (global_mission_keywords or []):
        if kw not in mission_keywords:
            mission_keywords.append(kw)
    for kw in (mission_keyword_overrides or []):
        if kw not in mission_keywords:
            mission_keywords.append(kw)

    global_parser = global_cfg.pipeline_defaults.get("parser", {})
    if isinstance(global_parser, dict):
        global_replacements = global_parser.get("text_replacements", {})
    else:
        global_replacements = {}
    mission_replacements = mission_overrides.get("text_replacements", {})
    text_replacements = {**global_replacements, **mission_replacements}

    global_correctors = global_cfg.pipeline_defaults.get("correctors", {})
    global_speaker_ocr_fixes = global_correctors.get("speaker_ocr_fixes", {})
    mission_speaker_ocr_fixes = mission_overrides.get("speaker_ocr_fixes", {})
    speaker_ocr_fixes = {**global_speaker_ocr_fixes, **mission_speaker_ocr_fixes}
    invalid_location_annotations = global_correctors.get("invalid_location_annotations", {}).get("annotations", [])
    manual_speaker_corrections = mission_overrides.get("manual_speaker_corrections", {})
    raw_footer_overrides = mission_overrides.get("footer_text_overrides", {})
    footer_text_overrides = {int(k): v for k, v in raw_footer_overrides.items()} if raw_footer_overrides else None

    output_dir = global_cfg.output_dir / pdf_path.stem / "pages"
    if not output_dir.exists():
        print(f"Error: output not found: {output_dir}")
        raise SystemExit(1)

    index_path = global_cfg.state_dir / f"{pdf_path.stem}_timestamps_index.json"
    ts_index = GlobalTimestampIndex(index_path)

    # Filter pages if range provided
    if pages:
        # Get total pages from PDF info for bounds checking
        info = get_pdf_info(pdf_path)
        page_indices = parse_pages(pages, info["page_count"])
        page_dirs = []
        for p_idx in page_indices:
            p_dir = output_dir / f"Page_{p_idx + 1:03d}"
            if p_dir.exists():
                page_dirs.append(p_dir)
    else:
        page_dirs = sorted(
            output_dir.glob("Page_*"),
            key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)) if re.search(r"(\d+)$", p.name) else 0
        )

    # Initialize PostProcessor
    logger.info(f"Initializing PostProcessor with {len(valid_speakers) if valid_speakers else 0} valid speakers")
    post_processor = PostProcessor(
        valid_speakers=valid_speakers,
        valid_locations=valid_locations,
        mission_keywords=mission_keywords,
        text_replacements=text_replacements,
        speaker_ocr_fixes=speaker_ocr_fixes,
        invalid_location_annotations=invalid_location_annotations,
        manual_speaker_corrections=manual_speaker_corrections,
        lexicon_path=lexicon_path,
    )

    tape_x = 1
    tape_y = 1
    tape_started = False
    prev_has_end_of_tape = False
    previous_block_type = None
    updated = 0

    for page_dir in page_dirs:
        ocr_dir = page_dir / "ocr"
        raw_paths = sorted(ocr_dir.glob("*_ocr_raw.txt"))
        if not raw_paths:
            continue
        raw_path = raw_paths[0]

        match = re.search(r"_page_(\d{4})_ocr_raw\.txt$", raw_path.name)
        if not match:
            continue
        page_num = int(match.group(1)) - 1
        page_id = f"{pdf_path.stem}_page_{page_num + 1:04d}"

        raw_text = raw_path.read_text(encoding="utf-8")
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        rows, has_footer = parse_ocr_text(text, page_num, mission_keywords, valid_speakers)

        initial_ts = ts_index.get_last_timestamp_before(page_num)

        payload = build_page_json(
            rows, lines, page_num, mission_cfg.page_offset,
            valid_speakers, text_replacements,
            mission_keywords, valid_locations,
            inline_annotation_terms=invalid_location_annotations,
            initial_ts=initial_ts,
            previous_block_type=previous_block_type,
            lexicon_path=lexicon_path,
            footer_text_overrides=footer_text_overrides,
            speaker_ocr_fixes=speaker_ocr_fixes,
            has_footer=has_footer
        )

        # Optional fallback OCR passes (if files exist)
        raw_fallback_paths = sorted(ocr_dir.glob("*_ocr_raw_fallback.txt"))
        if raw_fallback_paths:
            raw_fallback = raw_fallback_paths[0].read_text(encoding="utf-8")
            raw_fallback = raw_fallback.replace("\r\n", "\n").replace("\r", "\n")
            fallback_lines = [line.strip() for line in raw_fallback.splitlines() if line.strip()]
            fallback_rows, fallback_has_footer = parse_ocr_text(raw_fallback, page_num, mission_keywords, valid_speakers)
            fallback_payload = build_page_json(
                fallback_rows, fallback_lines, page_num, mission_cfg.page_offset,
                valid_speakers, text_replacements,
                mission_keywords, valid_locations,
                inline_annotation_terms=invalid_location_annotations,
                initial_ts=initial_ts,
                previous_block_type=previous_block_type,
                lexicon_path=lexicon_path,
                footer_text_overrides=footer_text_overrides,
                speaker_ocr_fixes=speaker_ocr_fixes,
                has_footer=fallback_has_footer
            )
            payload = merge_payloads(payload, fallback_payload)

        faint_paths = sorted(ocr_dir.glob("*_ocr_faint_fallback.txt"))
        if faint_paths:
            faint_text = faint_paths[0].read_text(encoding="utf-8")
            faint_text = faint_text.replace("\r\n", "\n").replace("\r", "\n")
            faint_lines = [line.strip() for line in faint_text.splitlines() if line.strip()]
            faint_rows, faint_has_footer = parse_ocr_text(faint_text, page_num, mission_keywords, valid_speakers)
            faint_payload = build_page_json(
                faint_rows, faint_lines, page_num, mission_cfg.page_offset,
                valid_speakers, text_replacements,
                mission_keywords, valid_locations,
                inline_annotation_terms=invalid_location_annotations,
                initial_ts=initial_ts,
                previous_block_type=previous_block_type,
                lexicon_path=lexicon_path,
                footer_text_overrides=footer_text_overrides,
                speaker_ocr_fixes=speaker_ocr_fixes,
                has_footer=faint_has_footer
            )
            payload = merge_payloads(payload, faint_payload)

        # Run Post-Processing on the merged payload
        payload["blocks"] = post_processor.process_blocks(payload.get("blocks", []), initial_ts)

        # Recompute page/tape metadata
        header = payload.get("header", {})
        logical_page = page_num + 1 + mission_cfg.page_offset
        header["page"] = logical_page

        # Get blocks for tape validation
        blocks = payload.get("blocks", [])

        # Start tape numbering when logical page reaches 1
        if not tape_started and logical_page >= 1:
            tape_started = True
            tape_y = 1

        # Validate and correct tape using OCR + logic
        if tape_started:
            ocr_tape = header.get("tape")
            tape_x, tape_y, was_corrected = validate_and_correct_tape(
                ocr_tape, tape_x, tape_y, prev_has_end_of_tape
            )
            header["tape"] = format_tape(tape_x, tape_y)
        else:
            header["tape"] = None

        payload["header"] = header

        # Detect END OF TAPE marker for next page's validation
        prev_has_end_of_tape = any(
            b.get("type") == "meta"
            and isinstance(b.get("text"), str)
            and "END OF TAPE" in b.get("text").upper()
            for b in blocks
        )

        # Update index
        page_timestamps = [
            b.get("timestamp") for b in blocks
            if b.get("type") == "comm" and b.get("timestamp")
        ]
        ts_index.add_timestamps(page_num, page_timestamps)

        # Prune empty blocks
        blocks = post_processor.prune_empty_blocks(blocks)
        payload["blocks"] = blocks

        # Update previous_block_type for continuation handling
        previous_block_type = blocks[-1].get("type") if blocks else None

        # Persist page JSON
        page_json = page_dir / f"{page_id}.json"
        page_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        updated += 1

    ts_index.save()
    print(f"Reparsed pages from OCR text: {updated}")

@cli.command()
@click.argument("pdf_name")
@click.option("-p", "--pages", help="Page range (e.g., '1-50', '10,12-14')")
@click.option("--clean", is_flag=True, help="Remove existing output first")
@click.option("--no-ocr", is_flag=True, help="Skip OCR step")
@click.option("--ocr-url", help="LM Studio URL (overrides config)")
@click.option(
    "--ocr-prompt",
    type=click.Choice(["plain", "column"], case_sensitive=False),
    help="OCR prompt mode (overrides config)"
)
@click.option("--timing/--no-timing", default=None, help="Print per-page timing breakdowns")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def process(
    pdf_name: str,
    pages: str,
    clean: bool,
    no_ocr: bool,
    ocr_url: str,
    ocr_prompt: str,
    timing: bool | None,
    verbose: bool
):
    """
    Process a PDF document.
    """
    setup_logging(verbose)

    # Load configuration
    global_cfg = load_global_config(Path("config/defaults.toml"))
    if ocr_url:
        global_cfg.ocr_url = ocr_url
    if ocr_prompt:
        global_cfg.ocr_prompt = ocr_prompt.lower()

    if timing is None:
        timing = bool(global_cfg.pipeline_defaults.get("timing", False))

    # Resolve PDF
    pdf_path = resolve_pdf_path(pdf_name, global_cfg.input_dir)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        raise SystemExit(1)

    mission_cfg = load_mission_config(Path("config"), pdf_path.name)

    # Apply mission overrides to global config
    config = global_cfg
    if mission_cfg.layout_overrides:
        # Create a new config object with updates
        updated_data = config.model_dump()
        # Only update fields that exist in the top-level config
        # (Nested configs like parser/correctors are handled separately downstream)
        overrides = {
            k: v for k, v in mission_cfg.layout_overrides.items()
            if k in updated_data
        }
        updated_data.update(overrides)
        config = GlobalConfig(**updated_data)

    # Setup output
    output_dir = global_cfg.output_dir / pdf_path.stem
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    # Initialize pipeline
    pipeline = TranscriptPipeline(pdf_path, global_cfg.output_dir, config)

    # Parse pages
    page_numbers = parse_pages(pages or "", pipeline.page_count)
    if not page_numbers:
        print("No valid pages to process.")
        raise SystemExit(1)

    # Start Console UI
    console.start_pipeline(len(page_numbers), pdf_path.name)

    # Run image processing pipeline
    # We pass a lambda as callback to update the rich progress bar
    result = pipeline.process_pages(
        page_numbers,
        progress_callback=lambda current, total: console.update_image_progress(1)
    )

    # Run OCR
    if not no_ocr:
        # Merge global and mission replacements
        global_parser = global_cfg.pipeline_defaults.get("parser", {})
        if isinstance(global_parser, dict):
            global_replacements = global_parser.get("text_replacements", {})
        else:
            global_replacements = {}

        mission_replacements = mission_cfg.layout_overrides.get("text_replacements", {})
        text_replacements = {**global_replacements, **mission_replacements}

        lexicon_cfg = global_cfg.pipeline_defaults.get("lexicon", {})
        # Get lexicon path from config with fallback to default
        lexicon_path_str = lexicon_cfg.get("path", "resources/lexicon/apollo11_lexicon.json")
        lexicon_path = Path(lexicon_path_str)
        global_mission_keywords = (
            lexicon_cfg.get("mission_keywords")
            if isinstance(lexicon_cfg, dict)
            else []
        )
        mission_keyword_overrides = mission_cfg.layout_overrides.get("mission_keywords", [])
        mission_keywords: list[str] = []
        for kw in (global_mission_keywords or []):
            if kw not in mission_keywords:
                mission_keywords.append(kw)
        for kw in (mission_keyword_overrides or []):
            if kw not in mission_keywords:
                mission_keywords.append(kw)
        valid_speakers = mission_cfg.layout_overrides.get("valid_speakers")
        valid_locations = mission_cfg.layout_overrides.get("valid_locations")

        ocr_failures, ocr_timings = run_ocr_pipeline(
            pdf_path,
            global_cfg,
            result.page_results,
            mission_cfg.page_offset,
            valid_speakers,
            text_replacements,
            mission_keywords,
            valid_locations,
            mission_cfg.layout_overrides,
            lexicon_path
        )
        if timing:
            for pr in result.page_results:
                if not pr.success:
                    continue
                page_no = pr.page_num + 1
                ocr_t = ocr_timings.get(pr.page_num, {})
                print(
                    f"Page {page_no:04d} timings: "
                    f"extract={pr.extract_s or 0:.3f}s "
                    f"process={pr.process_s or 0:.3f}s "
                    f"output={pr.output_s or 0:.3f}s "
                    f"ocr_plain={ocr_t.get('ocr_plain', 0):.3f}s "
                    f"ocr_raw={ocr_t.get('ocr_raw', 0):.3f}s "
                    f"ocr_faint={ocr_t.get('ocr_faint', 0):.3f}s "
                    f"ocr_textcol={ocr_t.get('ocr_textcol', 0):.3f}s "
                    f"ocr={ocr_t.get('ocr_s', 0):.3f}s "
                    f"postprocess={ocr_t.get('postprocess_s', 0):.3f}s "
                    f"ocr_total={ocr_t.get('ocr_total_s', 0):.3f}s"
                )

    # Always refresh global outputs at the end of a process run.
    merged_json = write_global_outputs(global_cfg.output_dir, pdf_path.stem)
    print(f"Wrote global JSON: {merged_json}")

    console.finish()

    if result.failed_pages > 0:
        raise SystemExit(1)


@cli.command()
@click.argument("pdf_name")
def info(pdf_name: str):
    """Display PDF information."""
    setup_logging()
    global_cfg = load_global_config(Path("config/defaults.toml"))
    pdf_path = resolve_pdf_path(pdf_name, global_cfg.input_dir)

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        raise SystemExit(1)

    info = get_pdf_info(pdf_path)
    # Using rich to display info nicely
    from rich import box
    from rich.table import Table

    table = Table(title=f"PDF Info: {pdf_path.name}", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Pages", str(info['page_count']))
    for key in ("title", "author", "creator", "producer"):
        table.add_row(key.title(), str(info[key] or '(none)'))

    rich_console.print(table)


if __name__ == "__main__":
    cli()
