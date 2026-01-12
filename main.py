#!/usr/bin/env python3
"""
NASA Transcript Processing Pipeline - CLI Interface.

This module provides the command-line interface for processing NASA
mission transcript PDFs.

Usage:
    python main.py process AS11_TEC.PDF
    python main.py process AS11_TEC.PDF --pages 1-50
    python main.py process AS11_TEC.PDF --ocr-url http://localhost:1234
    python main.py info AS11_TEC.PDF

For AI Agents:
    - Main entry point for the pipeline
    - Uses click for CLI argument parsing
    - Supports page range selection with --pages
    - Parallel processing enabled by default
"""

import sys
import shutil
import time
import json
import re
from pathlib import Path
from typing import Optional

import click
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.image_processor import ImageProcessor
from src.layout_detector import LayoutDetector
from src.ocr_client import (
    LMStudioOCRClient,
    PLAIN_OCR_PROMPT,
)
from src.output_generator import OutputGenerator
from src.page_extractor import get_pdf_info
from src.page_extractor import PageExtractor
from src.pipeline import TranscriptPipeline
from src.mission_config import load_mission_config
from src.global_config import load_global_config


def setup_logging() -> None:
    """Configure logging for the CLI."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="WARNING",
        colorize=True
    )


def resolve_pdf_path(pdf_arg: str, input_dir: Path) -> Path:
    """
    Resolve a PDF path from an argument and an input directory.

    If the argument points to an existing file, use it directly.
    Otherwise, treat it as a filename within the input directory.
    """
    candidate = Path(pdf_arg)
    if candidate.exists():
        return candidate

    name = pdf_arg
    if Path(name).suffix == "":
        name = f"{name}.pdf"
    return input_dir / name


def parse_page_range(page_range: str, max_pages: int) -> tuple[int, int]:
    """
    Parse a page range string like "1-50" or "10".

    Args:
        page_range: Page range string (1-indexed)
        max_pages: Maximum page number

    Returns:
        Tuple of (start, end) 0-indexed page numbers

    Raises:
        click.BadParameter: If range is invalid
    """
    try:
        if "-" in page_range:
            parts = page_range.split("-")
            if len(parts) != 2:
                raise ValueError("Invalid range format")
            start = int(parts[0]) - 1  # Convert to 0-indexed
            end = int(parts[1])  # Keep as exclusive end
        else:
            # Single page
            page = int(page_range) - 1
            start = page
            end = page + 1

        # Validate range
        if start < 0:
            raise ValueError("Start page must be >= 1")
        if end > max_pages:
            end = max_pages
        if start >= end:
            raise ValueError("Start must be less than end")

        return start, end

    except ValueError as e:
        raise click.BadParameter(f"Invalid page range '{page_range}': {e}")


def parse_pages(page_ranges: str, max_pages: int) -> list[int]:
    """
    Parse page ranges like "1-50, 10, 12-14" into a list of page numbers.

    Args:
        page_ranges: Comma-separated page ranges (1-indexed)
        max_pages: Maximum page number

    Returns:
        Sorted list of 0-indexed page numbers
    """
    if not page_ranges:
        return []

    pages = set()
    tokens = [part.strip() for part in page_ranges.split(",") if part.strip()]
    for token in tokens:
        if "-" in token:
            start, end = parse_page_range(token, max_pages)
            pages.update(range(start, end))
        else:
            try:
                page = int(token) - 1
            except ValueError as e:
                raise click.BadParameter(f"Invalid page '{token}': {e}")
            if page < 0:
                raise click.BadParameter(f"Invalid page '{token}': must be >= 1")
            if page >= max_pages:
                continue
            pages.add(page)

    return sorted(pages)


def resolve_output_dir(base_output: Path, pdf_stem: str) -> Path:
    """Resolve output directory matching pipeline behavior."""
    if base_output.name == pdf_stem:
        return base_output
    return base_output / pdf_stem


def run_ocr(
    pdf_name: str,
    global_config,
    clean: bool,
    pages: Optional[str],
    base_url: Optional[str]
) -> None:
    """Run OCR via LM Studio with the simplified defaults."""
    setup_logging()

    input_dir = global_config.input_dir
    output = global_config.output_dir
    dpi = global_config.dpi
    base_url = base_url or global_config.ocr_url

    pdf_path = resolve_pdf_path(pdf_name, input_dir)
    if not pdf_path.exists():
        click.echo(f"Error: PDF not found: {pdf_path}", err=True)
        raise click.Abort()

    config = PipelineConfig(
        dpi=dpi,
        parallel=False,
        max_workers=1,
        debug=False
    )

    extractor = PageExtractor(pdf_path, config)
    processor = ImageProcessor(config)
    output_dir = resolve_output_dir(output, pdf_path.stem)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layout_detector = LayoutDetector(config)
    output_generator = OutputGenerator(output_dir, pdf_path.stem, config)
    mission_config = load_mission_config(Path("config"), pdf_path.name)
    page_offset = mission_config.page_offset
    if pages:
        page_numbers = parse_pages(pages, extractor.page_count)
        if not page_numbers:
            click.echo("No valid pages to process.", err=True)
            raise click.Abort()
        click.echo(
            f"OCR {len(page_numbers)} pages "
            f"(from {page_numbers[0] + 1} to {page_numbers[-1] + 1}) "
            f"of {extractor.page_count}"
        )
    else:
        page_numbers = list(range(extractor.page_count))
        click.echo(f"OCR all {extractor.page_count} pages")

    client = LMStudioOCRClient(
        base_url=base_url,
        model="qwen3-vl-4b",
        timeout_s=120,
        max_tokens=4096,
        prompt=PLAIN_OCR_PROMPT,
        image_mode="auto",
        image_token="auto"
    )

    total_start = time.perf_counter()
    failures = 0

    for page_num in page_numbers:
        page_start = time.perf_counter()
        image = extractor.extract_page_image(page_num)
        enhanced = processor.process(image).image

        page_dir = output_generator.get_page_dir(page_num)
        page_id = f"{pdf_path.stem}_page_{page_num + 1:04d}"
        raw_path = page_dir / f"{page_id}_ocr_raw.txt"
        json_path = page_dir / f"{page_id}.json"
        layout = layout_detector.detect(enhanced)
        output_generator.generate(page_num, enhanced, layout)

        try:
            text = client.ocr_image(enhanced)
            raw_path.write_text(text + "\n", encoding="utf-8")
            raw_lines = [line.strip("\r").strip() for line in text.splitlines()]
            raw_lines = [line for line in raw_lines if line]
            rows = parse_ocr_output_plain(text, page_num)
            payload = build_page_json(rows, raw_lines, page_num, page_offset=page_offset)
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            failures += 1
            raw_path.write_text(f"[OCR_ERROR] {exc}\n", encoding="utf-8")
            payload = {
                "page": {"number": page_num + 1 + page_offset},
                "blocks": [],
                "error": str(exc),
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        elapsed = time.perf_counter() - page_start
        click.echo(f"Page {page_num + 1}/{extractor.page_count} in {format_duration(elapsed)}")

    total_elapsed = time.perf_counter() - total_start
    avg = total_elapsed / max(1, len(page_numbers))
    estimate = avg * extractor.page_count

    click.echo("")
    click.echo("OCR Complete")
    click.echo(f"Processed: {len(page_numbers)} pages")
    click.echo(f"Failed: {failures}")
    click.echo(f"Total time: {format_duration(total_elapsed)}")
    click.echo(f"Average per page: {avg:.2f}s")
    click.echo(f"Estimated full run ({extractor.page_count} pages): {format_duration(estimate)}")
    click.echo(f"Output: {output_dir}")


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


TIMESTAMP_LINE_RE = re.compile(r"^\d{2}\s+\d{2}\s+\d{2}\s+\d{1,2}$")
TIMESTAMP_PREFIX_RE = re.compile(r"^(\d{2}\s+\d{2}\s+\d{2}\s+\d{1,2})\b")
SPEAKER_LINE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,6}(?:\s*\([A-Z0-9]+\))?$")
SPEAKER_PAREN_RE = re.compile(r"^\([A-Z0-9]+\)$")
HEADER_PAGE_RE = re.compile(r"\bPAGE\s*(\d{1,4})\b", re.IGNORECASE)
HEADER_TAPE_RE = re.compile(r"\bTAPE\s*([0-9]{1,2}\s*/\s*[0-9]{1,2})\b", re.IGNORECASE)
HEADER_APOLLO_KEYS = ("APOLLO", "AIR", "GROUND", "VOICE", "TRANSCRIPTION")


def normalize_header_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def extract_header_metadata(raw_lines: list[str], page_num: int, page_offset: int) -> dict:
    header = {"number": page_num + 1 + page_offset, "tape": "", "apollo": ""}

    first_ts_idx = None
    for idx, line in enumerate(raw_lines):
        if TIMESTAMP_LINE_RE.match(line) or TIMESTAMP_PREFIX_RE.match(line):
            first_ts_idx = idx
            break

    if first_ts_idx is None:
        header_lines = raw_lines[:10]
    else:
        header_lines = raw_lines[: first_ts_idx + 1]

    apollo_parts: list[str] = []
    for line in header_lines:
        if not line:
            continue
        if TIMESTAMP_PREFIX_RE.match(line):
            continue
        normalized = normalize_header_line(line)
        if not normalized:
            continue
        page_match = HEADER_PAGE_RE.search(normalized)
        if page_match:
            header["number"] = int(page_match.group(1))
        tape_match = HEADER_TAPE_RE.search(normalized)
        if tape_match:
            header["tape"] = tape_match.group(1).replace(" ", "")
        upper = normalized.upper()
        if any(key in upper for key in HEADER_APOLLO_KEYS):
            apollo_parts.append(normalized)

    if apollo_parts:
        apollo_text = normalize_header_line(" ".join(apollo_parts))
        if "APOLLO" in apollo_text.upper():
            header["apollo"] = apollo_text

    return header


def build_page_json(rows: list[dict], raw_lines: list[str], page_num: int, page_offset: int) -> dict:
    page_info = extract_header_metadata(raw_lines, page_num, page_offset)
    blocks = []

    for row in rows:
        if row["type"] == "header":
            normalized = normalize_header_line(row["text"])
            if normalized:
                page_match = HEADER_PAGE_RE.search(normalized)
                if page_match:
                    page_info["number"] = int(page_match.group(1))
                tape_match = HEADER_TAPE_RE.search(normalized)
                if tape_match:
                    page_info["tape"] = tape_match.group(1).replace(" ", "")
                upper = normalized.upper()
                if "APOLLO" in upper and "AIR" in upper:
                    page_info["apollo"] = normalized
            continue

        block_type = row["type"]
        if block_type == "text":
            block_type = "continuation"

        block = {"type": block_type}
        if block_type == "comm":
            if row["timestamp"]:
                block["timestamp"] = row["timestamp"]
            if row["speaker"]:
                block["speaker"] = row["speaker"]
            if row["text"]:
                block["text"] = row["text"]
        else:
            if row["text"]:
                block["text"] = row["text"]
        blocks.append(block)

    return {"page": page_info, "blocks": blocks}


def parse_ocr_output_plain(text: str, page_num: int) -> list[dict]:
    """Parse plain OCR output into rows using simple heuristics."""
    raw_lines = [line.strip("\r").strip() for line in text.splitlines()]
    raw_lines = [line for line in raw_lines if line]

    first_ts_idx = None
    for idx, line in enumerate(raw_lines):
        if TIMESTAMP_LINE_RE.match(line) or TIMESTAMP_PREFIX_RE.match(line):
            first_ts_idx = idx
            break

    header_keywords = ("GOSS", "NET", "TAPE", "PAGE", "APOLLO", "AIR-TO-GROUND")
    rows = []
    line_index = 0
    pending_ts = ""
    pending_speaker = ""
    pending_text_lines: list[str] = []

    def flush_comm() -> None:
        nonlocal pending_ts, pending_speaker, pending_text_lines, line_index
        if not pending_ts and not pending_speaker and not pending_text_lines:
            return
        line_index += 1
        rows.append({
            "page": page_num + 1,
            "line": line_index,
            "type": "comm" if pending_ts else "text",
            "timestamp": pending_ts,
            "speaker": pending_speaker,
            "text": " ".join(pending_text_lines).strip(),
        })
        pending_ts = ""
        pending_speaker = ""
        pending_text_lines = []

    for idx, line in enumerate(raw_lines):
        upper = line.upper()
        is_header = (
            first_ts_idx is not None
            and idx <= first_ts_idx
            and not TIMESTAMP_PREFIX_RE.match(line)
            and any(key in upper for key in header_keywords)
        )
        is_footer = "***" in line or "ASTERISK" in upper
        is_annotation = "(REV" in upper

        if is_header or is_footer or is_annotation:
            flush_comm()
            line_index += 1
            rows.append({
                "page": page_num + 1,
                "line": line_index,
                "type": "header" if is_header else ("footer" if is_footer else "annotation"),
                "timestamp": "",
                "speaker": "",
                "text": line,
            })
            continue

        if TIMESTAMP_LINE_RE.match(line):
            flush_comm()
            pending_ts = line
            pending_speaker = ""
            pending_text_lines = []
            continue

        prefix_match = TIMESTAMP_PREFIX_RE.match(line)
        if prefix_match:
            flush_comm()
            pending_ts = prefix_match.group(1)
            remainder = line[len(pending_ts):].strip()
            pending_speaker = ""
            pending_text_lines = []
            if remainder:
                tokens = remainder.split()
                if tokens and SPEAKER_LINE_RE.match(tokens[0]):
                    pending_speaker = tokens[0]
                    tokens = tokens[1:]
                    if tokens and SPEAKER_PAREN_RE.match(tokens[0]):
                        pending_speaker = f"{pending_speaker} {tokens[0]}".strip()
                        tokens = tokens[1:]
                if tokens:
                    pending_text_lines.append(" ".join(tokens))
            continue

        if pending_ts:
            if SPEAKER_LINE_RE.match(line):
                if pending_speaker:
                    pending_speaker = f"{pending_speaker} {line}".strip()
                else:
                    pending_speaker = line
                continue
            if SPEAKER_PAREN_RE.match(line):
                pending_speaker = f"{pending_speaker} {line}".strip() if pending_speaker else line
                continue
            pending_text_lines.append(line)
            continue

        line_index += 1
        rows.append({
            "page": page_num + 1,
            "line": line_index,
            "type": "text",
            "timestamp": "",
            "speaker": "",
            "text": line,
        })

    flush_comm()
    return rows


@click.group()
@click.version_option(version="1.0.0", prog_name="nasa-transcript")
def cli():
    """
    NASA Transcript Processing Pipeline.

    Process scanned NASA mission transcripts with image enhancement
    and geometric layout detection.
    """
    pass


@cli.command()
@click.argument("pdf_name", type=str)
@click.option(
    "--pages", "-p",
    type=str,
    default=None,
    help="Page range to process (e.g., '1-50', '10', or '10,12,14-16'). Default: all pages."
)
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Remove existing output directory before processing"
)
@click.option(
    "--ocr-url",
    type=str,
    default=None,
    help="LM Studio base URL (overrides config/defaults.toml)"
)
def process(
    pdf_name: str,
    pages: Optional[str],
    clean: bool,
    ocr_url: Optional[str]
):
    """
    Process a PDF document.

    Extracts each page, enhances the image, detects layout blocks,
    and generates output files.

    Example:
        python main.py process AS11_TEC.PDF --pages 1-10
    """
    setup_logging()

    defaults_path = Path("config/defaults.toml")
    global_cfg = load_global_config(defaults_path)
    input_dir = global_cfg.input_dir
    output = global_cfg.output_dir
    ocr_url = ocr_url or global_cfg.ocr_url

    # Resolve PDF path
    pdf_path = resolve_pdf_path(pdf_name, input_dir)
    if not pdf_path.exists():
        click.echo(f"Error: PDF not found: {pdf_path}", err=True)
        raise click.Abort()

    # Create configuration
    config = PipelineConfig(
        dpi=global_cfg.dpi,
        parallel=global_cfg.parallel,
        max_workers=global_cfg.workers,
        debug=False
    )

    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            click.echo(f"Configuration error: {error}", err=True)
        raise click.Abort()

    output_dir = resolve_output_dir(output, pdf_path.stem)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    # Initialize pipeline
    try:
        pipeline = TranscriptPipeline(pdf_path, output, config)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

    # Parse page range
    if pages:
        page_numbers = parse_pages(pages, pipeline.page_count)
        if not page_numbers:
            click.echo("No valid pages to process.", err=True)
            raise click.Abort()
        if len(page_numbers) == 1:
            click.echo(f"Processing page {page_numbers[0] + 1} of {pipeline.page_count}")
        else:
            click.echo(
                f"Processing {len(page_numbers)} pages "
                f"(from {page_numbers[0] + 1} to {page_numbers[-1] + 1}) "
                f"of {pipeline.page_count}"
            )
    else:
        page_numbers = []
        click.echo(f"Processing all {pipeline.page_count} pages")

    # Show configuration
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"DPI: {config.dpi}")
    click.echo(f"Parallel: {config.parallel} (workers: {config.max_workers})")
    click.echo("")

    # Run pipeline
    if page_numbers:
        result = pipeline.process_pages(page_numbers)
    else:
        result = pipeline.process_range(0, pipeline.page_count)

    # Summary
    click.echo("")
    click.echo("=" * 50)
    click.echo("Processing Complete")
    click.echo("=" * 50)
    click.echo(f"Total pages: {result.total_pages}")
    click.echo(f"Processed: {result.processed_pages}")
    click.echo(f"Successful: {result.successful_pages}")
    click.echo(f"Failed: {result.failed_pages}")
    click.echo(f"Output: {result.output_dir}")

    # List failed pages if any
    if result.failed_pages > 0:
        click.echo("")
        click.echo("Failed pages:")
        for page_result in result.page_results:
            if not page_result.success:
                click.echo(f"  Page {page_result.page_num + 1}: {page_result.error}")

    # Exit with error code if any failures
    if result.failed_pages > 0:
        sys.exit(1)

    run_ocr(
        pdf_name=str(pdf_path),
        global_config=global_cfg,
        clean=False,
        pages=pages,
        base_url=ocr_url
    )


@cli.command()
@click.argument("pdf_name", type=str)
def info(pdf_name: str):
    """
    Display information about a PDF document.

    Shows page count, metadata, and page dimensions.
    """
    setup_logging()

    defaults_path = Path("config/defaults.toml")
    global_cfg = load_global_config(defaults_path)
    pdf_path = resolve_pdf_path(pdf_name, global_cfg.input_dir)
    if not pdf_path.exists():
        click.echo(f"Error: PDF not found: {pdf_path}", err=True)
        raise click.Abort()

    pdf_info = get_pdf_info(pdf_path)

    click.echo(f"File: {pdf_path}")
    click.echo(f"Pages: {pdf_info['page_count']}")
    click.echo(f"Title: {pdf_info['title'] or '(none)'}")
    click.echo(f"Author: {pdf_info['author'] or '(none)'}")
    click.echo(f"Creator: {pdf_info['creator'] or '(none)'}")
    click.echo(f"Producer: {pdf_info['producer'] or '(none)'}")
    click.echo(f"Created: {pdf_info['creation_date'] or '(none)'}")
    click.echo(f"Modified: {pdf_info['modification_date'] or '(none)'}")


if __name__ == "__main__":
    cli()
