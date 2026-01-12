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
import shutil
import sys
import time
from pathlib import Path

import click
import cv2
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.global_config import GlobalConfig, load_global_config
from src.image_processor import ImageProcessor
from src.layout_detector import LayoutDetector
from src.mission_config import load_mission_config
from src.ocr_client import PLAIN_OCR_PROMPT, LMStudioOCRClient
from src.ocr_parser import build_page_json, parse_ocr_text
from src.output_generator import OutputGenerator
from src.page_extractor import PageExtractor, get_pdf_info
from src.pipeline import PageResult, TranscriptPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
        level="DEBUG" if verbose else "WARNING",
        colorize=True
    )


def resolve_pdf_path(pdf_arg: str, input_dir: Path) -> Path:
    """Resolve PDF path from argument or input directory."""
    path = Path(pdf_arg)
    if path.exists():
        return path
    name = pdf_arg if path.suffix else f"{pdf_arg}.pdf"
    return input_dir / name


def parse_pages(page_spec: str, max_pages: int) -> list[int]:
    """
    Parse page specification like "1-50,10,12-14" into page numbers.

    Args:
        page_spec: Comma-separated page ranges (1-indexed)
        max_pages: Maximum page count

    Returns:
        Sorted list of 0-indexed page numbers
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


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    total = int(round(seconds))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_ocr_pipeline(
    pdf_path: Path,
    config: GlobalConfig,
    page_results: list[PageResult],
    page_offset: int = 0
) -> int:
    """
    Run OCR on already processed pages.

    Returns:
        Number of failed pages
    """
    client = LMStudioOCRClient(
        base_url=config.ocr_url,
        model=config.ocr_model,
        timeout_s=120,
        max_tokens=4096,
        prompt=PLAIN_OCR_PROMPT,
    )

    failures = 0
    total_start = time.perf_counter()
    total_to_ocr = sum(1 for pr in page_results if pr.success)

    logger.info(f"Starting OCR for {total_to_ocr} successfully processed pages")

    for pr in page_results:
        if not pr.success or not pr.output:
            continue

        page_num = pr.page_num
        page_start = time.perf_counter()
        page_dir = pr.output.page_dir
        page_id = f"{pdf_path.stem}_page_{page_num + 1:04d}"

        try:
            # Load enhanced image from disk instead of re-processing
            enhanced_path = pr.output.enhanced_image
            enhanced = cv2.imread(str(enhanced_path))
            if enhanced is None:
                raise FileNotFoundError(f"Enhanced image not found: {enhanced_path}")

            text = client.ocr_image(enhanced)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            rows = parse_ocr_text(text, page_num)
            payload = build_page_json(rows, lines, page_num, page_offset)

            (page_dir / f"{page_id}_ocr_raw.txt").write_text(text + "\n", encoding="utf-8")
            (page_dir / f"{page_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            failures += 1
            logger.error(f"OCR Error on page {page_num + 1}: {exc}")
            error_payload = {
                "page": {"number": page_num + 1 + page_offset},
                "blocks": [],
                "error": str(exc),
            }
            (page_dir / f"{page_id}.json").write_text(
                json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        elapsed = time.perf_counter() - page_start
        click.echo(f"  OCR page {page_num + 1} ({format_duration(elapsed)})")

    total = time.perf_counter() - total_start
    click.echo(f"OCR completed in {format_duration(total)} ({failures} failures)")
    return failures


@click.group()
@click.version_option(version="1.0.0", prog_name="nasa-transcript")
def cli():
    """NASA Transcript Processing Pipeline."""
    pass


@cli.command()
@click.argument("pdf_name")
@click.option("-p", "--pages", help="Page range (e.g., '1-50', '10,12-14')")
@click.option("--clean", is_flag=True, help="Remove existing output first")
@click.option("--no-ocr", is_flag=True, help="Skip OCR step")
@click.option("--ocr-url", help="LM Studio URL (overrides config)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def process(pdf_name: str, pages: str, clean: bool, no_ocr: bool, ocr_url: str, verbose: bool):
    """
    Process a PDF document.

    Extracts pages, enhances images, detects layout, and optionally runs OCR.
    """
    setup_logging(verbose)

    # Load configuration
    global_cfg = load_global_config(Path("config/defaults.toml"))
    if ocr_url:
        global_cfg.ocr_url = ocr_url

    # Resolve PDF
    pdf_path = resolve_pdf_path(pdf_name, global_cfg.input_dir)
    if not pdf_path.exists():
        click.echo(f"Error: PDF not found: {pdf_path}", err=True)
        raise SystemExit(1)

    # Load mission config to get layout overrides
    mission_cfg = load_mission_config(Path("config"), pdf_path.name)

    # Create pipeline config
    config = PipelineConfig(
        dpi=global_cfg.dpi,
        parallel=global_cfg.parallel,
        max_workers=global_cfg.workers,
    )
    
    # Apply global defaults (from defaults.toml)
    for key, value in global_cfg.pipeline_defaults.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Apply mission-specific overrides (from missions.toml)
    for key, value in mission_cfg.layout_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.debug(f"Applied mission override: {key}={value}")

    if errors := config.validate():
        for e in errors:
            click.echo(f"Config error: {e}", err=True)
        raise SystemExit(1)

    # Setup output
    output_dir = global_cfg.output_dir / pdf_path.stem
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    # Initialize pipeline
    try:
        pipeline = TranscriptPipeline(pdf_path, global_cfg.output_dir, config)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e

    # Parse pages
    page_numbers = parse_pages(pages or "", pipeline.page_count)
    if not page_numbers:
        click.echo("No valid pages to process.", err=True)
        raise SystemExit(1)

    # Status
    if pages:
        click.echo(f"Processing {len(page_numbers)} pages of {pipeline.page_count}")
    else:
        click.echo(f"Processing all {pipeline.page_count} pages")
    click.echo(f"Output: {output_dir}")
    click.echo(f"DPI: {config.dpi}, Parallel: {config.parallel} ({config.max_workers} workers)")
    click.echo()

    # Run image processing pipeline
    result = pipeline.process_pages(page_numbers)

    click.echo()
    click.echo(f"Image processing: {result.successful_pages}/{result.processed_pages} successful")

    if result.failed_pages > 0:
        click.echo("Failed pages:")
        for pr in result.page_results:
            if not pr.success:
                click.echo(f"  Page {pr.page_num + 1}: {pr.error}")

    # Run OCR
    if not no_ocr:
        click.echo()
        mission_cfg = load_mission_config(Path("config"), pdf_path.name)
        ocr_failures = run_ocr_pipeline(pdf_path, global_cfg, result.page_results, mission_cfg.page_offset)
        if ocr_failures > 0:
            raise SystemExit(1)

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
        click.echo(f"Error: PDF not found: {pdf_path}", err=True)
        raise SystemExit(1)

    info = get_pdf_info(pdf_path)
    click.echo(f"File: {pdf_path}")
    click.echo(f"Pages: {info['page_count']}")
    for key in ("title", "author", "creator", "producer"):
        click.echo(f"{key.title()}: {info[key] or '(none)'}")
    click.echo(f"Created: {info['creation_date'] or '(none)'}")
    click.echo(f"Modified: {info['modification_date'] or '(none)'}")


if __name__ == "__main__":
    cli()
