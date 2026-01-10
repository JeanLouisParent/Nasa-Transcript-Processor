#!/usr/bin/env python3
"""
NASA Transcript Processing Pipeline - CLI Interface.

This module provides the command-line interface for processing NASA
mission transcript PDFs.

Usage:
    python main.py process AS11_TEC.PDF --output output/
    python main.py process AS11_TEC.PDF --pages 1-50 --output output/
    python main.py info AS11_TEC.PDF
    python main.py config show

For AI Agents:
    - Main entry point for the pipeline
    - Uses click for CLI argument parsing
    - Supports page range selection with --pages
    - Parallel processing enabled by default
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

import click
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.page_extractor import get_pdf_info
from src.pipeline import TranscriptPipeline


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging for the CLI.

    Args:
        verbose: Enable verbose output
        debug: Enable debug-level logging
    """
    # Remove default handler
    logger.remove()

    # Set level based on flags
    if debug:
        level = "DEBUG"
    elif verbose:
        level = "INFO"
    else:
        level = "WARNING"

    # Add console handler with formatting
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
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
    "--input-dir",
    type=click.Path(path_type=Path),
    default=Path("input"),
    help="Input directory for PDFs (default: input/)"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Output directory (default: output/)"
)
@click.option(
    "--pages", "-p",
    type=str,
    default=None,
    help="Page range to process (e.g., '1-50', '10', or '10,12,14-16'). Default: all pages."
)
@click.option(
    "--workers", "-w",
    type=int,
    default=4,
    help="Number of parallel workers (default: 4)"
)
@click.option(
    "--no-parallel",
    is_flag=True,
    default=False,
    help="Disable parallel processing"
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="Output resolution in DPI (default: 300)"
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode with verbose logging"
)
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Remove existing output directory before processing"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output"
)
def process(
    pdf_name: str,
    input_dir: Path,
    output: Path,
    pages: Optional[str],
    workers: int,
    no_parallel: bool,
    dpi: int,
    debug: bool,
    clean: bool,
    verbose: bool
):
    """
    Process a PDF document.

    Extracts each page, enhances the image, detects layout blocks,
    and generates output files.

    Example:
        python main.py process AS11_TEC.PDF --pages 1-10 --output ./results
    """
    setup_logging(verbose=verbose, debug=debug)

    # Resolve PDF path
    pdf_path = resolve_pdf_path(pdf_name, input_dir)
    if not pdf_path.exists():
        click.echo(f"Error: PDF not found: {pdf_path}", err=True)
        raise click.Abort()

    # Create configuration
    config = PipelineConfig(
        dpi=dpi,
        parallel=not no_parallel,
        max_workers=workers,
        debug=debug
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
    click.echo(f"DPI: {dpi}")
    click.echo(f"Parallel: {not no_parallel} (workers: {workers})")
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


@cli.command()
@click.argument("pdf_name", type=str)
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path),
    default=Path("input"),
    help="Input directory for PDFs (default: input/)"
)
def info(pdf_name: str, input_dir: Path):
    """
    Display information about a PDF document.

    Shows page count, metadata, and page dimensions.
    """
    setup_logging()

    pdf_path = resolve_pdf_path(pdf_name, input_dir)
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


@cli.group()
def config():
    """
    Configuration management commands.
    """
    pass


@config.command("show")
def config_show():
    """
    Display current default configuration.
    """
    setup_logging()

    click.echo("Default Configuration")
    click.echo("=" * 40)

    config = DEFAULT_CONFIG

    click.echo(f"DPI: {config.dpi}")
    click.echo(f"Output format: {config.output_format}")
    click.echo(f"Parallel: {config.parallel}")
    click.echo(f"Max workers: {config.max_workers}")
    click.echo("")
    click.echo(f"Target size: {config.target_width}x{config.target_height} px")
    click.echo(f"Margin: {config.margin_px} px")
    click.echo("")
    click.echo(f"CLAHE clip limit: {config.clahe_clip_limit}")
    click.echo(f"Bilateral filter d: {config.bilateral_d}")
    click.echo("")
    click.echo(f"Min block area: {config.min_block_area}")
    click.echo(f"Max block area ratio: {config.max_block_area_ratio}")
    click.echo("")
    click.echo(f"Column boundaries: {config.col1_end}, {config.col2_end}")
    click.echo(f"Header ratio: {config.header_ratio}")


@config.command("save")
@click.argument("output_path", type=click.Path(path_type=Path))
def config_save(output_path: Path):
    """
    Save default configuration to a YAML file.

    This creates a configuration file that can be customized
    for different missions.
    """
    setup_logging()

    config = DEFAULT_CONFIG
    config.to_yaml(output_path)

    click.echo(f"Configuration saved to: {output_path}")


@config.command("validate")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def config_validate(config_path: Path):
    """
    Validate a configuration YAML file.
    """
    setup_logging()

    try:
        config = PipelineConfig.from_yaml(config_path)
        errors = config.validate()

        if errors:
            click.echo("Configuration errors:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            sys.exit(1)
        else:
            click.echo("Configuration is valid.")

    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
