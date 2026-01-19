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

from src.config.global_config import GlobalConfig, load_global_config, load_prompt_config
from src.config.mission_config import load_mission_config
from src.core.config import PipelineConfig
from src.core.pipeline import PageResult, TranscriptPipeline
from src.correctors.timestamp_index import GlobalTimestampIndex
from src.ocr.ocr_client import (
    COLUMN_OCR_PROMPT,
    PLAIN_OCR_PROMPT,
    TEXT_COLUMN_OCR_PROMPT,
    LMStudioOCRClient,
)
from src.ocr.ocr_parser import (
    HEADER_PAGE_ONLY_RE,
    HEADER_TAPE_ONLY_RE,
    HEADER_TAPE_PAGE_ONLY_RE,
    LOCATION_PAREN_RE,
    SPEAKER_LINE_RE,
    build_page_json,
    parse_ocr_text,
)
from src.processors.page_extractor import get_pdf_info
from src.utils.console import PipelineConsole
from src.utils.console import console as rich_console

# Initialize console manager
console = PipelineConsole()

def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    # Log strictly to file to avoid breaking the Rich Live dashboard
    logger.add("pipeline.log", rotation="10 MB", level="DEBUG" if verbose else "INFO")


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
    mission_overrides: dict[str, object] | None = None
) -> tuple[int, dict[int, dict[str, float]]]:
    """Run OCR on already processed pages."""
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
    index_path = config.output_dir / pdf_path.stem / "timestamps_index.json"
    ts_index = GlobalTimestampIndex.load(index_path)

    if mission_overrides is None:
        mission_overrides = {}
    ocr_text_column_pass = bool(
        mission_overrides.get("ocr_text_column_pass")
        or config.pipeline_defaults.get("ocr_text_column_pass")
        or False
    )
    col2_end_val = (
        mission_overrides.get("col2_end")
        or config.pipeline_defaults.get("col2_end")
        or 0.30
    )
    col2_end = float(str(col2_end_val)) if col2_end_val else 0.30
    previous_block_type = None
    tape_x = 1
    tape_y = 1
    for pr in page_results:
        if not pr.success or not pr.output:
            continue

        page_num = pr.page_num
        page_start = time.perf_counter()
        page_dir = pr.output.page_dir
        ocr_dir = pr.output.ocr_dir
        page_id = f"{pdf_path.stem}_page_{page_num + 1:04d}"

        try:
            # Load enhanced image from disk
            enhanced_path = pr.output.enhanced_image
            enhanced = cv2.imread(str(enhanced_path))
            if enhanced is None:
                raise FileNotFoundError(f"Enhanced image not found: {enhanced_path}")

            raw_text = client.ocr_image(enhanced)
            # Preserve raw OCR response for diagnostics.
            raw_text_output = raw_text
            if raw_text_output and not raw_text_output.endswith("\n"):
                raw_text_output += "\n"
            # Normalize newlines for parsing.
            text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

            heuristics_start = time.perf_counter()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            rows = parse_ocr_text(text, page_num, mission_keywords)

            # Get last timestamp from previous pages
            initial_ts = ts_index.get_last_timestamp_before(page_num)

            payload = build_page_json(
                rows, lines, page_num, page_offset,
                valid_speakers, text_replacements,
                mission_keywords, valid_locations,
                initial_ts=initial_ts,
                previous_block_type=previous_block_type
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
                    column_text = client.ocr_image_with_prompt(column_crop, text_column_prompt)
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
                        if text[:1] in ";,.)" or text[:1].islower():
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

            # Recompute page/tape metadata (ignore OCR header Page/Tape lines)
            header = payload.get("header", {})
            header["page"] = page_num + 1 + page_offset
            header["tape"] = f"{tape_x}/{tape_y}"
            payload["header"] = header
            heuristics_time = time.perf_counter() - heuristics_start

            # Update index
            page_timestamps = [
                b.get("timestamp") for b in payload.get("blocks", [])
                if b.get("type") == "comm" and b.get("timestamp")
            ]
            ts_index.add_timestamps(page_num, page_timestamps)

            (ocr_dir / f"{page_id}_ocr_raw.txt").write_text(raw_text_output, encoding="utf-8")
            (page_dir / f"{page_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            page_duration = time.perf_counter() - page_start
            timings[page_num] = {
                "ocr_s": page_duration - heuristics_time,
                "postprocess_s": heuristics_time,
                "ocr_total_s": page_duration
            }
            timing_info = (
                f"extract={pr.extract_s or 0:.3f}s "
                f"process={pr.process_s or 0:.3f}s "
                f"output={pr.output_s or 0:.3f}s "
                f"ocr={timings[page_num]['ocr_s']:.3f}s "
                f"postprocess={timings[page_num]['postprocess_s']:.3f}s "
                f"ocr_total={timings[page_num]['ocr_total_s']:.3f}s"
            )
            console.update_ocr_progress(page_num, page_duration, timing_info=timing_info)

            blocks = payload.get("blocks", [])
            if blocks:
                previous_block_type = blocks[-1].get("type")
            else:
                previous_block_type = None

            end_of_tape = any(
                b.get("type") == "meta"
                and isinstance(b.get("text"), str)
                and "END OF TAPE" in b.get("text").upper()
                for b in payload.get("blocks", [])
            )
            if end_of_tape:
                tape_x += 1
                tape_y = 1
            else:
                tape_y += 1

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

    # Create pipeline config
    config = PipelineConfig(
        dpi=global_cfg.dpi,
        parallel=global_cfg.parallel,
        max_workers=global_cfg.workers,
    )

    # Apply defaults and overrides
    for key, value in global_cfg.pipeline_defaults.items():
        if hasattr(config, key):
            setattr(config, key, value)
    for key, value in mission_cfg.layout_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    if errors := config.validate():
        for e in errors:
            print(f"Config error: {e}")
        raise SystemExit(1)

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
        mission_keywords = lexicon_cfg.get("mission_keywords") if isinstance(lexicon_cfg, dict) else None
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
            mission_cfg.layout_overrides
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
                    f"ocr={ocr_t.get('ocr_s', 0):.3f}s "
                    f"postprocess={ocr_t.get('postprocess_s', 0):.3f}s "
                    f"ocr_total={ocr_t.get('ocr_total_s', 0):.3f}s"
                )

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
