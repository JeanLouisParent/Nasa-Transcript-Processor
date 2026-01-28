"""
Merge per-page JSON outputs into a single global JSON and formatted TXT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PageBundle:
    page_num: int
    header: dict
    blocks: list[dict]
    source_path: Path


def collect_page_jsons(output_dir: Path, pdf_stem: str) -> list[PageBundle]:
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
        bundles.append(PageBundle(page_num=page_num, header=header, blocks=blocks, source_path=json_path))
    bundles.sort(key=lambda b: (b.page_num, b.source_path.name))
    return bundles


def build_global_json(pdf_stem: str, pages: list[PageBundle]) -> dict:
    page_map: dict[str, dict] = {}
    tape_x = 1
    tape_y = 1
    tape_started = False
    for page in pages:
        page_num = page.page_num
        header = dict(page.header or {})
        logical_page = header.get("page", page_num)
        if not tape_started and isinstance(logical_page, int) and logical_page >= 1:
            tape_started = True
            tape_y = 1
        if tape_started:
            header["tape"] = f"{tape_x}/{tape_y}"
        else:
            header["tape"] = None

        end_of_tape = any(
            (b.get("meta_type") == "end_of_tape")
            or ("END OF TAPE" in (b.get("text") or "").upper())
            for b in page.blocks
        )
        rest_period = header.get("page_type") == "rest_period"
        if tape_started:
            if end_of_tape or rest_period:
                tape_x += 1
                tape_y = 1
            else:
                tape_y += 1

        key = f"Page {page_num:03d}"
        page_map[key] = {
            "header": header,
            "blocks": page.blocks,
        }
    return {
        "document": pdf_stem,
        "pages": page_map,
    }


def render_transcript_text(pages: list[PageBundle]) -> str:
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
    """Render transcript as Markdown with tables."""
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


def write_global_outputs(output_dir: Path, pdf_stem: str) -> tuple[Path, Path, Path]:
    pages = collect_page_jsons(output_dir, pdf_stem)
    global_json = build_global_json(pdf_stem, pages)
    output_root = output_dir / pdf_stem
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / f"{pdf_stem}_merged.json"
    txt_path = output_root / f"{pdf_stem}_transcript.txt"
    md_path = output_root / f"{pdf_stem}_transcript.md"

    json_path.write_text(json.dumps(global_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(render_transcript_text(pages), encoding="utf-8")
    md_path.write_text(render_transcript_markdown(pages, pdf_stem), encoding="utf-8")

    return json_path, txt_path, md_path
