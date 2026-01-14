"""
Script to merge all page outputs into consolidated files.
"""

import json
from pathlib import Path
import re

def merge_outputs():
    output_dir = Path("output/AS11_TEC")
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist.")
        return

    # Find all page directories
    page_dirs = sorted(list(output_dir.glob("Page_*")), key=lambda x: int(x.name.split("_")[1]) if "_" in x.name else 0)
    print(f"Found {len(page_dirs)} page directories.")

    full_txt_path = output_dir / "AS11_TEC_FULL.txt"
    full_json_path = output_dir / "AS11_TEC_FULL.json"

    all_blocks = []
    
    with open(full_txt_path, "w", encoding="utf-8") as f_txt:
        for page_dir in page_dirs:
            # Extract page number from directory name
            try:
                page_num = int(page_dir.name.split("_")[1])
            except (ValueError, IndexError):
                continue

            # Merge Raw Text
            txt_file = list(page_dir.glob("*_ocr_raw.txt"))
            if txt_file:
                content = txt_file[0].read_text(encoding="utf-8")
                f_txt.write(f"--- PAGE {page_num} ---\n")
                f_txt.write(content)
                f_txt.write("\n\n")

            # Merge JSON Blocks
            json_file = list(page_dir.glob("*.json"))
            # Filter out the consolidated file if it already exists
            json_file = [jf for jf in json_file if jf.name != "AS11_TEC_FULL.json"]
            
            if json_file:
                try:
                    data = json.loads(json_file[0].read_text(encoding="utf-8"))
                    # Add page metadata to each block if missing
                    page_meta = data.get("header", {})
                    
                    for block in data.get("blocks", []):
                        block["page_number"] = page_num
                        block["tape_number"] = page_meta.get("tape")
                        all_blocks.append(block)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for page {page_num}")

    # Write consolidated JSON
    final_structure = {
        "mission": "Apollo 11",
        "document": "Technical Air-to-Ground Voice Transcription",
        "total_blocks": len(all_blocks),
        "transcript": all_blocks
    }
    
    with open(full_json_path, "w", encoding="utf-8") as f_json:
        json.dump(final_structure, f_json, indent=2, ensure_ascii=False)

    print(f"Successfully merged to:")
    print(f"  - {full_txt_path}")
    print(f"  - {full_json_path}")

if __name__ == "__main__":
    merge_outputs()