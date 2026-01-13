import csv
import json
import difflib
import sys
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.ocr_parser import build_page_json, parse_ocr_text
from src.config.mission_config import load_mission_config
from src.config.global_config import load_global_config

def normalize_text(text):
    if not text: return ""
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()

def load_ground_truth(csv_path):
    gt_data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=';', quotechar='"')
        for row in reader:
            if len(row) < 5: continue
            ts = row[1].strip()
            gt_data[ts] = {"speaker": row[2].strip(), "text": row[4].strip()}
    return gt_data

def run_all():
    csv_path = Path("assets/a11tec.csv")
    json_dir = Path("output/AS11_TEC")
    defaults_path = Path("config/defaults.toml")
    
    if not csv_path.exists() or not json_dir.exists():
        print("Data missing.")
        return

    # Load Configs
    global_cfg = load_global_config(defaults_path)
    mission_cfg = load_mission_config(Path("config"), "AS11_TEC.PDF")
    
    valid_speakers = mission_cfg.layout_overrides.get("valid_speakers")
    
    # Merge replacements
    global_parser = global_cfg.pipeline_defaults.get("parser", {})
    if isinstance(global_parser, dict):
        global_replacements = global_parser.get("text_replacements", {})
    else:
        global_replacements = {}
        
    mission_replacements = mission_cfg.layout_overrides.get("text_replacements", {})
    text_replacements = {**global_replacements, **mission_replacements}
    
    mission_keywords = global_cfg.pipeline_defaults.get("lexicon", {}).get("mission_keywords")
    
    gt_data = load_ground_truth(csv_path)
    
    discrepancies = []
    total_checked = 0
    total_perfect = 0
    
    print(f"Reprocessing with mission_keywords: {mission_keywords}")

    for page_dir in sorted(json_dir.glob("Page_*")):
        raw_txts = list(page_dir.glob("*_ocr_raw.txt"))
        if not raw_txts: continue
        
        raw_txt_path = raw_txts[0]
        json_path = raw_txt_path.with_name(raw_txt_path.stem.replace("_ocr_raw", "") + ".json")
        try:
            page_idx = int(page_dir.name.split("_")[1]) - 1
        except ValueError:
            continue
        
        human_page_num = page_idx + 1 + mission_cfg.page_offset

        text = raw_txt_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Reprocess with latest code and keywords
        rows = parse_ocr_text(text, page_idx, mission_keywords)
        payload = build_page_json(rows, lines, page_idx, mission_cfg.page_offset, valid_speakers, text_replacements, mission_keywords)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        
        # Compare
        for block in payload.get("blocks", []):
            if block.get("type") != "comm": continue
            ts = block.get("timestamp")
            if not ts or ts not in gt_data: continue
            
            total_checked += 1
            ocr_text = block.get("text", "")
            ocr_speaker = block.get("speaker", "")
            gt_entry = gt_data[ts]
            
            norm_ocr = normalize_text(ocr_text)
            norm_gt = normalize_text(gt_entry["text"])
            ratio = difflib.SequenceMatcher(None, norm_ocr, norm_gt).ratio()
            
            if ratio >= 0.98 and ocr_speaker == gt_entry["speaker"]:
                total_perfect += 1
            else:
                discrepancies.append({
                    "page": human_page_num,
                    "timestamp": ts, "json_speaker": ocr_speaker, "gt_speaker": gt_entry["speaker"],
                    "json_text": ocr_text, "gt_text": gt_entry["text"], "ratio": ratio
                })

    discrepancies.sort(key=lambda x: x["timestamp"])
    
    with open("report_comparison_v5.txt", "w", encoding="utf-8") as f:
        f.write(f"Comparison Report V5 (Keywords + Contractions + Program Codes)\n")
        f.write(f"Total blocks checked: {total_checked}\n")
        f.write(f"Matches (>98%): {total_perfect} ({total_perfect/total_checked*100:.1f}%)\n\n")
        for d in discrepancies:
            f.write(f"--- Page {d['page']} | {d['timestamp']} ---\n")
            if d['json_speaker'] != d['gt_speaker']:
                f.write(f"SPEAKER MISMATCH: OCR='{d['json_speaker']}' vs GT='{d['gt_speaker']}'\n")
            if d['ratio'] < 0.98:
                f.write(f"TEXT DIFF ({d['ratio']:.2f}):\n")
                f.write(f"  OCR: {d['json_text']}\n")
                f.write(f"   GT: {d['gt_text']}\n")
            f.write("\n")
    print(f"Report report_comparison_v5.txt generated. Accuracy: {total_perfect/total_checked*100:.1f}%")

if __name__ == "__main__":
    run_all()
