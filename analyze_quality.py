#!/usr/bin/env python3
"""Compare JSON output quality against HTML reference."""

import json
import re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

def parse_html_reference(html_path: Path) -> list[dict]:
    """Parse the HTML reference file into structured blocks."""
    with open(html_path, encoding='latin-1') as f:
        content = f.read()

    # Pattern: timestamp followed by optional speaker/location, then text
    # Example: "00 00 00 04 CDR Roger. Clock."
    pattern = r'(\d{2}\s+\d{2}\s+\d{2}\s+\d{2})(?:\s+([A-Z/]+(?:\s+[A-Z/]+)?))?\s+(.+?)(?=\d{2}\s+\d{2}\s+\d{2}\s+\d{2}|$)'

    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        timestamp = match.group(1)
        speaker_loc = match.group(2) or ""
        text = match.group(3).strip()

        # Clean up text (remove HTML artifacts)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if not text:
            continue

        blocks.append({
            'timestamp': timestamp,
            'speaker_location': speaker_loc.strip(),
            'text': text
        })

    return blocks

def parse_json_output(json_path: Path) -> list[dict]:
    """Parse the JSON output into structured blocks."""
    with open(json_path) as f:
        data = json.load(f)

    blocks = []
    pages = data.get('pages', {})
    for page_key in sorted(pages.keys()):
        page_data = pages[page_key]
        for block in page_data.get('blocks', []):
            if block.get('type') == 'comm':
                timestamp = block.get('timestamp', '')
                speaker = block.get('speaker', '')
                location = block.get('location', '')
                text = block.get('text', '').strip()

                speaker_loc = speaker
                if location:
                    speaker_loc = f"{speaker}/{location}" if speaker else location

                blocks.append({
                    'timestamp': timestamp,
                    'speaker': speaker,
                    'location': location,
                    'speaker_location': speaker_loc,
                    'text': text
                })

    return blocks

def text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity ratio (0-1)."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def analyze_quality():
    """Analyze quality by comparing JSON output to HTML reference."""
    html_path = Path("assets/a11tec.html")
    json_path = Path("output/AS11_TEC/AS11_TEC_merged.json")

    if not html_path.exists():
        print(f"Error: HTML reference not found: {html_path}")
        return

    if not json_path.exists():
        print(f"Error: JSON output not found: {json_path}")
        return

    print("Loading HTML reference...")
    html_blocks = parse_html_reference(html_path)

    print("Loading JSON output...")
    json_blocks = parse_json_output(json_path)

    print(f"\n{'='*70}")
    print(f"QUALITY ANALYSIS: JSON vs HTML Reference")
    print(f"{'='*70}\n")

    # Basic counts
    print(f"Total blocks:")
    print(f"  HTML reference: {len(html_blocks)}")
    print(f"  JSON output:    {len(json_blocks)}")
    print(f"  Difference:     {len(json_blocks) - len(html_blocks):+d}")
    print()

    # Create timestamp index for HTML
    html_by_ts = {b['timestamp']: b for b in html_blocks}
    json_by_ts = {b['timestamp']: b for b in json_blocks}

    # Timestamp analysis
    common_ts = set(html_by_ts.keys()) & set(json_by_ts.keys())
    html_only_ts = set(html_by_ts.keys()) - set(json_by_ts.keys())
    json_only_ts = set(json_by_ts.keys()) - set(html_by_ts.keys())

    print(f"Timestamp matching:")
    print(f"  Common timestamps:     {len(common_ts)} ({len(common_ts)/len(html_blocks)*100:.1f}%)")
    print(f"  HTML-only timestamps:  {len(html_only_ts)}")
    print(f"  JSON-only timestamps:  {len(json_only_ts)}")
    print()

    # Timestamp distribution by day
    html_days = Counter(ts.split()[0] for ts in html_by_ts.keys())
    json_days = Counter(ts.split()[0] for ts in json_by_ts.keys())

    print(f"Timestamp distribution by day:")
    print(f"  Day   HTML   JSON   Diff")
    print(f"  ---   ----   ----   ----")
    all_days = sorted(set(html_days.keys()) | set(json_days.keys()))
    for day in all_days:
        html_count = html_days.get(day, 0)
        json_count = json_days.get(day, 0)
        diff = json_count - html_count
        print(f"  {day:>3}   {html_count:>4}   {json_count:>4}   {diff:>+4}")
    print()

    # Speaker analysis (for common timestamps)
    speaker_match = 0
    speaker_mismatch = 0
    empty_speakers = 0

    for ts in common_ts:
        html_sl = html_by_ts[ts]['speaker_location']
        json_sl = json_by_ts[ts]['speaker_location']

        if not json_sl:
            empty_speakers += 1
        elif html_sl == json_sl:
            speaker_match += 1
        else:
            speaker_mismatch += 1

    if common_ts:
        print(f"Speaker/Location accuracy (for {len(common_ts)} common timestamps):")
        print(f"  Exact match:    {speaker_match:>4} ({speaker_match/len(common_ts)*100:.1f}%)")
        print(f"  Mismatch:       {speaker_mismatch:>4} ({speaker_mismatch/len(common_ts)*100:.1f}%)")
        print(f"  Empty speaker:  {empty_speakers:>4} ({empty_speakers/len(common_ts)*100:.1f}%)")
        print()

    # Text similarity analysis (for common timestamps)
    similarities = []
    low_quality = []

    for ts in common_ts:
        html_text = html_by_ts[ts]['text']
        json_text = json_by_ts[ts]['text']

        sim = text_similarity(html_text, json_text)
        similarities.append(sim)

        if sim < 0.8:
            low_quality.append((ts, sim, html_text[:60], json_text[:60]))

    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        high_quality = sum(1 for s in similarities if s >= 0.95)
        medium_quality = sum(1 for s in similarities if 0.8 <= s < 0.95)
        low_quality_count = sum(1 for s in similarities if s < 0.8)

        print(f"Text similarity (for {len(common_ts)} common timestamps):")
        print(f"  Average similarity:  {avg_sim:.1%}")
        print(f"  High quality (≥95%): {high_quality:>4} ({high_quality/len(similarities)*100:.1f}%)")
        print(f"  Medium (80-95%):     {medium_quality:>4} ({medium_quality/len(similarities)*100:.1f}%)")
        print(f"  Low quality (<80%):  {low_quality_count:>4} ({low_quality_count/len(similarities)*100:.1f}%)")
        print()

    # Show some examples of low quality matches
    if low_quality:
        print(f"Sample low quality matches (similarity < 80%):")
        for ts, sim, html_text, json_text in sorted(low_quality, key=lambda x: x[1])[:5]:
            print(f"  {ts} ({sim:.1%}):")
            print(f"    HTML: {html_text}...")
            print(f"    JSON: {json_text}...")
        print()

    # Show speaker mismatches
    speaker_errors = []
    for ts in common_ts:
        html_sl = html_by_ts[ts]['speaker_location']
        json_sl = json_by_ts[ts]['speaker_location']
        if html_sl != json_sl and json_sl:
            speaker_errors.append((ts, html_sl, json_sl))

    if speaker_errors:
        print(f"Sample speaker/location mismatches:")
        for ts, html_sl, json_sl in speaker_errors[:10]:
            print(f"  {ts}: HTML={html_sl:20s} JSON={json_sl}")
        print()

    # Empty speakers in JSON
    json_empty = [b for b in json_blocks if not b['speaker']]
    if json_empty:
        print(f"Empty speakers in JSON ({len(json_empty)} total):")
        for block in json_empty[:5]:
            print(f"  {block['timestamp']}: {block['text'][:60]}...")
        print()

    print(f"{'='*70}")

if __name__ == "__main__":
    analyze_quality()
