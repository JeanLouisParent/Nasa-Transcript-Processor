#!/usr/bin/env python3
"""
Fast comparison of reference HTML transcript with generated JSON transcript.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set


def parse_html_simple(html_content: str) -> List[Dict]:
    """Parse the simple HTML format used in the reference transcript."""
    communications = []
    lines = html_content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp pattern: DD HH MM SS (with spaces)
        timestamp_match = re.match(r'^(\d{2}\s+\d{2}\s+\d{2}\s+\d{2})\s+(.*)$', line)

        if timestamp_match:
            timestamp = timestamp_match.group(1)
            rest_of_line = timestamp_match.group(2)

            # Extract speaker from font tag
            speaker_match = re.search(r'<font[^>]*>([^<]+)</font>', rest_of_line)
            if speaker_match:
                speaker = speaker_match.group(1).strip()

                # Collect text from subsequent lines
                text_lines = []
                i += 1

                while i < len(lines):
                    text_line = lines[i].strip()

                    # Stop at next timestamp or section marker
                    if re.match(r'^\d{2}\s+\d{2}\s+\d{2}\s+\d{2}', text_line):
                        break
                    if text_line.startswith('(GOSS NET'):
                        break
                    if text_line == '<br><br>':
                        break

                    # Remove HTML tags
                    text_line = re.sub(r'<[^>]+>', '', text_line)
                    text_line = text_line.strip()

                    if text_line:
                        text_lines.append(text_line)

                    i += 1

                text = ' '.join(text_lines).strip()

                if text:
                    # Convert timestamp format
                    ts = timestamp.replace(' ', ':')
                    communications.append({
                        'timestamp': ts,
                        'speaker': speaker,
                        'text': text
                    })

                continue

        i += 1

    return communications


def normalize_timestamp(ts: str) -> str:
    """Normalize timestamp to common format."""
    # Replace any spaces with colons
    ts = ts.replace(' ', ':')
    # Remove any leading zeros from parts (optional)
    return ts


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove common punctuation
    text = text.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    text = text.replace('--', ' ').replace('-', ' ')
    return text.strip()


def load_json_transcript(json_path: Path) -> List[Dict]:
    """Load the generated JSON transcript."""
    print(f"Loading JSON transcript: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    communications = []

    if isinstance(data, dict) and 'pages' in data:
        pages = data['pages']
        if isinstance(pages, dict):
            for page_name, page_data in pages.items():
                if isinstance(page_data, dict) and 'blocks' in page_data:
                    for block in page_data['blocks']:
                        if block.get('type') == 'comm':
                            communications.append({
                                'timestamp': block.get('timestamp', ''),
                                'speaker': block.get('speaker', ''),
                                'text': block.get('text', '')
                            })

    print(f"  Found {len(communications)} communication blocks in JSON")
    return communications


def parse_html_reference(html_path: Path) -> List[Dict]:
    """Parse the reference HTML and extract communications."""
    print(f"Parsing reference HTML: {html_path}")

    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    html_content = None

    for encoding in encodings:
        try:
            with open(html_path, 'r', encoding=encoding) as f:
                html_content = f.read()
            print(f"  Successfully read with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if html_content is None:
        raise ValueError(f"Could not read file with any encoding")

    communications = parse_html_simple(html_content)
    print(f"  Found {len(communications)} communication blocks in HTML")
    return communications


def build_lookup_index(communications: List[Dict]) -> Set[str]:
    """Build a set of normalized text for fast lookup."""
    return {normalize_text(c.get('text', '')) for c in communications}


def compare_transcripts(html_comms: List[Dict], json_comms: List[Dict]) -> Dict:
    """Fast comparison of transcripts."""
    print("\nBuilding lookup indices...")

    # Build normalized text index for JSON
    json_normalized = build_lookup_index(json_comms)

    print("Comparing transcripts...")

    # Find missing communications
    missing = []
    partial_missing = []

    for i, html_comm in enumerate(html_comms):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(html_comms)} HTML communications...")

        html_text = html_comm.get('text', '')
        html_norm = normalize_text(html_text)

        if not html_norm:
            continue

        # Check for exact normalized match
        if html_norm in json_normalized:
            continue

        # Check if any significant substring is present (for partial matches)
        words = html_norm.split()
        if len(words) > 5:
            # Check if we can find a good substring match
            substring_found = False
            for json_norm in json_normalized:
                if len(words) > 8:
                    # For longer texts, check if most words are present
                    matching_words = sum(1 for word in words if word in json_norm)
                    if matching_words >= len(words) * 0.7:
                        substring_found = True
                        break

            if substring_found:
                partial_missing.append(html_comm)
                continue

        # This is a missing communication
        missing.append(html_comm)

    print(f"  Completed comparison.")

    return {
        'missing': missing,
        'partial_missing': partial_missing
    }


def check_famous_quotes(html_comms: List[Dict], json_comms: List[Dict]):
    """Check if famous quotes are present."""
    famous_quotes = [
        ("that's one small step for man", "Armstrong's moon landing quote (part 1)"),
        ("one giant leap for mankind", "Armstrong's moon landing quote (part 2)"),
        ("houston tranquility base here", "Landing announcement (part 1)"),
        ("the eagle has landed", "Landing announcement (part 2)"),
        ("magnificent desolation", "Aldrin's moon description"),
        ("beautiful beautiful magnificent desolation", "Aldrin's full description"),
    ]

    json_all_text = ' '.join([c.get('text', '').lower() for c in json_comms])
    html_all_text = ' '.join([c.get('text', '').lower() for c in html_comms])

    print("\n" + "="*80)
    print("FAMOUS QUOTES CHECK")
    print("="*80)

    for quote, description in famous_quotes:
        in_html = quote.lower() in html_all_text
        in_json = quote.lower() in json_all_text

        status = "✓" if in_json else "✗"
        print(f"\n{status} {description}")
        print(f"   Quote: \"{quote}\"")
        print(f"   In HTML: {in_html}")
        print(f"   In JSON: {in_json}")

        if in_html and not in_json:
            print(f"   *** WARNING: Present in HTML but MISSING in JSON! ***")


def analyze_patterns(missing: List[Dict]) -> Dict:
    """Analyze patterns in missing communications."""
    by_speaker = {}
    by_hour = {}

    for comm in missing:
        speaker = comm.get('speaker', 'UNKNOWN')
        timestamp = comm.get('timestamp', '')

        if speaker not in by_speaker:
            by_speaker[speaker] = 0
        by_speaker[speaker] += 1

        if timestamp:
            # Extract hour from timestamp DD:HH:MM:SS
            parts = timestamp.split(':')
            if len(parts) >= 2:
                hour_key = f"{parts[0]}:{parts[1]}"
                if hour_key not in by_hour:
                    by_hour[hour_key] = 0
                by_hour[hour_key] += 1

    return {
        'by_speaker': by_speaker,
        'by_hour': by_hour
    }


def print_report(html_comms: List[Dict], json_comms: List[Dict], comparison: Dict):
    """Print comparison report."""
    missing = comparison['missing']
    partial_missing = comparison['partial_missing']

    print("\n" + "="*80)
    print("TRANSCRIPT COMPARISON REPORT")
    print("="*80)

    print(f"\nTotal communication blocks in HTML: {len(html_comms)}")
    print(f"Total communication blocks in JSON: {len(json_comms)}")
    print(f"Difference: {len(html_comms) - len(json_comms)}")

    print(f"\nCompletely missing communications: {len(missing)}")
    print(f"Partial/variant matches: {len(partial_missing)}")

    if missing:
        patterns = analyze_patterns(missing)

        print("\n" + "-"*80)
        print("PATTERNS IN MISSING CONTENT")
        print("-"*80)

        print("\nTop speakers with missing communications:")
        for speaker, count in sorted(patterns['by_speaker'].items(),
                                     key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {speaker}: {count}")

        print("\nTop time periods with missing communications:")
        for hour, count in sorted(patterns['by_hour'].items(),
                                 key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {hour}:xx:xx: {count}")

        print("\n" + "-"*80)
        print("SAMPLE MISSING COMMUNICATIONS")
        print("-"*80)
        print(f"\nShowing first 30 of {len(missing)} missing communications:\n")

        for i, comm in enumerate(missing[:30], 1):
            print(f"{i}. [{comm.get('timestamp', 'NO TIME')}] {comm.get('speaker', 'NO SPEAKER')}")
            text = comm.get('text', '')
            if len(text) > 150:
                text = text[:150] + "..."
            print(f"   {text}\n")

    # Save detailed results
    output_path = Path('/Users/jean-louis/Work/ocr_transcript_v2/comparison_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'html_count': len(html_comms),
            'json_count': len(json_comms),
            'missing_count': len(missing),
            'partial_missing_count': len(partial_missing),
            'missing_communications': missing[:200],  # First 200
            'patterns': analyze_patterns(missing) if missing else {}
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")


def main():
    base_path = Path('/Users/jean-louis/Work/ocr_transcript_v2')
    html_path = base_path / 'assets' / 'a11tec.html'
    json_path = base_path / 'output' / 'AS11_TEC' / 'AS11_TEC_merged.json'

    # Parse reference HTML
    html_comms = parse_html_reference(html_path)

    # Load JSON transcript
    json_comms = load_json_transcript(json_path)

    # Compare
    comparison = compare_transcripts(html_comms, json_comms)

    # Print report
    print_report(html_comms, json_comms, comparison)

    # Check famous quotes
    check_famous_quotes(html_comms, json_comms)


if __name__ == '__main__':
    main()
