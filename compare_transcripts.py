#!/usr/bin/env python3
"""
Compare reference HTML transcript with generated JSON transcript.
Find missing sentences and communication blocks.
"""

import json
import re
from pathlib import Path
from html.parser import HTMLParser
from typing import List, Dict, Set, Tuple
from difflib import SequenceMatcher


def parse_html_simple(html_content: str) -> List[Dict]:
    """Parse the simple HTML format used in the reference transcript."""
    communications = []

    # Split into lines for easier processing
    lines = html_content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp pattern: DD HH MM SS (with spaces)
        timestamp_match = re.match(r'^(\d{2}\s+\d{2}\s+\d{2}\s+\d{2})\s+(.*)$', line)

        if timestamp_match:
            timestamp = timestamp_match.group(1).replace(' ', ' ')
            rest_of_line = timestamp_match.group(2)

            # Extract speaker from font tag
            speaker_match = re.search(r'<font[^>]*>([^<]+)</font>', rest_of_line)
            if speaker_match:
                speaker = speaker_match.group(1).strip()

                # Collect text from subsequent lines until next timestamp or empty line
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

                    # Remove HTML tags and collect text
                    text_line = re.sub(r'<[^>]+>', '', text_line)
                    text_line = text_line.strip()

                    if text_line and text_line != '<br><br>':
                        text_lines.append(text_line)

                    i += 1

                # Join text lines
                text = ' '.join(text_lines).strip()

                if text:
                    communications.append({
                        'timestamp': timestamp.replace(' ', ':'),
                        'speaker': speaker,
                        'text': text
                    })

                continue

        i += 1

    return communications


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and standardizing."""
    if not text:
        return ""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation variations
    text = text.strip()
    return text.lower()


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text for granular comparison."""
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+\s+', text)
    return [normalize_text(s) for s in sentences if normalize_text(s)]


def parse_html_reference(html_path: Path) -> List[Dict]:
    """Parse the reference HTML and extract communications."""
    print(f"Parsing reference HTML: {html_path}")

    # Try different encodings
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
        raise ValueError(f"Could not read file with any of: {encodings}")

    communications = parse_html_simple(html_content)

    print(f"  Found {len(communications)} communication blocks in HTML")
    return communications


def load_json_transcript(json_path: Path) -> List[Dict]:
    """Load the generated JSON transcript."""
    print(f"Loading JSON transcript: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    communications = []

    # Handle merged format with pages dict
    if isinstance(data, dict) and 'pages' in data:
        pages = data['pages']
        if isinstance(pages, dict):
            # Pages is a dict with page names as keys
            for page_name, page_data in pages.items():
                if isinstance(page_data, dict) and 'blocks' in page_data:
                    for block in page_data['blocks']:
                        if block.get('type') == 'comm':
                            communications.append({
                                'timestamp': block.get('timestamp', ''),
                                'speaker': block.get('speaker', ''),
                                'text': block.get('text', '')
                            })
        elif isinstance(pages, list):
            # Pages is a list
            for page in pages:
                if isinstance(page, dict):
                    for comm in page.get('communications', []):
                        communications.append(comm)
    elif isinstance(data, list):
        # Direct list of communications
        communications = data

    print(f"  Found {len(communications)} communication blocks in JSON")
    return communications


def build_text_index(communications: List[Dict]) -> Dict[str, List[Dict]]:
    """Build an index of normalized text to communications for fast lookup."""
    index = {}
    for comm in communications:
        text = comm.get('text', '')
        normalized = normalize_text(text)
        if normalized not in index:
            index[normalized] = []
        index[normalized].append(comm)
    return index


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, s1, s2).ratio()


def find_missing_content(html_comms: List[Dict], json_comms: List[Dict]) -> Dict:
    """Compare communications and find missing content."""
    print("\nComparing transcripts...")

    # Build indices for fast lookup
    json_text_index = build_text_index(json_comms)
    json_texts_normalized = set(json_text_index.keys())

    # Track missing communications
    missing_comms = []
    partial_matches = []

    for i, html_comm in enumerate(html_comms):
        html_text = html_comm.get('text', '')
        html_text_norm = normalize_text(html_text)

        if not html_text_norm:
            continue

        # Exact match check
        if html_text_norm in json_texts_normalized:
            continue

        # Check for partial matches (OCR variations)
        best_match_ratio = 0.0
        best_match = None

        for json_text_norm in json_texts_normalized:
            ratio = similarity_ratio(html_text_norm, json_text_norm)
            if ratio > best_match_ratio:
                best_match_ratio = ratio
                best_match = json_text_norm

        # If similarity is very high (>0.85), consider it a minor OCR variation
        if best_match_ratio > 0.85:
            partial_matches.append({
                'html_comm': html_comm,
                'similarity': best_match_ratio,
                'best_match': best_match
            })
        else:
            # This is potentially missing content
            missing_comms.append({
                'html_comm': html_comm,
                'index': i,
                'similarity': best_match_ratio,
                'best_match': best_match if best_match_ratio > 0.5 else None
            })

    # Check for sentence-level missing content in partial matches
    significant_missing_sentences = []

    for missing in missing_comms:
        html_comm = missing['html_comm']
        html_sentences = extract_sentences(html_comm.get('text', ''))

        # Check if any full sentence is missing
        for sentence in html_sentences:
            if len(sentence) < 10:  # Skip very short fragments
                continue

            found = False
            for json_text_norm in json_texts_normalized:
                if sentence in json_text_norm or similarity_ratio(sentence, json_text_norm) > 0.9:
                    found = True
                    break

            if not found:
                significant_missing_sentences.append({
                    'sentence': sentence,
                    'html_comm': html_comm,
                    'length': len(sentence)
                })

    return {
        'missing_comms': missing_comms,
        'partial_matches': partial_matches,
        'significant_missing_sentences': significant_missing_sentences
    }


def analyze_patterns(missing_comms: List[Dict]) -> Dict:
    """Analyze patterns in missing communications."""
    patterns = {
        'by_speaker': {},
        'by_timestamp_prefix': {},
        'total_missing': len(missing_comms)
    }

    for missing in missing_comms:
        html_comm = missing['html_comm']
        speaker = html_comm.get('speaker', 'UNKNOWN')
        timestamp = html_comm.get('timestamp', '')

        # Count by speaker
        if speaker not in patterns['by_speaker']:
            patterns['by_speaker'][speaker] = 0
        patterns['by_speaker'][speaker] += 1

        # Count by timestamp prefix (day/hour)
        if timestamp:
            prefix = ':'.join(timestamp.split(':')[:2])  # Get DD:HH
            if prefix not in patterns['by_timestamp_prefix']:
                patterns['by_timestamp_prefix'][prefix] = 0
            patterns['by_timestamp_prefix'][prefix] += 1

    return patterns


def print_report(html_comms: List[Dict], json_comms: List[Dict], comparison: Dict):
    """Print detailed comparison report."""
    print("\n" + "="*80)
    print("TRANSCRIPT COMPARISON REPORT")
    print("="*80)

    print(f"\nTotal communication blocks in HTML: {len(html_comms)}")
    print(f"Total communication blocks in JSON: {len(json_comms)}")
    print(f"Difference: {len(html_comms) - len(json_comms)}")

    missing_comms = comparison['missing_comms']
    partial_matches = comparison['partial_matches']
    significant_missing = comparison['significant_missing_sentences']

    print(f"\nCompletely missing communications: {len(missing_comms)}")
    print(f"Partial matches (likely OCR variations): {len(partial_matches)}")
    print(f"Significant missing sentences: {len(significant_missing)}")

    # Analyze patterns
    if missing_comms:
        patterns = analyze_patterns(missing_comms)

        print("\n" + "-"*80)
        print("PATTERNS IN MISSING CONTENT")
        print("-"*80)

        print("\nMissing by Speaker:")
        for speaker, count in sorted(patterns['by_speaker'].items(),
                                     key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {speaker}: {count}")

        print("\nMissing by Timestamp (top 10):")
        for ts_prefix, count in sorted(patterns['by_timestamp_prefix'].items(),
                                       key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {ts_prefix}:xx:xx: {count}")

    # Show significant missing communications
    if missing_comms:
        print("\n" + "-"*80)
        print("SIGNIFICANT MISSING COMMUNICATIONS")
        print("-"*80)
        print(f"\nShowing first 50 of {len(missing_comms)} missing communications:")

        for i, missing in enumerate(missing_comms[:50], 1):
            html_comm = missing['html_comm']
            print(f"\n{i}. [{html_comm.get('timestamp', 'NO TIME')}] "
                  f"{html_comm.get('speaker', 'NO SPEAKER')}")
            print(f"   Text: {html_comm.get('text', '')[:200]}...")
            if missing['best_match']:
                print(f"   Best match similarity: {missing['similarity']:.2f}")

    # Show significant missing sentences
    if significant_missing:
        print("\n" + "-"*80)
        print("SIGNIFICANT MISSING SENTENCES")
        print("-"*80)
        print(f"\nShowing first 30 of {len(significant_missing)} missing sentences:")

        # Sort by length to show most significant first
        sorted_missing = sorted(significant_missing,
                               key=lambda x: x['length'], reverse=True)

        for i, missing in enumerate(sorted_missing[:30], 1):
            html_comm = missing['html_comm']
            print(f"\n{i}. [{html_comm.get('timestamp', 'NO TIME')}] "
                  f"{html_comm.get('speaker', 'NO SPEAKER')}")
            print(f"   Missing sentence: {missing['sentence'][:300]}")

    # Check for famous quotes
    famous_quotes = [
        "that's one small step for man",
        "one giant leap for mankind",
        "houston tranquility base here",
        "the eagle has landed",
        "magnificent desolation"
    ]

    print("\n" + "-"*80)
    print("FAMOUS QUOTES CHECK")
    print("-"*80)

    json_all_text = ' '.join([normalize_text(c.get('text', ''))
                              for c in json_comms]).lower()
    html_all_text = ' '.join([normalize_text(c.get('text', ''))
                              for c in html_comms]).lower()

    for quote in famous_quotes:
        in_html = quote.lower() in html_all_text
        in_json = quote.lower() in json_all_text

        status = "✓" if in_json else "✗"
        print(f"{status} \"{quote}\"")
        if in_html and not in_json:
            print(f"  WARNING: Present in HTML but MISSING in JSON!")


def main():
    base_path = Path('/Users/jean-louis/Work/ocr_transcript_v2')
    html_path = base_path / 'assets' / 'a11tec.html'
    json_path = base_path / 'output' / 'AS11_TEC' / 'AS11_TEC_merged.json'

    # Parse reference HTML
    html_comms = parse_html_reference(html_path)

    # Load JSON transcript
    json_comms = load_json_transcript(json_path)

    # Compare
    comparison = find_missing_content(html_comms, json_comms)

    # Print report
    print_report(html_comms, json_comms, comparison)

    # Save detailed results to JSON
    output_path = base_path / 'comparison_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'html_count': len(html_comms),
            'json_count': len(json_comms),
            'missing_count': len(comparison['missing_comms']),
            'partial_matches_count': len(comparison['partial_matches']),
            'missing_communications': comparison['missing_comms'][:100],  # First 100
            'significant_missing_sentences': comparison['significant_missing_sentences'][:100]
        }, f, indent=2)

    print(f"\n\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
