# Quick Start Guide

> Get up and running with the NASA Transcript Processing Pipeline in 10 minutes.

## Prerequisites

1. **Python 3.10+** installed
2. **LM Studio** running a vision model:
   - Recommended: `qwen/qwen3-vl-4b`
   - Start server on `http://localhost:1234`
3. **PDF transcript** in `input/` directory

## Installation

```bash
# Clone repository
git clone <repository-url>
cd ocr_transcript_v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## First Run: Process a Transcript

```bash
# Place your PDF in input/
# Example: input/AS11_TEC.PDF

# Run the full pipeline
python main.py process AS11_TEC.PDF
```

**What happens:**
1. Extracts pages from PDF (300 DPI)
2. Enhances images (deskew, normalize, clean)
3. Runs multi-pass OCR (enhanced + raw + faint)
4. Parses OCR text into structured blocks
5. Corrects timestamps, speakers, locations, text
6. Exports to JSON and TXT

**Duration**: ~3-4 hours for a 600-page transcript

**Output**: `output/AS11_TEC/`

## Check the Results

```bash
# View merged JSON
cat output/AS11_TEC/AS11_TEC_merged.json | python -m json.tool | head -100

# View formatted transcript
head -50 output/AS11_TEC/AS11_TEC_transcript.txt

# View markdown transcript
head -50 output/AS11_TEC/AS11_TEC_transcript.md
```

## Common Workflows

### Process Specific Pages (Faster)

```bash
# Process pages 1-50 only (for testing)
python main.py process AS11_TEC.PDF --pages 1-50

# Process specific pages
python main.py process AS11_TEC.PDF --pages 100,150-160,200
```

### Fast Iteration After Config Changes

```bash
# 1. Make changes to config/missions.toml
vim config/missions.toml

# 2. Reparse from cached OCR (1-2 minutes)
python main.py reparse AS11_TEC.PDF

# 3. Post-process blocks (30 seconds)
python main.py postprocess AS11_TEC.PDF

# 4. Export to merged files (5 seconds)
python main.py export AS11_TEC.PDF

# 5. Check results
head -50 output/AS11_TEC/AS11_TEC_transcript.txt
```

### Fix Common Issues

#### Issue 1: Empty Speakers

**Symptom**: Some blocks have `"speaker": ""`

**Solution**: Check OCR text and add manual correction

```toml
# config/missions.toml
[mission.11.manual_speaker_corrections]
"00 05 41 36" = "CMP"  # Exact timestamp → speaker
```

Then reparse:
```bash
python main.py reparse AS11_TEC.PDF
```

#### Issue 2: Speaker OCR Errors

**Symptom**: Speaker "CT" should be "CMP", "IMF" should be "LMP"

**Solution**: Add OCR fix mapping

```toml
# config/defaults.toml
[pipeline_defaults.correctors.speaker_ocr_fixes]
"CT" = "CMP"
"IMF" = "LMP"
```

Then reparse:
```bash
python main.py reparse AS11_TEC.PDF
```

#### Issue 3: Text Replacement Needed

**Symptom**: OCR reads "Gunymas" but should be "Guaymas"

**Solution**: Add regex replacement

```toml
# config/missions.toml
[mission.11]
text_replacements = { "Gunymas" = "Guaymas" }
```

Then reparse:
```bash
python main.py reparse AS11_TEC.PDF
```

#### Issue 4: Invalid Location Annotations

**Symptom**: Location field contains "LAUGHING" or "GARBLE"

**Solution**: These are already filtered by default in `config/defaults.toml`:

```toml
[pipeline_defaults.correctors.invalid_location_annotations]
annotations = ["LAUGHING", "LAUGHTER", "GARBLE", "GARBLED", "PAUSE"]
```

If you need to add more, edit this list and reparse.

## Configuration Basics

### Mission Configuration

Each mission has its own section in `config/missions.toml`:

```toml
[mission.11]  # Mission 11 (Apollo 11)
file_name = "AS11_TEC.PDF"
page_offset = -2  # PDF page 3 = transcript page 1
valid_speakers = ["CDR", "CC", "CMP", "LMP", "SC", "HOUSTON"]
valid_locations = ["EAGLE", "COLUMBIA", "TRANQ"]
text_replacements = { "\\bll\\b" = "11", "CDY" = "CDR" }
```

### Global Configuration

Applies to all missions in `config/defaults.toml`:

```toml
# OCR settings
ocr_url = "http://localhost:1234"
ocr_model = "qwen/qwen3-vl-4b"
ocr_timeout = 120
dpi = 300

# Multi-pass OCR
ocr_dual_pass = true
ocr_faint_pass = true
ocr_text_column_pass = true

# Processing
parallel = true
workers = 4
```

## Output Structure

```
output/AS11_TEC/
├── AS11_TEC_merged.json       # Complete structured transcript
├── AS11_TEC_transcript.txt    # Human-readable text
├── AS11_TEC_transcript.md     # Markdown format
└── pages/
    ├── Page_001/
    │   ├── AS11_TEC_page_0001.json  # Per-page structured data
    │   ├── assets/
    │   │   ├── *_enhanced.png       # Processed image (sent to OCR)
    │   │   ├── *_raw.png            # Original render
    │   │   └── *_faint.png          # High-contrast fallback
    │   └── ocr/
    │       ├── *_ocr_raw.txt        # Primary OCR output
    │       ├── *_ocr_raw_fallback.txt
    │       └── *_ocr_faint_fallback.txt
    ├── Page_002/
    └── ...
```

## Understanding the JSON Output

### Page JSON Structure

```json
{
  "header": {
    "page": 42,
    "tape": "5/3",
    "is_apollo_title": false,
    "page_type": null
  },
  "blocks": [
    {
      "type": "comm",
      "timestamp": "04 12 33 51",
      "speaker": "CDR",
      "location": "TRANQ",
      "text": "Houston, Tranquility Base here. The Eagle has landed."
    }
  ]
}
```

### Block Types

| Type | Description | Example |
|:-----|:------------|:--------|
| `comm` | Communication | Dialogue with timestamp, speaker, text |
| `meta` | Metadata | "END OF TAPE", "BEGIN LUNAR REV 1" |
| `continuation` | Text continuation | Text without timestamp/speaker |
| `annotation` | Annotation | "(MILA REV 1)", transcriber notes |
| `header` | Page header | Repeated header content |
| `footer` | Page footer | Repeated footer content |

## Performance Tips

### Faster Processing

1. **Reduce DPI** for testing (lower quality):
   ```toml
   dpi = 200  # Instead of 300
   ```

2. **Process subset of pages**:
   ```bash
   python main.py process AS11_TEC.PDF --pages 1-50
   ```

3. **Disable multi-pass OCR** (single pass):
   ```toml
   ocr_dual_pass = false
   ocr_faint_pass = false
   ocr_text_column_pass = false
   ```

4. **Use more workers** (if you have CPU cores):
   ```toml
   workers = 8  # Instead of 4
   ```

### Remote OCR Server

If running OCR on a different machine:

```bash
python main.py process AS11_TEC.PDF --ocr-url http://192.168.1.100:1234
```

Or configure globally:

```toml
# config/defaults.toml
ocr_url = "http://192.168.1.100:1234"
```

## Troubleshooting

### OCR Server Connection Refused

**Symptom**: `ConnectionRefusedError: [Errno 61] Connection refused`

**Solution**:
1. Check LM Studio is running
2. Check server is on port 1234
3. Try: `curl http://localhost:1234/v1/models`

### Out of Memory

**Symptom**: Process killed during OCR

**Solution**:
1. Reduce `workers` in config
2. Process smaller page ranges
3. Use smaller OCR model
4. Close other applications

### Image Processing Errors

**Symptom**: `cv2.error` or processing failures

**Solution**:
1. Check PDF is not corrupted: `python main.py info AS11_TEC.PDF`
2. Try processing single page: `--pages 1`
3. Check PDF has text content (not just images)

### Timestamp Issues

**Symptom**: Many timestamps "inferred_monotonic"

**Check**:
```bash
# Look at OCR text quality
cat output/AS11_TEC/pages/Page_001/ocr/*_ocr_raw.txt
```

**Solutions**:
- If OCR quality is poor: adjust image processing settings
- If timestamps are garbled: check patterns in `src/ocr/parsing/patterns.py`
- If specific day corrections needed: add to day correction logic

## Next Steps

1. **Read the full documentation**:
   - [Architecture](docs/ARCHITECTURE.md) — System design
   - [Pipeline](docs/PIPELINE.md) — Processing stages
   - [Configuration](docs/CONFIGURATION.md) — All settings
   - [Post-Processing](docs/POST_PROCESSING.md) — Correction algorithms

2. **Improve output quality**:
   - Review transcript output for issues
   - Add corrections to `config/missions.toml`
   - Use fast iteration (reparse) to test changes

3. **Customize for your mission**:
   - Add mission config in `config/missions.toml`
   - Define valid_speakers, valid_locations
   - Add mission-specific text_replacements

4. **Contribute**:
   - Report issues
   - Submit improvements
   - Share your mission configs

## Support

For detailed documentation, see:
- [README.md](README.md) — Project overview
- [docs/](docs/) — Full documentation
- [CHANGELOG.md](CHANGELOG.md) — Recent improvements

## Example Session

```bash
# Full workflow example

# 1. Initial processing
python main.py process AS11_TEC.PDF --pages 1-100

# 2. Check results
head -100 output/AS11_TEC/AS11_TEC_transcript.txt

# 3. Notice issues: empty speakers, text errors
# Edit config/missions.toml:
[mission.11.manual_speaker_corrections]
"00 04 49 33" = "CC"

[mission.11.text_replacements]
"Gunymas" = "Guaymas"

# 4. Fast reparse (1-2 min)
python main.py reparse AS11_TEC.PDF

# 5. Post-process
python main.py postprocess AS11_TEC.PDF

# 6. Export
python main.py export AS11_TEC.PDF

# 7. Re-check results
head -100 output/AS11_TEC/AS11_TEC_transcript.txt

# 8. Success! Issues corrected
```

Happy transcribing! 🚀
