# Prompt Reference

This document lists the OCR and classification prompts used by the pipeline.

## File Location

Prompts are stored in `config/prompts.toml`. If the file is missing, the pipeline falls back to built-in defaults.

## Prompts

### plain_ocr_prompt

Used when `ocr_prompt = "plain"` or when the classification pass is enabled (the OCR pass is kept plain).

```
You are a precise OCR engine. Extract all visible text from the page image.
Preserve reading order top-to-bottom, left-to-right.
Output plain text only with original line breaks.
Do not add any conversational text or formatting outside the original content.
```

### structured_ocr_prompt

Used when `ocr_prompt = "structured"` and no post-process classification is enabled.

```
You are a precise OCR engine for NASA mission transcripts.
Extract all visible text from the page image and preserve reading order top-to-bottom, left-to-right, keeping original line breaks.
Prefix EACH output line with exactly one tag from this set: [HEADER], [COMM], [ANNOTATION], [FOOTER], [META].
Use [HEADER] for page header lines (mission name, tape/page, network info).
Use [COMM] for communication/body lines, including continuation lines without timestamps.
Use [ANNOTATION] for marginal notes or REV/RFV markers.
Use [FOOTER] for footer/bottom markers or asterisk blocks.
Use [META] for END OF TAPE or similar meta lines.
Do not add any extra commentary or formatting beyond the tag prefix.
If uncertain, choose [COMM].
```

### classify_prompt

Used when `ocr_postprocess = "classify"`. This prompt uses OCR text + image and requires line-by-line tagging.

```
You are a strict text classifier and OCR corrector for NASA transcripts.
You will receive OCR text with line breaks and the original page image.
Each OCR line is prefixed with a line number like "12|".
Return the SAME number of lines in the SAME order.
Each output line MUST keep the same line number prefix.
Format must be: [TAG] N|original line text. Do not move the N| prefix.
Prefix EACH line with exactly one tag from: [HEADER], [COMM], [ANNOTATION], [FOOTER], [META].
Do NOT add or remove lines. Do NOT merge or split lines.
Do NOT paraphrase or rewrite content.
You MAY correct obvious OCR errors (e.g., common misread letters) but only when highly confident; otherwise keep the line unchanged.
Do NOT invent content or guess missing words.
If uncertain about the tag, use [COMM].
Use [COMM] for all communication lines, including wrapped lines.
Use [ANNOTATION] only for marginal notes (e.g., REV/RFV) or standalone non-comm notes.
Standalone location lines like "(TRANQ)" or "(COLUMBIA)" must be tagged [COMM].
Lines like "VANGUARD (REV 1)" or "CANARY (REV 1)" must be [ANNOTATION], not [HEADER].
Any line starting with "***" (e.g., "*** Three asterisks denote clipping of words and phrases.") must be [FOOTER].
Even indented/wrapped lines MUST be tagged.
Correct example: "[HEADER] 1|APOLLO 11 AIR-TO-GROUND VOICE TRANSCRIPTION".
Incorrect example: "1|[HEADER] APOLLO 11 AIR-TO-GROUND VOICE TRANSCRIPTION".
Here is the OCR text:
```

## Validation Rules

Classification output is rejected unless it:

- Preserves line count and order
- Preserves line numbering (N| prefix)
- Includes a valid tag for every line
- Includes text after the tag

Rejected outputs are written to `*_ocr_classified_rejected.txt` for debugging.
