# Prompt Reference

This document lists the OCR and classification prompts used by the pipeline.

## File Location

Prompts are stored in `config/prompts.toml`. If the file is missing, the pipeline falls back to built-in defaults.

## Prompts

### plain_ocr_prompt

Used when `ocr_prompt = "plain"` or when the classification pass is enabled (the OCR pass is kept plain).

### structured_ocr_prompt

Used when `ocr_prompt = "structured"` and no post-process classification is enabled.

### classify_prompt

Used when `ocr_postprocess = "classify"`. This prompt uses OCR text + image and requires line-by-line tagging.

## Validation Rules

Classification output is rejected unless it:

- Preserves line count and order
- Preserves line numbering (N| prefix)
- Includes a valid tag for every line
- Includes text after the tag

Rejected outputs are written to `*_ocr_classified_rejected.txt` for debugging.
