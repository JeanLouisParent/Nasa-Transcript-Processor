# JSON Schema Reference

Formal schema definitions for pipeline output files.

## Overview

The pipeline outputs are validated against JSON Schema (Draft 2020-12). Schema files are located in `schemas/`.

| Schema | File | Validates |
|:-------|:-----|:----------|
| Page | `page.schema.json` | Individual page JSON (`output/.../Page_NNN.json`) |
| Merged | `merged.schema.json` | Complete document (`output/.../<stem>_merged.json`) |

---

## Page Schema

### Structure

```json
{
  "header": {
    "page": 42,
    "tape": "5/3",
    "is_apollo_title": false,
    "page_type": "rest_period"  // optional
  },
  "blocks": [
    { "type": "comm", ... },
    { "type": "continuation", ... },
    ...
  ]
}
```

### Header Fields

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `page` | integer | Yes | Logical page number (with mission offset) |
| `tape` | string\|null | Yes | Tape/reel identifier (`"X/Y"` format or `null`) |
| `is_apollo_title` | boolean | Yes | True if this is a title/cover page |
| `page_type` | string | No | Special page type (e.g., `"rest_period"`) |
| `footer` | boolean | No | True if a standard footer was detected |

### Block Types

#### `comm` - Communication Block

The primary block type for dialogue.

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `type` | `"comm"` | Yes | Block type identifier |
| `timestamp` | string | No | Mission elapsed time (`"DD HH MM SS"`) |
| `speaker` | string | No | Speaker callsign (e.g., `"CDR"`, `"CC"`) |
| `location` | string | No | Station identifier (e.g., `"TRANQ"`) |
| `text` | string | Yes | Spoken content |
| `timestamp_correction` | string | No | Debug flag (see below) |

**Timestamp Correction Flags:**

| Value | Meaning |
|:------|:--------|
| `inferred_suffix` | Seconds digit inferred from context |
| `inferred_tens` | Tens digit corrected (50->10 fix) |
| `inferred_monotonic` | Adjusted to maintain monotonic order |
| `inferred_missing` | Timestamp was missing, inferred from previous |
| `out_of_order` | Timestamp is out of order (kept as-is) |
| `corrected_jump` | Large forward jump was corrected |

#### `continuation` - Continuation Block

Text continuing from a previous block or page.

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `type` | `"continuation"` | Yes | Block type identifier |
| `text` | string | Yes | Continuation text |
| `continuation_from_prev` | boolean | No | True if continues from previous page |

#### `meta` - Metadata Block

Structural markers and metadata.

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `type` | `"meta"` | Yes | Block type identifier |
| `text` | string | Yes | Metadata content |
| `meta_type` | string | No | Category (see below) |
| `timestamp` | string | No | Associated timestamp (for `lunar_rev`) |

**Meta Types:**

| Value | Trigger Pattern |
|:------|:----------------|
| `end_of_tape` | "END OF TAPE" |
| `lunar_rev` | "BEGIN/END LUNAR REV N" |
| `rest_period` | "REST PERIOD - NO COMMUNICATIONS" |
| `transcript_header` | "AIR-TO-GROUND VOICE TRANSCRIPTION" |

#### `header` - Page Header

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `type` | `"header"` | Yes | Block type identifier |
| `text` | string | Yes | Header content |

#### `footer` - Page Footer

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `type` | `"footer"` | Yes | Block type identifier |
| `text` | string | Yes | Footer content |

#### `annotation` - Annotation

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `type` | `"annotation"` | Yes | Block type identifier |
| `text` | string | Yes | Annotation content |

#### `text` - Plain Text

Unclassified text block.

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `type` | `"text"` | Yes | Block type identifier |
| `text` | string | Yes | Text content |

---

## Merged Schema

### Structure

```json
{
  "document": "AS11_TEC",
  "page_count": 423,
  "pages": [
    {
      "header": { ... },
      "blocks": [ ... ],
      "source": "output/AS11_TEC/pages/Page_001/AS11_TEC_page_0001.json"
    },
    ...
  ]
}
```

### Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `document` | string | Document stem name |
| `page_count` | integer | Total number of pages |
| `pages` | array | Array of page objects |

Each page object contains:

| Field | Type | Description |
|:------|:-----|:------------|
| `header` | object | Same as page schema header |
| `blocks` | array | Same as page schema blocks |
| `source` | string | Path to source page JSON |

---

## Validation

### Using Python

```python
import json
from jsonschema import validate, Draft202012Validator

# Load schema
with open("schemas/page.schema.json") as f:
    schema = json.load(f)

# Load and validate page
with open("output/AS11_TEC/pages/Page_001/AS11_TEC_page_0001.json") as f:
    page = json.load(f)

validate(instance=page, schema=schema)
```

### Using CLI (ajv)

```bash
# Install ajv-cli
npm install -g ajv-cli

# Validate a page
ajv validate -s schemas/page.schema.json -d "output/AS11_TEC/pages/Page_001/*.json"
```

---

## Schema Files

```
schemas/
├── page.schema.json    # Individual page validation
└── merged.schema.json  # Complete document validation
```