"""
LM Studio OCR client (OpenAI-compatible API).

Sends enhanced page images to a local LM Studio server and returns OCR text.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np


PLAIN_OCR_PROMPT = (
    "You are a precise OCR engine. Extract all visible text from the page image. "
    "Preserve reading order top-to-bottom, left-to-right. "
    "Output plain text only with original line breaks. "
    "Do not add tags, headers, or commentary."
)

STRUCTURED_OCR_PROMPT = (
    "You are a precise OCR engine. Extract all visible text from the page image. "
    "Preserve reading order top-to-bottom, left-to-right. "
    "Output one line per logical line using TSV with 4 columns: "
    "TYPE\\tTIMESTAMP\\tSPEAKER\\tTEXT. "
    "Type codes: H (header), A (annotation), F (footer), C (comm), T (other). "
    "For H/A/F/T lines, leave TIMESTAMP and SPEAKER empty but keep tabs. "
    "Never output an empty TEXT field; use [UNK] if unreadable. "
    "Do not include a header row or any commentary."
)

SNIPPET_OCR_PROMPT = (
    "You are a precise OCR engine. Extract the exact characters from the image region. "
    "Preserve spacing as best as possible. Output plain text only. "
    "Do not add labels, guesses, or commentary."
)


@dataclass
class LMStudioOCRClient:
    base_url: str
    model: str
    timeout_s: int = 120
    max_tokens: int = 4096
    temperature: float = 0.0
    prompt: str = PLAIN_OCR_PROMPT
    image_mode: str = "auto"
    image_token: str = "auto"

    def _looks_like_ocr(self, text: str, image_token: str) -> bool:
        if not text or not text.strip():
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        if lines[0].lower().startswith("type\t") and "timestamp" in lines[0].lower():
            return False
        if image_token and any(image_token in line for line in lines):
            return False
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count < 10:
            return False
        typed = 0
        with_text = 0
        for line in lines:
            if len(line) > 1 and line[0] in "HAFCT" and line[1] == "\t":
                typed += 1
                parts = line.split("\t", 3)
                text_part = parts[3] if len(parts) >= 4 else ""
                if text_part.strip() and text_part.strip() not in ("<image>", "[UNK]"):
                    with_text += 1
        return typed > 0 and with_text > 0

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc

        return json.loads(body.decode("utf-8"))

    def ocr_image(self, image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image for OCR")

        b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
        attempts = []
        if self.image_mode in ("images", "auto"):
            if self.image_token == "auto":
                attempts = [
                    ("images", "<|vision_start|><|image_pad|><|vision_end|>"),
                    ("images", "<|image_pad|>"),
                    ("images", "<image>"),
                    ("images", "<img>"),
                    ("images", ""),
                ]
            else:
                attempts = [("images", self.image_token)]
            attempts.append(("image_url", ""))
        else:
            attempts = [("image_url", "")]

        last_error = None
        last_text = ""
        for mode, token in attempts:
            try:
                if mode == "images":
                    prompt = f"{token}\n{self.prompt}" if token else self.prompt
                    payload = {
                        "model": self.model,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                        "images": [b64],
                    }
                else:
                    payload = {
                        "model": self.model,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                                    },
                                ],
                            }
                        ],
                    }

                response = self._post_json("/v1/chat/completions", payload)
                choices = response.get("choices", [])
                if not choices:
                    last_error = RuntimeError(f"Empty OCR response: {response}")
                    continue

                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    content = "\n".join(parts)

                text = str(content).strip()
                last_text = text
                if self._looks_like_ocr(text, token):
                    return text
            except Exception as exc:
                last_error = exc

        if last_text:
            return last_text
        if last_error:
            raise last_error
        raise RuntimeError("OCR request failed with no response")
