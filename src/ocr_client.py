"""
LM Studio OCR client (OpenAI-compatible API).

Sends enhanced page images to a local LM Studio server and returns OCR text.
"""

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger

PLAIN_OCR_PROMPT = (
    "You are a precise OCR engine. Extract all visible text from the page image. "
    "Preserve reading order top-to-bottom, left-to-right. "
    "Output plain text only with original line breaks. "
    "Do not add tags, headers, or commentary."
)

# Image tokens to try for different vision models
DEFAULT_IMAGE_TOKENS = [
    "<|vision_start|><|image_pad|><|vision_end|>",
    "<|image_pad|>",
    "<image>",
    "<img>",
    "",
]


class OCRError(Exception):
    """Base exception for OCR errors."""
    pass


class OCRConnectionError(OCRError):
    """Connection to OCR server failed."""
    pass


class OCRResponseError(OCRError):
    """Invalid response from OCR server."""
    pass


@dataclass
class LMStudioOCRClient:
    """Client for LM Studio vision models."""

    base_url: str
    model: str = "qwen3-vl-4b"
    timeout_s: int = 120
    max_tokens: int = 4096
    temperature: float = 0.0
    prompt: str = PLAIN_OCR_PROMPT
    image_tokens: list[str] = field(default_factory=lambda: DEFAULT_IMAGE_TOKENS.copy())

    def _post_json(self, path: str, payload: dict) -> dict:
        """POST JSON to the API and return response."""
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OCRResponseError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise OCRConnectionError(f"Connection failed: {exc.reason}") from exc
        except TimeoutError:
            raise OCRConnectionError(f"Request timed out after {self.timeout_s}s") from None

    def _extract_text(self, response: dict) -> str:
        """Extract text content from API response."""
        choices = response.get("choices", [])
        if not choices:
            raise OCRResponseError(f"Empty response: {response}")

        content = choices[0].get("message", {}).get("content", "")

        # Handle list content (some models return structured content)
        if isinstance(content, list):
            parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "\n".join(parts)

        return str(content).strip()

    def _is_valid_ocr(self, text: str) -> bool:
        """Check if text looks like valid OCR output."""
        if not text:
            return False
        # Must have some alphabetic content
        alpha_count = sum(1 for c in text if c.isalpha())
        return alpha_count >= 2

    def ocr_image(self, image: np.ndarray) -> str:
        """
        Perform OCR on an image.

        Args:
            image: Grayscale or BGR image as numpy array

        Returns:
            Extracted text

        Raises:
            OCRError: If OCR fails
        """
        # Encode image
        # Use JPEG with quality 85 to reduce payload size significantly
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        if not success:
            raise OCRError("Failed to encode image") from None

        b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

        # Try OpenAI-style image_url format first (standard for modern LM Studio/vLLM)
        last_error = None
        try:
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
            }
            response = self._post_json("/v1/chat/completions", payload)
            text = self._extract_text(response)
            if self._is_valid_ocr(text):
                logger.debug("OCR successful using standard OpenAI vision format")
                return text
        except OCRError as exc:
            last_error = exc

        # Fallback to manual token injection for older models/backends
        for token in self.image_tokens:
            try:
                prompt = f"{token}\n{self.prompt}" if token else self.prompt
                payload = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "images": [b64],
                }

                response = self._post_json("/v1/chat/completions", payload)
                text = self._extract_text(response)

                if self._is_valid_ocr(text):
                    logger.debug(f"OCR successful using token format: {token!r}")
                    return text
            except OCRError as exc:
                last_error = exc

        if last_error:
            raise last_error from None
        raise OCRError("OCR failed: no valid text extracted")
