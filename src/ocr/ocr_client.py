"""
LM Studio OCR client (OpenAI-compatible API).

Sends enhanced page images to a local LM Studio server and returns OCR text.
"""

import base64
import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

PLAIN_OCR_PROMPT = (
    "You are a precise OCR engine. Extract all visible text from the page image. "
    "Preserve reading order top-to-bottom, left-to-right, keeping original line breaks. "
    "Each transcript line spans multiple columns (timestamp, speaker, text). Read the full line across the page. "
    "Do not stop at the speaker column; include the rightmost text for each line. "
    "You may apply minimal corrections to obvious OCR artifacts (e.g., G() -> GO, O/0, I/1) but only when highly confident. "
    "Ensure sentences read sensibly without inventing or adding any words. "
    "Do NOT hallucinate or guess missing content. "
    "Output plain text only with original line breaks. "
    "Do not add any conversational text or formatting outside the original content."
)

COLUMN_OCR_PROMPT = (
    "You are a precise OCR engine for NASA mission transcripts. "
    "Each line is laid out as columns: timestamp (left), speaker (middle), text (right). "
    "Read full lines across the page, do not stop at column boundaries. "
    "Preserve reading order top-to-bottom, left-to-right, keeping original line breaks. "
    "Output plain text only with original line breaks. "
    "Do not add any conversational text or formatting outside the original content."
)

TEXT_COLUMN_OCR_PROMPT = (
    "You are a precise OCR engine. "
    "The image is a cropped right-side text column of a transcript page. "
    "Extract ONLY the visible text in that column. "
    "Return one line per transcript line, preserving order top-to-bottom. "
    "Do not add timestamps or speakers. "
    "Output plain text only with original line breaks."
)


class OCRError(Exception):
    """Base exception for OCR errors."""
    pass


class OCRConnectionError(OCRError):
    """Failed to connect to OCR server."""
    pass


class OCRResponseError(OCRError):
    """Received an invalid or empty response from OCR server."""
    pass


@dataclass
class LMStudioOCRClient:
    """
    Client for LM Studio vision-enabled OCR models using an OpenAI-compatible API.
    
    This client handles image encoding, payload preparation, and communication
    with the local LLM server.
    """
    base_url: str = "http://localhost:1234"
    model: str = "qwen3-vl-4b"
    timeout_s: int = 120
    max_tokens: int = 4096
    prompt: str = PLAIN_OCR_PROMPT
    verify_ssl: bool = False

    def __post_init__(self):
        """Initializes API endpoints and SSL context."""
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.api_url = f"{self.base_url}/v1/chat/completions"
        else:
            self.api_url = f"{self.base_url}/chat/completions"

        # Create unverified SSL context if requested
        self.ssl_context = None
        if not self.verify_ssl:
            self.ssl_context = ssl._create_unverified_context()

    def ocr_image(self, image: np.ndarray) -> str:
        """
        Performs OCR on an image using the default prompt.

        Args:
            image: BGR image array (OpenCV format).

        Returns:
            Extracted text content.
        """
        return self.ocr_image_with_prompt(image, self.prompt)

    def ocr_image_with_prompt(self, image: np.ndarray, prompt: str) -> str:
        """
        Performs OCR on an image using a specific prompt override.

        Encodes the image as a high-quality JPEG and sends it via base64
        to the vision-language model.

        Args:
            image: BGR image array.
            prompt: Text instructions for the OCR engine.

        Returns:
            Extracted text content.

        Raises:
            OCRConnectionError: If the server is unreachable.
            OCRError: For other processing failures.
        """
        # Encode image to JPEG with higher quality to preserve faint text
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        base64_image = base64.b64encode(buffer).decode("utf-8")

        # Prepare payload
        # Standard OpenAI vision format
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0
        }

        # Try OCR
        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=self.timeout_s, context=self.ssl_context) as response:
                result = json.loads(response.read().decode("utf-8"))

            text = result["choices"][0]["message"]["content"].strip()

            # Simple validation: warn on empty OCR result, but don't hard-fail
            if not text.strip():
                logger.warning("Received empty OCR result")

            return text

        except urllib.error.URLError as e:
            raise OCRConnectionError(f"Connection failed: {e.reason}") from e
        except Exception as e:
            if "OCRResponseError" in str(type(e)):
                raise
            raise OCRError(f"OCR processing failed: {e}") from e
