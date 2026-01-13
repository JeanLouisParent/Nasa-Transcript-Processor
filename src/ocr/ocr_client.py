"""
LM Studio OCR client (OpenAI-compatible API).

Sends enhanced page images to a local LM Studio server and returns OCR text.
"""

import base64
import json
import ssl
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
    "Do not add any conversational text or formatting outside the original content."
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
    Client for LM Studio vision-enabled OCR.
    """
    base_url: str = "http://localhost:1234"
    model: str = "qwen3-vl-4b"
    timeout_s: int = 120
    max_tokens: int = 4096
    prompt: str = PLAIN_OCR_PROMPT
    verify_ssl: bool = False

    def __post_init__(self):
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
        Send image to LM Studio for OCR processing.
        """
        # Encode image to JPEG
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        base64_image = base64.b64encode(buffer).decode("utf-8")

        # Prepare payload
        # Standard OpenAI vision format
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
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
            
            # Simple validation: ensure we got some text
            if not any(c.isalpha() for c in text):
                # Fallback: maybe retry with a different token format if needed
                # But for now, we'll just raise error if empty
                raise OCRResponseError("Received empty or non-alphabetic OCR result")
                
            return text

        except urllib.error.URLError as e:
            raise OCRConnectionError(f"Connection failed: {e.reason}")
        except Exception as e:
            if "OCRResponseError" in str(type(e)):
                raise e
            raise OCRError(f"OCR processing failed: {e}")