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

STRUCTURED_OCR_PROMPT = (
    "You are a precise OCR engine for NASA mission transcripts. "
    "Extract all visible text from the page image and preserve reading order "
    "top-to-bottom, left-to-right, keeping original line breaks. "
    "Prefix EACH output line with exactly one tag from this set: "
    "[HEADER], [COMM], [ANNOTATION], [FOOTER], [META]. "
    "Use [HEADER] for page header lines (mission name, tape/page, network info). "
    "Use [COMM] for communication/body lines, including continuation lines without timestamps. "
    "Use [ANNOTATION] for marginal notes or REV/RFV markers. "
    "Use [FOOTER] for footer/bottom markers or asterisk blocks. "
    "Use [META] for END OF TAPE or similar meta lines. "
    "Do not add any extra commentary or formatting beyond the tag prefix. "
    "If uncertain, choose [COMM]."
)

CLASSIFY_OCR_PROMPT = (
    "You are a strict text classifier and OCR corrector for NASA transcripts. "
    "You will receive OCR text with line breaks and the original page image. "
    "Each OCR line is prefixed with a line number like \"12|\". "
    "Return the SAME number of lines in the SAME order. "
    "Each output line MUST keep the same line number prefix. "
    "Format must be: [TAG] N|original line text. "
    "Prefix EACH line with exactly one tag from: [HEADER], [COMM], [ANNOTATION], [FOOTER], [META]. "
    "Do NOT add or remove lines. Do NOT merge or split lines. "
    "Do NOT paraphrase or rewrite content. "
    "You MAY correct obvious OCR errors (e.g., common misread letters) "
    "but only when highly confident; otherwise keep the line unchanged. "
    "Do NOT invent content or guess missing words. "
    "If uncertain about the tag, use [COMM]. "
    "Each output line MUST include the original line text after the tag. "
    "Every line must start with a tag, even if it is a continuation line. "
    "Use [COMM] for all communication lines, including wrapped lines. "
    "Use [ANNOTATION] only for marginal notes (e.g., REV/RFV) or standalone non-comm notes. "
    "Lines like \"VANGUARD (REV 1)\" or \"CANARY (REV 1)\" must be [ANNOTATION], not [HEADER]. "
    "Any line starting with \"***\" (e.g., \"*** Three asterisks denote clipping of word and phrases.\") must be [FOOTER]. "
    "Even indented/wrapped lines MUST be tagged. "
    "Example: \"[HEADER] APOLLO 11 AIR-TO-GROUND VOICE TRANSCRIPTION\". "
    "Here is the OCR text:\n"
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
    prompt: str = STRUCTURED_OCR_PROMPT
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

    def classify_text(self, text: str) -> str:
        """
        Classify and lightly correct OCR text using a text-only prompt.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{CLASSIFY_OCR_PROMPT}{text}"}
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s, context=self.ssl_context) as response:
                result = json.loads(response.read().decode("utf-8"))
            classified = result["choices"][0]["message"]["content"].strip()
            if not classified:
                raise OCRResponseError("Received empty classification result")
            return classified
        except urllib.error.URLError as e:
            raise OCRConnectionError(f"Connection failed: {e.reason}")
        except Exception as e:
            if "OCRResponseError" in str(type(e)):
                raise e
            raise OCRError(f"OCR classification failed: {e}")

    def classify_image_text(self, image: np.ndarray, text: str, extra_instruction: str | None = None) -> str:
        """
        Classify and lightly correct OCR text using the page image for context.
        """
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        base64_image = base64.b64encode(buffer).decode("utf-8")
        instruction = CLASSIFY_OCR_PROMPT
        if extra_instruction:
            instruction = f"{CLASSIFY_OCR_PROMPT}{extra_instruction}\n"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instruction}{text}"},
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

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s, context=self.ssl_context) as response:
                result = json.loads(response.read().decode("utf-8"))
            classified = result["choices"][0]["message"]["content"].strip()
            if not classified:
                raise OCRResponseError("Received empty classification result")
            return classified
        except urllib.error.URLError as e:
            raise OCRConnectionError(f"Connection failed: {e.reason}")
        except Exception as e:
            if "OCRResponseError" in str(type(e)):
                raise e
            raise OCRError(f"OCR classification failed: {e}")
