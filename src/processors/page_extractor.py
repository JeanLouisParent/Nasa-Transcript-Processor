"""
PDF Page Extraction Module.

This module handles the extraction of individual pages from PDF documents
as high-resolution images. It is designed to process pages one at a time
without loading the entire document into memory.

For AI Agents:
    - Uses pymupdf (fitz) for PDF manipulation
    - Thread-safe for parallel processing
    - Extracts both raw PDF page and rasterized image
    - No text extraction or OCR is performed
"""

from collections.abc import Iterator
from pathlib import Path

import fitz  # pymupdf
import numpy as np
from loguru import logger

from src.config.global_config import GlobalConfig


class PageExtractor:
    """
    Extracts individual pages from PDF documents.

    This class provides methods to:
    - Get page count without loading entire document
    - Extract single pages as images (numpy arrays)
    - Extract single pages as separate PDF files
    - Iterate over pages lazily

    Attributes:
        pdf_path: Path to the source PDF file
        config: Pipeline configuration
    """

    def __init__(self, pdf_path: Path, config: GlobalConfig | None = None):
        """
        Initialize the page extractor.

        Args:
            pdf_path: Path to the PDF file to process
            config: Pipeline configuration (uses defaults if None)
        """
        self.pdf_path = Path(pdf_path)
        self.config = config or GlobalConfig()

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        # Validate PDF can be opened
        with fitz.open(self.pdf_path) as doc:
            self._page_count = len(doc)
            logger.debug(f"Opened PDF with {self._page_count} pages: {self.pdf_path}")

    @property
    def page_count(self) -> int:
        """Return the total number of pages in the PDF."""
        return self._page_count

    def extract_page_image(self, page_num: int) -> np.ndarray:
        """
        Extract a single page as a numpy image array.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            numpy.ndarray: Image in BGR format (OpenCV compatible)

        Raises:
            IndexError: If page_num is out of range
        """
        if page_num < 0 or page_num >= self._page_count:
            raise IndexError(f"Page {page_num} out of range (0-{self._page_count - 1})")

        with fitz.open(self.pdf_path) as doc:
            page = doc.load_page(page_num)

            # Create pixmap at target DPI
            # zoom factor = target_dpi / 72 (PDF default is 72 DPI)
            zoom = self.config.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            # Convert to numpy array
            # Pixmap samples are in RGB format
            img_data = np.frombuffer(pixmap.samples, dtype=np.uint8)
            img = img_data.reshape(pixmap.height, pixmap.width, 3)

            # Convert RGB to BGR for OpenCV compatibility
            img_bgr = img[:, :, ::-1].copy()

            logger.debug(
                f"Extracted page {page_num + 1}/{self._page_count} "
                f"at {self.config.dpi} DPI: {img_bgr.shape}"
            )

            return img_bgr

    def extract_page_pdf(self, page_num: int, output_path: Path) -> Path:
        """
        Extract a single page as a separate PDF file.

        Args:
            page_num: Page number (0-indexed)
            output_path: Path for the output PDF file

        Returns:
            Path to the created PDF file

        Raises:
            IndexError: If page_num is out of range
        """
        if page_num < 0 or page_num >= self._page_count:
            raise IndexError(f"Page {page_num} out of range (0-{self._page_count - 1})")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with fitz.open(self.pdf_path) as src_doc:
            # Create new document with single page
            with fitz.open() as dst_doc:
                dst_doc.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
                dst_doc.save(output_path, garbage=4, deflate=True)

        logger.debug(f"Extracted page {page_num + 1} to {output_path}")
        return output_path

    def iter_pages(
        self,
        start: int = 0,
        end: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Iterate over pages, yielding page number and image.

        This is a lazy iterator that loads one page at a time.

        Args:
            start: Starting page number (0-indexed, inclusive)
            end: Ending page number (0-indexed, exclusive). None means all pages.

        Yields:
            Tuple of (page_number, image_array)

        Example:
            >>> extractor = PageExtractor("document.pdf")
            >>> for page_num, img in extractor.iter_pages(start=0, end=10):
            ...     process(img)
        """
        if end is None:
            end = self._page_count

        start = max(0, start)
        end = min(end, self._page_count)

        for page_num in range(start, end):
            yield page_num, self.extract_page_image(page_num)

    def get_page_dimensions(self, page_num: int = 0) -> tuple[float, float]:
        """
        Get the dimensions of a page in points (1/72 inch).

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Tuple of (width, height) in points
        """
        with fitz.open(self.pdf_path) as doc:
            page = doc.load_page(page_num)
            rect = page.rect
            return rect.width, rect.height

    def get_page_info(self, page_num: int = 0) -> dict:
        """
        Get metadata about a specific page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Dictionary with page information including:
            - width_pts: Width in points
            - height_pts: Height in points
            - width_px: Width in pixels at configured DPI
            - height_px: Height in pixels at configured DPI
            - rotation: Page rotation in degrees
        """
        with fitz.open(self.pdf_path) as doc:
            page = doc.load_page(page_num)
            rect = page.rect
            zoom = self.config.dpi / 72.0

            return {
                "width_pts": rect.width,
                "height_pts": rect.height,
                "width_px": int(rect.width * zoom),
                "height_px": int(rect.height * zoom),
                "rotation": page.rotation,
            }


def get_pdf_info(pdf_path: Path) -> dict:
    """
    Extracts basic metadata and page count from a PDF document.

    Args:
        pdf_path: Path to the source PDF file.

    Returns:
        Dictionary containing metadata fields: page_count, title, author,
        creator, producer, creation_date, and modification_date.
    """
    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata or {}

        return {
            "page_count": len(doc),
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
        }
