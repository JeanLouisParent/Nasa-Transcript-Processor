"""
Main Pipeline Orchestrator.

This module coordinates all processing steps for NASA transcript documents.
It supports both sequential and parallel processing modes.

For AI Agents:
    - Entry point for processing PDF documents
    - Supports page range selection
    - Parallel processing via ThreadPoolExecutor
    - Each page is processed independently (no inter-page dependencies)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Callable
import threading

from loguru import logger
from tqdm import tqdm

from .config import PipelineConfig
from .page_extractor import PageExtractor
from .image_processor import ImageProcessor, ProcessingResult
from .layout_detector import LayoutDetector, LayoutResult
from .output_generator import OutputGenerator, PageOutput


@dataclass
class PageResult:
    """
    Complete processing result for a single page.

    Attributes:
        page_num: Page number (0-indexed)
        processing: Image processing result
        layout: Layout detection result
        output: Output file paths
        success: True if processing completed without errors
        error: Error message if processing failed
    """
    page_num: int
    processing: Optional[ProcessingResult] = None
    layout: Optional[LayoutResult] = None
    output: Optional[PageOutput] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """
    Result of processing an entire document.

    Attributes:
        total_pages: Total number of pages in document
        processed_pages: Number of pages processed
        successful_pages: Number of pages processed successfully
        failed_pages: Number of pages that failed
        page_results: List of individual page results
        output_dir: Directory containing outputs
    """
    total_pages: int
    processed_pages: int
    successful_pages: int
    failed_pages: int
    page_results: list[PageResult]
    output_dir: Path


class TranscriptPipeline:
    """
    Main pipeline for processing NASA transcript PDFs.

    This class orchestrates the complete processing workflow:
    1. Extract pages from PDF
    2. Process images (deskew, enhance)
    3. Detect layout blocks
    4. Classify blocks
    5. Generate outputs

    Supports parallel processing for improved performance on
    multi-core systems.

    Attributes:
        pdf_path: Path to source PDF
        output_dir: Directory for outputs
        config: Pipeline configuration
    """

    def __init__(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize the pipeline.

        Args:
            pdf_path: Path to the PDF file to process
            output_dir: Directory for output files
            config: Pipeline configuration (uses defaults if None)
        """
        self.pdf_path = Path(pdf_path)
        base_output_dir = Path(output_dir)
        pdf_stem = self.pdf_path.stem
        if base_output_dir.name == pdf_stem:
            self.output_dir = base_output_dir
        else:
            self.output_dir = base_output_dir / pdf_stem
        self.config = config or PipelineConfig()

        # Initialize components
        self.extractor = PageExtractor(self.pdf_path, self.config)
        self.processor = ImageProcessor(self.config)
        self.layout_detector = LayoutDetector(self.config)
        self.output_generator = OutputGenerator(self.output_dir, pdf_stem, self.config)

        # Thread lock for thread-safe PDF extraction
        self._extract_lock = threading.Lock()

        logger.info(
            f"Initialized pipeline for {self.pdf_path.name} "
            f"({self.extractor.page_count} pages)"
        )

    @property
    def page_count(self) -> int:
        """Total number of pages in the document."""
        return self.extractor.page_count

    def process_page(self, page_num: int) -> PageResult:
        """
        Process a single page through the complete pipeline.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            PageResult with processing results and outputs
        """
        result = PageResult(page_num=page_num)

        try:
            # Step 1: Extract page image and PDF
            # Thread-safe extraction
            with self._extract_lock:
                image = self.extractor.extract_page_image(page_num)
                raw_pdf_path = self.output_generator.get_raw_pdf_path(page_num)
                self.extractor.extract_page_pdf(page_num, raw_pdf_path)

            # Step 2: Process image
            processing_result = self.processor.process(image)
            result.processing = processing_result

            # Step 3: Detect layout (includes block classification)
            layout_result = self.layout_detector.detect(processing_result.image)
            result.layout = layout_result

            # Step 4: Generate outputs
            output = self.output_generator.generate(
                page_num=page_num,
                enhanced_image=processing_result.image,
                layout=layout_result
            )
            result.output = output

            result.success = True
            logger.debug(f"Successfully processed page {page_num + 1}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Failed to process page {page_num + 1}: {e}")

        return result

    def process_range(
        self,
        start: int = 0,
        end: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> PipelineResult:
        """
        Process a range of pages.

        Args:
            start: Starting page number (0-indexed, inclusive)
            end: Ending page number (0-indexed, exclusive). None means all pages.
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            PipelineResult with all page results
        """
        if end is None:
            end = self.page_count

        start = max(0, start)
        end = min(end, self.page_count)
        total = end - start

        page_results = []
        successful = 0
        failed = 0

        if self.config.parallel and self.config.max_workers > 1:
            # Parallel processing
            page_results = self._process_parallel(start, end, progress_callback)
        else:
            # Sequential processing
            page_results = self._process_sequential(start, end, progress_callback)

        # Count successes and failures
        for result in page_results:
            if result.success:
                successful += 1
            else:
                failed += 1

        logger.info(
            f"Completed processing: {successful}/{total} pages successful, "
            f"{failed} failed"
        )

        return PipelineResult(
            total_pages=self.page_count,
            processed_pages=total,
            successful_pages=successful,
            failed_pages=failed,
            page_results=page_results,
            output_dir=self.output_dir
        )

    def process_pages(
        self,
        page_numbers: list[int],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> PipelineResult:
        """
        Process a list of specific page numbers.

        Args:
            page_numbers: List of page numbers (0-indexed)
            progress_callback: Optional progress callback

        Returns:
            PipelineResult with page results
        """
        unique_pages = sorted({p for p in page_numbers if 0 <= p < self.page_count})
        total = len(unique_pages)
        if total == 0:
            return PipelineResult(
                total_pages=self.page_count,
                processed_pages=0,
                successful_pages=0,
                failed_pages=0,
                page_results=[],
                output_dir=self.output_dir
            )

        if self.config.parallel and self.config.max_workers > 1:
            page_results = self._process_pages_parallel(unique_pages, progress_callback)
        else:
            page_results = self._process_pages_sequential(unique_pages, progress_callback)

        successful = sum(1 for r in page_results if r.success)
        failed = total - successful

        logger.info(
            f"Completed processing: {successful}/{total} pages successful, "
            f"{failed} failed"
        )

        return PipelineResult(
            total_pages=self.page_count,
            processed_pages=total,
            successful_pages=successful,
            failed_pages=failed,
            page_results=page_results,
            output_dir=self.output_dir
        )

    def _process_sequential(
        self,
        start: int,
        end: int,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[PageResult]:
        """
        Process pages sequentially.

        Args:
            start: Start page (inclusive)
            end: End page (exclusive)
            progress_callback: Progress callback

        Returns:
            List of page results
        """
        results = []
        total = end - start

        with tqdm(total=total, desc="Processing pages", unit="page") as pbar:
            for page_num in range(start, end):
                result = self.process_page(page_num)
                results.append(result)
                pbar.update(1)

                if progress_callback:
                    progress_callback(len(results), total)

        return results

    def _process_parallel(
        self,
        start: int,
        end: int,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[PageResult]:
        """
        Process pages in parallel using ThreadPoolExecutor.

        Args:
            start: Start page (inclusive)
            end: End page (exclusive)
            progress_callback: Progress callback

        Returns:
            List of page results (sorted by page number)
        """
        results = []
        total = end - start
        completed = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(self.process_page, page_num): page_num
                for page_num in range(start, end)
            }

            # Collect results with progress bar
            with tqdm(total=total, desc="Processing pages", unit="page") as pbar:
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # Handle unexpected errors
                        results.append(PageResult(
                            page_num=page_num,
                            success=False,
                            error=str(e)
                        ))
                        logger.error(f"Unexpected error on page {page_num + 1}: {e}")

                    completed += 1
                    pbar.update(1)

                    if progress_callback:
                        progress_callback(completed, total)

        # Sort results by page number
        results.sort(key=lambda r: r.page_num)
        return results

    def _process_pages_sequential(
        self,
        page_numbers: list[int],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[PageResult]:
        """Process specific pages sequentially."""
        results = []
        total = len(page_numbers)

        with tqdm(total=total, desc="Processing pages", unit="page") as pbar:
            for page_num in page_numbers:
                result = self.process_page(page_num)
                results.append(result)
                pbar.update(1)

                if progress_callback:
                    progress_callback(len(results), total)

        return results

    def _process_pages_parallel(
        self,
        page_numbers: list[int],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[PageResult]:
        """Process specific pages in parallel."""
        results = []
        total = len(page_numbers)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_page = {
                executor.submit(self.process_page, page_num): page_num
                for page_num in page_numbers
            }

            with tqdm(total=total, desc="Processing pages", unit="page") as pbar:
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append(PageResult(
                            page_num=page_num,
                            success=False,
                            error=str(e)
                        ))
                        logger.error(f"Unexpected error on page {page_num + 1}: {e}")

                    completed += 1
                    pbar.update(1)

                    if progress_callback:
                        progress_callback(completed, total)

        results.sort(key=lambda r: r.page_num)
        return results

    def process_all(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> PipelineResult:
        """
        Process all pages in the document.

        Args:
            progress_callback: Optional progress callback

        Returns:
            PipelineResult with all page results
        """
        return self.process_range(
            start=0,
            end=self.page_count,
            progress_callback=progress_callback
        )


def run_pipeline(
    pdf_path: Path,
    output_dir: Path,
    start: int = 0,
    end: Optional[int] = None,
    config: Optional[PipelineConfig] = None
) -> PipelineResult:
    """
    Convenience function to run the complete pipeline.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        start: Start page (0-indexed)
        end: End page (exclusive)
        config: Pipeline configuration

    Returns:
        PipelineResult with processing results
    """
    pipeline = TranscriptPipeline(pdf_path, output_dir, config)
    return pipeline.process_range(start, end)
