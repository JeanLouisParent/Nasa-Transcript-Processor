"""
Image Processing Module.

This module handles all image preprocessing and enhancement operations:
- Deskew (rotation correction)
- Size normalization
- Light contrast enhancement
- Gentle noise reduction

For AI Agents:
    - All operations are purely geometric/visual (no OCR)
    - Uses OpenCV for image manipulation
    - Processing is CONSERVATIVE - preserves original quality
    - Enhancement is subtle to maintain readability
"""

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from src.config.global_config import GlobalConfig


@dataclass
class ProcessingResult:
    """
    Result of image processing operations.

    Attributes:
        image: Processed image (grayscale or BGR)
        deskew_angle: Detected and corrected skew angle in degrees
        original_size: Original image dimensions (height, width)
        final_size: Final image dimensions after normalization
        processing_steps: List of applied processing steps
    """
    image: np.ndarray
    deskew_angle: float = 0.0
    original_size: tuple[int, int] = (0, 0)
    final_size: tuple[int, int] = (0, 0)
    processing_steps: list[str] | None = None

    def __post_init__(self):
        if self.processing_steps is None:
            self.processing_steps = []


class ImageProcessor:
    """
    Processes scanned document images.

    This class applies a series of CONSERVATIVE image processing operations
    to improve quality while preserving readability.

    The philosophy is: less is more. We want to:
    - Fix geometric issues (deskew, normalize size)
    - Gently improve contrast
    - NOT destroy text quality with aggressive filtering

    Attributes:
        config: Pipeline configuration
    """

    def __init__(self, config: GlobalConfig | None = None):
        """
        Initialize the image processor.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or GlobalConfig()

    def process(self, image: np.ndarray) -> ProcessingResult:
        """
        Apply processing pipeline to an image.

        Pipeline steps:
        1. Convert to grayscale
        2. Deskew (rotation correction)
        3. Normalize size and margins
        4. Light enhancement (optional, controlled by config)

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            ProcessingResult with processed image and metadata
        """
        result = ProcessingResult(
            image=image.copy(),
            original_size=(image.shape[0], image.shape[1]),
        )
        if result.processing_steps is None:
            result.processing_steps = []

        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result.processing_steps.append("grayscale_conversion")
        else:
            gray = image.copy()

        result.image = gray

        # Step 2: Deskew
        gray, angle = self._deskew(gray)
        result.deskew_angle = angle
        result.image = gray
        if abs(angle) > 0.01:
            result.processing_steps.append(f"deskew_{angle:.2f}deg")

        # Step 3: Normalize size and margins
        gray = self._normalize_size(gray)
        result.image = gray
        result.final_size = (gray.shape[0], gray.shape[1])
        result.processing_steps.append("size_normalization")

        # Step 4: Light enhancement (conservative)
        gray = self._enhance_light(gray)
        result.image = gray
        result.processing_steps.append("light_enhancement")

        logger.debug(
            f"Processed image: {result.original_size} -> {result.final_size}, "
            f"deskew={result.deskew_angle:.2f}deg, steps={len(result.processing_steps)}"
        )

        return result

    def _deskew(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detects and corrects rotation skew in the document image.

        Analyzes text lines to calculate the median skew angle. Supports
        high-quality rotation with white padding.

        Args:
            image: Grayscale source image.

        Returns:
            Tuple of (deskewed_image, angle_in_degrees).
        """
        # Threshold to create binary image
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Method 1: Analyze text lines using horizontal dilation
        # This connects characters into lines for better angle detection
        k_w = self.config.line_kernel_width
        k_h = self.config.line_kernel_height
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_angles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filter small contours
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1]
                if w > 0 and h > 0:
                    aspect = max(w, h) / min(w, h)
                    # Only consider elongated shapes (text lines)
                    if aspect > 5:
                        angle = rect[-1]
                        # Normalize angle to [-45, 45] range
                        if angle < -45:
                            angle = 90 + angle
                        elif angle > 45:
                            angle = angle - 90
                        if abs(angle) < self.config.deskew_max_angle:
                            line_angles.append(angle)

        # Method 2: Fallback to Hough lines if not enough text lines found
        if len(line_angles) < 5:
            edges = cv2.Canny(binary, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180,
                threshold=100, minLineLength=100, maxLineGap=10
            )
            if lines is not None:
                for x1, y1, x2, y2 in lines.reshape(-1, 4):
                    if x2 - x1 != 0:
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        if abs(angle) < self.config.deskew_max_angle:
                            line_angles.append(angle)

        if not line_angles:
            logger.debug("No lines detected for deskew")
            return image, 0.0

        # Use median for robustness
        median_angle = float(np.median(line_angles))

        # Use configured threshold
        if abs(median_angle) < self.config.deskew_angle_threshold:
            logger.debug(f"Skew angle {median_angle:.3f}° below threshold, skipping")
            return image, median_angle

        # Rotate to correct skew
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Calculate new bounding box size
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        # Apply rotation with white background and high-quality interpolation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        logger.debug(f"Corrected skew of {median_angle:.3f}° ({len(line_angles)} lines analyzed)")
        return rotated, median_angle

    def _normalize_size(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes image dimensions and centers content with consistent margins.

        Detects the content bounding box, crops, scales to fit target height/width
        while maintaining aspect ratio, and places on a white canvas.

        Args:
            image: Grayscale image.

        Returns:
            Resized and centered image.
        """
        target_h = self.config.target_height
        target_w = self.config.target_width
        margin = self.config.margin_px

        # Find content bounding box
        # Threshold to find non-white pixels (more lenient threshold)
        _, binary = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours of content
        coords = cv2.findNonZero(binary)

        if coords is None:
            # Empty page - just resize
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Get bounding rectangle of all content
        x, y, w, h = cv2.boundingRect(coords)

        # Add small padding around content
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2 * pad)
        h = min(image.shape[0] - y, h + 2 * pad)

        # Crop to content
        content = image[y:y+h, x:x+w]

        # Calculate available space for content (target minus margins)
        content_area_w = target_w - 2 * margin
        content_area_h = target_h - 2 * margin

        # Calculate scale to fit content in available area
        scale_w = content_area_w / w
        scale_h = content_area_h / h
        scale = min(scale_w, scale_h)  # Maintain aspect ratio

        # Resize content using high-quality interpolation
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Use INTER_LANCZOS4 for high-quality downscaling
        if scale < 1:
            resized_content = cv2.resize(content, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_content = cv2.resize(content, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Create target canvas (white)
        normalized = np.full((target_h, target_w), 255, dtype=np.uint8)

        # Calculate position to center content
        start_x = margin + (content_area_w - new_w) // 2
        start_y = margin + (content_area_h - new_h) // 2

        # Place content on canvas
        normalized[start_y:start_y+new_h, start_x:start_x+new_w] = resized_content

        return normalized

    def _enhance_light(self, image: np.ndarray) -> np.ndarray:
        """
        Applies subtle, conservative enhancement to improve readability.

        Removes small specks, whitens the background, and adjusts contrast
        without compromising character integrity.

        Args:
            image: Grayscale image.

        Returns:
            Lightly enhanced image.
        """
        # Step 1: Light median filter to remove salt-and-pepper noise
        denoised = cv2.medianBlur(image, 3)

        # Step 2: Normalize brightness with better white/black points
        white_point = np.percentile(denoised.astype(float), 98)
        black_point = np.percentile(denoised.astype(float), 2)

        if white_point > black_point:
            scale = 255.0 / (white_point - black_point)
            enhanced = np.clip((denoised.astype(float) - black_point) * scale, 0, 255)
            enhanced = enhanced.astype(np.uint8)
        else:
            enhanced = denoised.copy()

        # Step 3: Clean up near-white pixels (make background uniformly white)
        enhanced = np.where(enhanced > 240, 255, enhanced).astype(np.uint8)

        # Step 4: Remove small isolated dark spots using connected components
        enhanced = self._remove_small_spots(enhanced, max_spot_size=15)

        # Step 5: Slight gamma correction to improve text contrast
        gamma = 0.92
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)

        return enhanced

    def _remove_small_spots(self, image: np.ndarray, max_spot_size: int = 15) -> np.ndarray:
        """
        Removes isolated dark pixels (noise) while protecting text lines.

        Uses connected component analysis to filter out regions that are
        too small or too square to be part of valid characters.

        Args:
            image: Grayscale image.
            max_spot_size: Maximum pixel area for a spot to be considered noise.

        Returns:
            Cleaned image.
        """
        # Create binary image (dark pixels = foreground)
        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Create mask of spots to remove
        spots_mask = np.zeros(image.shape, dtype=np.uint8)

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Remove if:
            # - Very small area (isolated dots)
            # - Small and roughly square (not part of text line)
            is_tiny = area <= max_spot_size
            is_small_square = area <= 50 and max(width, height) / max(1, min(width, height)) < 2

            if is_tiny or is_small_square:
                spots_mask[labels == i] = 255

        # Replace spots with white
        result = image.copy()
        result[spots_mask == 255] = 255

        return result

    def enhance_contrast_heavy(self, image: np.ndarray) -> np.ndarray:
        """
        Boost faint text visibility with strong local contrast and sharpening.
        Useful for OCR backup passes on faint/illegible pages.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        p_low, p_high = np.percentile(gray, (2, 98))
        if p_high > p_low:
            stretched = np.clip((gray - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        else:
            stretched = gray

        grid_size = (self.config.clahe_grid_size, self.config.clahe_grid_size)
        # Use a slightly higher clip limit for 'heavy' enhancement than the default config
        clip_limit = self.config.clahe_clip_limit * 2.0

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        boosted = clahe.apply(stretched)
        blurred = cv2.GaussianBlur(boosted, (0, 0), 1.2)
        sharpened = cv2.addWeighted(boosted, 1.7, blurred, -0.7, 0)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
