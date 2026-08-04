from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional


class ImageScaler:
    """
    Optimized downscaling and upscaling utility for computer vision and deep learning inference pipelines.

    Reduces computational complexity and memory usage by processing images at a lower resolution
    and accurately upscaling predicted probability/confidence maps back to full resolution.
    """
    def __init__(
        self,
        scale_factor: Optional[float] = None,
        max_dim: Optional[int] = None,
        downscale_interpolation: int = cv2.INTER_LINEAR,
        upscale_interpolation: int = cv2.INTER_LINEAR,
    ):
        """
        Args:
            scale_factor: Ratio to scale image by (e.g., 0.5 for half resolution). Must be > 0.
            max_dim: Maximum dimension (width or height) allowed. Downscales while preserving aspect ratio. Must be > 0.
            downscale_interpolation: OpenCV interpolation method for downscaling (default: cv2.INTER_LINEAR for speed).
            upscale_interpolation: OpenCV interpolation method for upscaling maps (default: cv2.INTER_LINEAR for smooth probabilities).
        """
        if scale_factor is not None and scale_factor <= 0:
            raise ValueError("scale_factor must be positive.")
        if max_dim is not None and max_dim <= 0:
            raise ValueError("max_dim must be positive.")

        self.scale_factor = scale_factor
        self.max_dim = max_dim
        self.downscale_interpolation = downscale_interpolation
        self.upscale_interpolation = upscale_interpolation

    def calculate_scaled_size(self, original_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Calculates target (width, height) preserving aspect ratio.

        Args:
            original_shape: Image shape tuple (H, W) or (H, W, C).

        Returns:
            Tuple of (target_width, target_height).
        """
        h, w = original_shape[:2]
        target_w, target_h = w, h

        if self.scale_factor is not None:
            target_w = int(round(w * self.scale_factor))
            target_h = int(round(h * self.scale_factor))

        if self.max_dim is not None:
            max_current = max(target_h, target_w)
            if max_current > self.max_dim:
                ratio = self.max_dim / float(max_current)
                target_w = int(round(target_w * ratio))
                target_h = int(round(target_h * ratio))

        return max(1, target_w), max(1, target_h)

    def downscale(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Downscales an image if scaling parameters are defined and result in dimension changes.

        Args:
            image: Input NumPy array (H, W) or (H, W, C).

        Returns:
            Tuple of (scaled_image, is_scaled_boolean).
        """
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.calculate_scaled_size((orig_h, orig_w))

        if (target_w, target_h) == (orig_w, orig_h):
            return image, False

        scaled = cv2.resize(image, (target_w, target_h), interpolation=self.downscale_interpolation)
        return scaled, True

    def upscale_map(self, prob_map: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Upscales a continuous probability/confidence map (H, W) to target_shape (target_h, target_w).

        Args:
            prob_map: Continuous floating-point probability map.
            target_shape: Target dimensions tuple (target_h, target_w) or (target_h, target_w, C).

        Returns:
            Upscaled NumPy probability map matching target dimensions.
        """
        target_h, target_w = target_shape[:2]
        map_h, map_w = prob_map.shape[:2]

        if (map_w, map_h) == (target_w, target_h):
            return prob_map

        return cv2.resize(prob_map, (target_w, target_h), interpolation=self.upscale_interpolation)

    def upscale_mask(self, binary_mask: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Upscales a discrete binary mask (H, W) to target_shape (target_h, target_w) using nearest-neighbor.

        Args:
            binary_mask: Binary mask NumPy array (0 or 255).
            target_shape: Target dimensions tuple (target_h, target_w) or (target_h, target_w, C).

        Returns:
            Upscaled binary mask matching target dimensions.
        """
        target_h, target_w = target_shape[:2]
        mask_h, mask_w = binary_mask.shape[:2]

        if (mask_w, mask_h) == (target_w, target_h):
            return binary_mask

        return cv2.resize(binary_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
