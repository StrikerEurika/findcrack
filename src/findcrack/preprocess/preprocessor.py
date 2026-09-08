from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional, Union
import albumentations as A
from .clahe import apply_lab_clahe


class Preprocessor:
    """
    Standard preprocessing pipeline class.
    Handles optional LAB-CLAHE contrast enhancement and Albumentations normalization.
    """
    def __init__(
        self,
        use_clahe: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        normalize: Union[bool, str] = True,
        additional_transforms: Optional[List[A.BasicTransform]] = None
    ):
        self.use_clahe = use_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.mean = mean
        self.std = std

        if isinstance(normalize, bool):
            self.normalize_mode = "standard" if normalize else "none"
        else:
            if normalize not in ("standard", "scale_only", "none"):
                raise ValueError(
                    f"normalize must be True, False, 'standard', 'scale_only', or 'none', got: {normalize}"
                )
            self.normalize_mode = normalize

        transforms = []
        if additional_transforms:
            transforms.extend(additional_transforms)

        if self.normalize_mode == "standard":
            transforms.append(A.Normalize(mean=self.mean, std=self.std))
        elif self.normalize_mode == "scale_only":
            transforms.append(A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)))

        self.transform = A.Compose(transforms)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Applies LAB-CLAHE contrast enhancement if enabled.
        """

        if self.use_clahe:
            return apply_lab_clahe(image, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        return image

    def transform_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Applies Albumentations normalization to a patch or image.
        """
        transformed = self.transform(image=patch)
        img = transformed["image"]

        if self.normalize_mode == "none":
            if isinstance(img, np.ndarray):
                img = img.astype(np.float32)
                if img.ndim == 3:
                    img = np.transpose(img, (2, 0, 1))
            return img

        if isinstance(img, np.ndarray):
            img = np.transpose(img, (2, 0, 1))
        return img

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs full preprocessing: contrast enhancement followed by transform.
        Returns:
            - enhanced_image: Contrast-enhanced RGB image (numpy array).
            - array: The normalized NumPy array (CHW format).
        """
        enhanced = self.enhance_contrast(image)
        array = self.transform_patch(enhanced)
        return enhanced, array
