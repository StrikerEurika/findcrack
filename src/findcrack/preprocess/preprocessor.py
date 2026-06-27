from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional
import albumentations as A
from .clahe import apply_lab_clahe

try:
    import torch
    from albumentations.pytorch import ToTensorV2
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    ToTensorV2 = None

class Preprocessor:
    """
    Standard preprocessing pipeline class.
    Handles optional LAB-CLAHE contrast enhancement and Albumentations normalization/tensorization.
    """
    def __init__(
        self,
        use_clahe: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        additional_transforms: Optional[List[A.BasicTransform]] = None
    ):
        self.use_clahe = use_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.mean = mean
        self.std = std
        
        transforms = []
        if additional_transforms:
            transforms.extend(additional_transforms)
        
        transforms.append(A.Normalize(mean=self.mean, std=self.std))
        if HAS_TORCH:
            transforms.append(ToTensorV2())
            
        self.transform = A.Compose(transforms)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Applies LAB-CLAHE contrast enhancement if enabled.
        """
        if self.use_clahe:
            return apply_lab_clahe(image, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        return image

    def transform_patch(self, patch: np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Applies Albumentations normalization and tensorization to a patch or image.
        """
        transformed = self.transform(image=patch)
        img = transformed["image"]
        if not HAS_TORCH:
            # Transpose HWC to CHW to match torch format
            img = np.transpose(img, (2, 0, 1))
        return img

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, torch.Tensor | np.ndarray]:
        """
        Performs full preprocessing: contrast enhancement followed by transform.
        Returns:
            - enhanced_image: Contrast-enhanced RGB image (numpy array).
            - tensor/array: The normalized PyTorch tensor or NumPy array (CHW format).
        """
        enhanced = self.enhance_contrast(image)
        tensor = self.transform_patch(enhanced)
        return enhanced, tensor

