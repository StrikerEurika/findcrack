import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional
from .clahe import apply_lab_clahe

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
        transforms.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])
        self.transform = A.Compose(transforms)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Applies LAB-CLAHE contrast enhancement if enabled.
        """
        if self.use_clahe:
            return apply_lab_clahe(image, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        return image

    def transform_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Applies Albumentations normalization and tensorization to a patch or image.
        """
        transformed = self.transform(image=patch)
        return transformed["image"]

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Performs full preprocessing: contrast enhancement followed by transform.
        Returns:
            - enhanced_image: Contrast-enhanced RGB image (numpy array).
            - tensor: The normalized PyTorch tensor.
        """
        enhanced = self.enhance_contrast(image)
        tensor = self.transform_patch(enhanced)
        return enhanced, tensor
