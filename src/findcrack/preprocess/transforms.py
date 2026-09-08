import numpy as np
import albumentations as A
from typing import Tuple


def get_inference_transform(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    """
    standard ImageNet normalization required by most pretrained models.
    Returns a callable that applies normalization and converts to CHW numpy array.
    """
    transform = A.Compose([
        A.Normalize(mean=mean, std=std),
    ])
    def _apply(image):
        out = transform(image=image)["image"]
        return np.transpose(out, (2, 0, 1))
    return _apply
