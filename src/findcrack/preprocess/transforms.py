import albumentations as A
from typing import Tuple

try:
    from albumentations.pytorch import ToTensorV2
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_inference_transform(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    """
    standard ImageNet normalization required by most pretrained models.
    """
    if not HAS_TORCH:
        raise ImportError(
            "get_inference_transform requires PyTorch. Please install PyTorch or "
            "install findcrack with standard extras: pip install findcrack[standard]"
        )
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

