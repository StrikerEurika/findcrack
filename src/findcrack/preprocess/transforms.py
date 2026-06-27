import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

def get_inference_transform(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    """
    standard ImageNet normalization required by most pretrained models.
    """
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
