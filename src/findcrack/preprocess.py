import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def apply_lab_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply clahe to the L-channel of the LAB color space to enhance
    local contrast without altering color balance.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def get_inference_transform():
    """
    standard ImageNet normalization required by most pretrained models.
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])