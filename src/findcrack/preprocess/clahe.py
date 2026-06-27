import cv2
import numpy as np
from typing import Tuple

def apply_lab_clahe(
    image: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply clahe to the L-channel of the LAB color space to enhance
    local contrast without altering color balance.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
