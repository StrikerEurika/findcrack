import cv2
import numpy as np
from typing import Tuple

def apply_lab_clahe(
    image: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE to the L-channel of the LAB color space (for RGB/RGBA)
    or directly (for grayscale) to enhance local contrast.
    """
    ndim = image.ndim
    if ndim == 2:
        # Grayscale 2D image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
        
    elif ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            # Grayscale 3D image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(image[:, :, 0])
            return np.expand_dims(cl, axis=-1)
        elif channels == 3:
            # Standard RGB
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_image)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a_channel, b_channel))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        elif channels == 4:
            # RGBA
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            alpha = image[:, :, 3]
            enhanced_rgb = apply_lab_clahe(rgb_image, clip_limit, tile_grid_size)
            return cv2.merge((*cv2.split(enhanced_rgb), alpha))
            
    # Fallback / unsupported format
    return image

