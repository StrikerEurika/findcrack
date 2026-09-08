from __future__ import annotations
import numpy as np
from findcrack.utils import sigmoid_np


def tta_forward_np(model, image_np: np.ndarray) -> np.ndarray:
    """
    Apply 4-way test-time augmentation using NumPy only.
    Expects image_np to be shape (C, H, W). Returns (H, W).
    """
    x = np.expand_dims(image_np, axis=0)
    
    pred_orig = sigmoid_np(model(x))
    
    x_hf = np.flip(x, axis=3)
    pred_hf = np.flip(sigmoid_np(model(x_hf)), axis=3)
    
    x_vf = np.flip(x, axis=2)
    pred_vf = np.flip(sigmoid_np(model(x_vf)), axis=2)
    
    x_r90 = np.rot90(x, k=1, axes=(2, 3))
    pred_r90 = np.rot90(sigmoid_np(model(x_r90)), k=-1, axes=(2, 3))
    
    averaged_prediction = (pred_orig + pred_hf + pred_vf + pred_r90) / 4.0

    return np.squeeze(averaged_prediction)
