from __future__ import annotations
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def tta_forward(model, image_tensor) -> torch.Tensor:
    """
    apply 4-way test-time augmentation and averages the sigmoid predictions.
    expects image tensor to be shape (C, H, W). returns (H, W)
    """
    if not HAS_TORCH:
        raise ImportError(
            "tta_forward requires PyTorch. Please install PyTorch or "
            "install findcrack with standard extras: pip install findcrack[standard]"
        )
    
    with torch.no_grad():
        x = image_tensor.unsqueeze(0)

        # original
        original_prediction = torch.sigmoid(model(x))
        
        # horizontal flip
        horizontal_flip_prediction = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))), dims=[3])

        # vertical flip
        vertical_flip_prediction = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2]))), dims=[2])

        # rotate 90 degrees clockwise
        rotated_90_prediction = torch.rot90(
            torch.sigmoid(model(torch.rot90(x, k=1, dims=[2, 3]))),
            k=-1, dims=[2,3]
        )
        
        # average predictions
        averaged_prediction = (
            original_prediction + 
            horizontal_flip_prediction + 
            vertical_flip_prediction + 
            rotated_90_prediction) / 4.0

    return averaged_prediction.squeeze()


def tta_forward_np(model, image_np: np.ndarray) -> np.ndarray:
    """
    Apply 4-way test-time augmentation using NumPy only.
    Expects image_np to be shape (C, H, W). Returns (H, W).
    """
    def sigmoid_np(x):
        return 1 / (1 + np.exp(-x))

    # Add batch dimension to simulate (1, C, H, W)
    x = np.expand_dims(image_np, axis=0)
    
    # original
    pred_orig = sigmoid_np(model(x))
    
    # horizontal flip (flip axis 3)
    x_hf = np.flip(x, axis=3)
    pred_hf = np.flip(sigmoid_np(model(x_hf)), axis=3)
    
    # vertical flip (flip axis 2)
    x_vf = np.flip(x, axis=2)
    pred_vf = np.flip(sigmoid_np(model(x_vf)), axis=2)
    
    # rotate 90 degrees clockwise (axes 2 and 3)
    x_r90 = np.rot90(x, k=1, axes=(2, 3))
    pred_r90 = np.rot90(sigmoid_np(model(x_r90)), k=-1, axes=(2, 3))
    
    # average the predictions
    averaged_prediction = (pred_orig + pred_hf + pred_vf + pred_r90) / 4.0

    return np.squeeze(averaged_prediction)