import numpy as np

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    smp = None

def sigmoid_np(val: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function for NumPy arrays.
    Clips input to prevent overflow/underflow warnings in np.exp.
    """
    # For float32, np.exp(88.7) is the overflow limit (~3.4e38).
    # Clipping to [-88.0, 88.0] is safe for both float32 and float64,
    # and keeps the output mathematically correct without warnings.
    return 1 / (1 + np.exp(-np.clip(val, -88.0, 88.0)))


def load_smp_model(model_path_or_hf_id: str, device: str = "cuda"):
    """Loads an SMP model from a local path or Hugging Face Hub ID."""
    if not HAS_SMP:
        raise ImportError(
            "segmentation-models-pytorch is required to load SMP models. "
            "Please install PyTorch & SMP with standard extra: 'pip install findcrack[standard]' or 'uv add findcrack --extra standard'."
        )
    model = smp.from_pretrained(model_path_or_hf_id)
    model = model.to(device).eval()
    return model