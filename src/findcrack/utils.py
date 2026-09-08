import numpy as np


def sigmoid_np(val: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function for NumPy arrays.
    Clips input to prevent overflow/underflow warnings in np.exp.
    """
    return 1 / (1 + np.exp(-np.clip(val, -88.0, 88.0)))
