import numpy as np
from typing import Tuple

class PatchBlender:
    """
    Reconstructs the full image by blending overlapping patches.
    Supports both standard "average" blending and "gaussian" window blending.
    """
    def __init__(self, shape: Tuple[int, int], blend_mode: str = "average"):
        if blend_mode not in ("average", "gaussian"):
            raise ValueError("blend_mode must be 'average' or 'gaussian'")
        self.prediction_map = np.zeros(shape, dtype=np.float32)
        self.count_map = np.zeros(shape, dtype=np.float32)
        self.blend_mode = blend_mode
        self._window_cache = {}

    def _get_window(self, patch_shape: Tuple[int, int]) -> np.ndarray:
        if patch_shape not in self._window_cache:
            if self.blend_mode == "average":
                window = np.ones(patch_shape, dtype=np.float32)
            else:
                # blend_mode == "gaussian"
                h, w = patch_shape
                # Create 1D Gaussian kernels
                # sigma is 0.25 (standard choice for smooth blend without too sharp dropoffs)
                h_kernel = np.arange(h) - (h - 1) / 2
                h_kernel = np.exp(-0.5 * (h_kernel / (0.25 * h)) ** 2)
                
                w_kernel = np.arange(w) - (w - 1) / 2
                w_kernel = np.exp(-0.5 * (w_kernel / (0.25 * w)) ** 2)
                
                window = np.outer(h_kernel, w_kernel).astype(np.float32)
                # Keep weights bounded slightly away from 0 to prevent numerical issues
                window = np.clip(window, 1e-4, 1.0)
                
            self._window_cache[patch_shape] = window
        return self._window_cache[patch_shape]

    def add(self, patch: np.ndarray, coordinates: Tuple[int, int]):
        """
        Adds a patch to the prediction map using the configured blend_mode.
        """
        y, x = coordinates
        height, width = patch.shape
        window = self._get_window((height, width))
        
        self.prediction_map[y:y+height, x:x+width] += patch * window
        self.count_map[y:y+height, x:x+width] += window

    def merge(self) -> np.ndarray:
        """
        Merges the prediction map with the count map to produce the final blended image.
        """
        valid = self.count_map > 1e-5
        self.prediction_map[valid] /= self.count_map[valid]
        return self.prediction_map
