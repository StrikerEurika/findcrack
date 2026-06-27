import numpy as np
from typing import Tuple

class PatchBlender:
    """
    Reconstructs the full image by averaging overlapping patches.
    """
    def __init__(self, shape: Tuple[int, int]):
        self.prediction_map = np.zeros(shape, dtype=np.float32)
        self.count_map = np.zeros(shape, dtype=np.int32)
        
    def add(self, patch: np.ndarray, coordinates: Tuple[int, int]):
        """
        Adds a patch to the prediction map and updates the count map.
        """
        y, x = coordinates
        height, width = patch.shape
        
        self.prediction_map[y:y+height, x:x+width] += patch
        self.count_map[y:y+height, x:x+width] += 1
        
    def merge(self) -> np.ndarray:
        """
        Merges the prediction map with the count map to produce the final blended image.
        """
        valid = self.count_map > 0
        self.prediction_map[valid] /= self.count_map[valid]
        return self.prediction_map
