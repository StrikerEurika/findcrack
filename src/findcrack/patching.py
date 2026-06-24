import numpy as np
from typing import Tuple, Generator

class PatchExtractor:
    """
    Extracts the overlapping patches from a large image.
    """
    def __init__(self, patch_size: Tuple[int, int], overlap_ratio: float = 0.5):
        """
        Args:
            patch_size (height, width): the size of the patch to be extracted.
            overlap_ratio: float number between o.0 and 0.99. the default value is 0.5,
            meaning that the patches will overlap by 50% in both dimensions. 
        """
        
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be between 0.0 and 1.0")
        
        self.patch_height, self.patch_width = patch_size
        self.stride_height = int(self.patch_height * (1 - overlap_ratio))
        self.stride_width = int(self.patch_width * (1 - overlap_ratio))
        
        # ensure that stride is at least 1 to avoid infinite loops
        self.stride_height = max(1, self.stride_height)
        self.stride_width = max(1, self.stride_width)

    def extract(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[int, int]], None, None]:
        """
        Yields patches and their top-left (y, x) coordinates.
        Handles edges by shifting the last patch to align with the image border.
        """
        
        image_height, image_width = image.shape[:2]
        seen_coordinates = set()
        
        for y in range(0, image_height, self.stride_height):
            for x in range(0, image_width, self.stride_width):
                
                # shift the patch if it goes out of bounds
                if y + self.patch_height > image_height:
                    y = image_height - self.patch_height
                
                if x + self.patch_width > image_width:
                    x = image_width - self.patch_width
                    
                # ensure we don't yield the same patch multiple times
                if (y, x) in seen_coordinates:
                    continue
                seen_coordinates.add((y, x))

                # yield the patch and its coordinates
                yield image[y:y+self.patch_height, x:x+self.patch_width], (y, x)
                
# Patch Reconstruction
class PatchBlender:
    """
    Reconstructs the full image from the overlapping patches using Gaussian Blending
    to eliminate seam artifacts
    """
    
    def __init__(self, output_shape: Tuple[int, int], patch_size: Tuple[int, int], num_classes: int = 1):
        self.output_height, self.output_width = output_shape
        self.patch_height, self.patch_width = patch_size
        self.num_classes = num_classes
        
        # accumulater for value and weights
        # using float32 to avoid overflow during accumulation
        self.accumulated_values = np.zeros((num_classes, self.output_height, self.output_width), dtype=np.float32)
        self.accumilated_weights = np.zeros((num_classes, self.output_height, self.output_width), dtype=np.float32)

        # pre-compute the 2d gaussian wieghts for a single patch
        self.weight_map = self._create_gaussian_weight(self.patch_height, self.patch_width)


    def _create_gaussian_weight(self, height: int, width: int, sigma_factor: float = 0.25) -> np.ndarray:
        """
        Create a 2d Gaussian mask. Center center is 1.0, edge fade to 0.0
        """
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)

        X, Y = np.meshgrid(x, y)

        # sigma_factor controls the drop-off. o.25 makes the edges fade to 0.0
        weights = np.exp(-(X**2 + Y**2) / (2 * sigma_factor**2))
        return weights.astype(np.float32)


    def add_patch(self, patch_prediction: np.ndarray, coordinates: Tuple[int, int]):
        """
        Adds a predicte patch to the accomultor.
        """
        y, x = coordinates
        c = patch_prediction.shape[0]
        
        # expand the weight map to march number of classes if necessary
        weights = np.expand_dims(self.weight_map, 0) if c == 1 else np.stack([self.weight_map] * c, axis=0)
        
        self.accumulated_values[:, y:y+self.patch_height, x:x+self.patch_width] += patch_prediction * weights
        self.accumilated_weights[:, y:y+self.patch_height, x:x+self.patch_width] += weights
        
    
    def merge(self) -> np.ndarray:
        """
        Return the final blended image of the shape (C, H, W)
        """
        
        # avoid division by zero in areas with no patches
        self.accumilated_weights[self.accumilated_weights == 0] = 1.0
        
        result = self.accumulated_values / self.accumilated_weights
        
        # if single class. squeeze the channel for convenience
        if self.num_classes == 1:
            return result.squeeze(0)
        return result