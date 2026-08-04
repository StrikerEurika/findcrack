import numpy as np
from typing import Tuple, Generator, Union

class PatchExtractor:
    """
    Extracts overlapping patches from a large image.
    Supports both square and rectangular patch sizes.
    """
    def __init__(self, patch_size: Union[int, Tuple[int, int]], overlap_ratio: float = 0.2):
        """
        Args:
            patch_size: the size of the patch to be extracted (int or Tuple[int, int]).
            overlap_ratio: float number between 0.0 and 0.99.
        """
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be between 0.0 and 1.0")

        if isinstance(patch_size, int):
            self.patch_height = patch_size
            self.patch_width = patch_size
        else:
            self.patch_height, self.patch_width = patch_size
        
        self.stride_height = max(1, int(self.patch_height * (1 - overlap_ratio)))
        self.stride_width = max(1, int(self.patch_width * (1 - overlap_ratio)))

    def extract(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[int, int]], None, None]:
        """
        Yields patches and their top-left (y, x) coordinates.
        Handles edges by shifting the last patch to align with the image border.
        """
        image_height, image_width = image.shape[:2]
        if image_height < self.patch_height or image_width < self.patch_width:
            raise ValueError(
                f"Image dimensions ({image_height}, {image_width}) must be at least "
                f"as large as the patch size ({self.patch_height}, {self.patch_width})."
            )
        seen_coordinates = set()
        
        for y in range(0, image_height, self.stride_height):
            for x in range(0, image_width, self.stride_width):
                # Shift the patch if it goes out of bounds
                patch_y = min(y, image_height - self.patch_height) if y + self.patch_height > image_height else y
                patch_x = min(x, image_width - self.patch_width) if x + self.patch_width > image_width else x
                
                # Check shifted coordinates to avoid duplicate boundary patches
                if (patch_y, patch_x) in seen_coordinates:
                    continue
                seen_coordinates.add((patch_y, patch_x))

                yield image[patch_y:patch_y+self.patch_height, patch_x:patch_x+self.patch_width], (patch_y, patch_x)

    @staticmethod
    def is_active_patch(patch: np.ndarray, std_threshold: float = 12.0) -> bool:
        """
        Fast C++ OpenCV check to evaluate whether a patch contains edges/textures.

        Args:
            patch: NumPy array image patch.
            std_threshold: Intensity standard deviation threshold. Patches with stddev below
                           this threshold are considered featureless background.

        Returns:
            True if patch stddev >= std_threshold, False otherwise.
        """
        import cv2
        if patch.ndim == 3:
            if patch.shape[2] == 3:
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            elif patch.shape[2] == 4:
                gray = cv2.cvtColor(patch, cv2.COLOR_RGBA2GRAY)
            else:
                gray = patch[:, :, 0]
        else:
            gray = patch

        _, stddev = cv2.meanStdDev(gray)
        return float(stddev[0][0]) >= std_threshold