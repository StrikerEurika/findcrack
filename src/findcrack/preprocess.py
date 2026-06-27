import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from typing import Tuple, List, Optional

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

def get_inference_transform(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    """
    standard ImageNet normalization required by most pretrained models.
    """
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

class Preprocessor:
    """
    Standard preprocessing pipeline class.
    Handles optional LAB-CLAHE contrast enhancement and Albumentations normalization/tensorization.
    """
    def __init__(
        self,
        use_clahe: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        additional_transforms: Optional[List[A.BasicTransform]] = None
    ):
        self.use_clahe = use_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.mean = mean
        self.std = std
        
        transforms = []
        if additional_transforms:
            transforms.extend(additional_transforms)
        transforms.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])
        self.transform = A.Compose(transforms)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Applies LAB-CLAHE contrast enhancement if enabled.
        """
        if self.use_clahe:
            return apply_lab_clahe(image, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        return image

    def transform_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Applies Albumentations normalization and tensorization to a patch or image.
        """
        transformed = self.transform(image=patch)
        return transformed["image"]

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Performs full preprocessing: contrast enhancement followed by transform.
        Returns:
            - enhanced_image: Contrast-enhanced RGB image (numpy array).
            - tensor: The normalized PyTorch tensor.
        """
        enhanced = self.enhance_contrast(image)
        tensor = self.transform_patch(enhanced)
        return enhanced, tensor

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PIL import Image

    parser = argparse.ArgumentParser(description="Preprocess images for crack detection using LAB-CLAHE.")
    parser.add_argument("input", type=str, help="Path to input image file or directory.")
    parser.add_argument("output", type=str, help="Path to output preprocessed image file or directory.")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit (contrast enhancement factor).")
    parser.add_argument("--tile-grid-size", type=int, nargs=2, default=[8, 8], help="CLAHE tile grid size (width height).")
    parser.add_argument("--no-clahe", action="store_true", help="Disable LAB-CLAHE contrast enhancement.")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    use_clahe = not args.no_clahe
    tile_grid = tuple(args.tile_grid_size)

    preprocessor = Preprocessor(
        use_clahe=use_clahe,
        clip_limit=args.clip_limit,
        tile_grid_size=tile_grid
    )

    def process_file(in_file: Path, out_file: Path):
        try:
            image = np.array(Image.open(in_file).convert("RGB"))
            enhanced = preprocessor.enhance_contrast(image)
            Image.fromarray(enhanced).save(out_file)
            print(f"Processed: {in_file} -> {out_file}")
        except Exception as e:
            print(f"Error processing {in_file}: {e}")

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.is_dir():
            print(f"Error: Output path must be a directory because input path is a directory.")
            exit(1)

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        for file in input_path.iterdir():
            if file.suffix.lower() in valid_exts:
                process_file(file, output_path / file.name)
    else:
        if output_path.is_dir():
            process_file(input_path, output_path / input_path.name)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            process_file(input_path, output_path)