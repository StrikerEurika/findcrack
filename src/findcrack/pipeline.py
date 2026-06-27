import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from .preprocess import Preprocessor
from .tta import tta_forward
from .preprocess.patching import CountMapBlender, SlidingWindowExtractor
from .models import load_model


class CrackInferencePipeline:
    """
    pipeline for running inference on large images.
    """
    def __init__(self, model: torch.nn.Module, device: str = "cuda",
                 patch_size: int = 512, overlap_ratio: float = 0.2, 
                 confidence_threhold: float = 0.5, use_tta: bool = False,
                 preprocessor: Preprocessor = None, use_clahe: bool = True,
                 clahe_clip_limit: float = 2.0):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threhold
        self.use_tta = use_tta
        
        self.extractor = SlidingWindowExtractor(self.patch_size, self.overlap_ratio)
        
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = Preprocessor(use_clahe=use_clahe, clip_limit=clahe_clip_limit)
        
        # Keep transform attribute for backwards compatibility
        self.transform = self.preprocessor.transform
        
    @classmethod
    def from_checkpoint(cls, model_class, checkpoint_path: str, **kwargs):
        """
        Helper function to load model
        """
        model = model_class(n_channels=3, n_classes=1)  # Assuming binary segmentation
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return cls(model, **kwargs)
    
    @classmethod
    def from_pretrained(cls, variant: str, device: str = "cuda", **kwargs):
        """
        Helper function to load pretrained model from the model zoo.
        """
        model = load_model(variant, device=device)
        return cls(model, device=device, **kwargs)
        
    
    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Runs full inference pipeline on a large image.
        Returns a dictionary with the original image, probability map, and binary mask.
        """
        # Load Image
        original_image = np.array(Image.open(image_path).convert('RGB'))
        height, width, _ = original_image.shape
        
        # Preprocess (LAB-CLAHE)
        preprocessed_image = self.preprocessor.enhance_contrast(original_image)
        
        # Initialize Blender
        blender = CountMapBlender(shape=(height, width))
        
         # Sliding Window Inference
        for patch_rgb, coordinates in self.extractor.extract(preprocessed_image):
            # Transform to tensor
            patch_tensor = self.preprocessor.transform_patch(patch_rgb).to(self.device)
            
            # Run Model
            if self.use_tta:
                pred_prob = tta_forward(self.model, patch_tensor)
            else:
                pred_prob = torch.sigmoid(self.model(patch_tensor.unsqueeze(0))).squeeze()
                
            # Add to blender
            blender.add(pred_prob.cpu().numpy(), coordinates)
            
        # Merge and Threshold
        confidence_map = blender.merge()
        binary_mask = (confidence_map > self.confidence_threshold).astype(np.uint8) * 255
        
        return {
            "original_image": original_image,
            "confidence_map": confidence_map,
            "binary_mask": binary_mask
        }