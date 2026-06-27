from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from ..preprocess import Preprocessor, PatchExtractor
from ..postprocess import PatchBlender
from .tta import tta_forward, tta_forward_np
from ..models import load_model


class CrackInferencePipeline:
    """
    pipeline for running inference on large images.
    """
    def __init__(self, model, device: str = "cuda",
                 patch_size: int = 512, overlap_ratio: float = 0.2, 
                 confidence_threshold: float = 0.5, use_tta: bool = False,
                 preprocessor: Optional[Preprocessor]  = None, use_clahe: bool = True,
                 clahe_clip_limit: float = 2.0,
                 overlay_alpha: float = 0.4,
                 overlay_color: tuple = (255, 0, 0),
                 box_color: tuple = (0, 255, 0),
                 box_thickness: int = 2,
                 contour_color: tuple = (0, 0, 255),
                 contour_thickness: int = 2):
        if HAS_TORCH:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            if isinstance(model, torch.nn.Module):
                self.model = model.to(self.device).eval()
            else:
                self.model = model
        else:
            self.device = device
            self.model = model
            
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        self.use_tta = use_tta
        
        self.overlay_alpha = overlay_alpha
        self.overlay_color = overlay_color
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.contour_color = contour_color
        self.contour_thickness = contour_thickness
        
        self.extractor = PatchExtractor(self.patch_size, self.overlap_ratio)
        
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
        if not HAS_TORCH:
            raise ImportError(
                "Loading a model from a checkpoint requires PyTorch. Please install PyTorch or "
                "install findcrack with standard extras: pip install findcrack[standard]"
            )
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
        
    
    def predict(self, image_path: str) -> dict:
        """
        Runs full inference pipeline on a large image.
        Returns a dictionary with:
        - original_image: Original RGB image (numpy array).
        - confidence_map: Float probability map [0.0 - 1.0].
        - binary_mask: Binary segmentation mask [0 or 255].
        - overlay: Original image with a colored transparent overlay on the cracks.
        - bounding_boxes: List of [xmin, ymin, xmax, ymax] boxes for detected crack components.
        - contours: List of segmentation contours for detected cracks.
        - visualization: Original image with bounding boxes drawn and contours outlined.
        """
        # Load Image
        original_image = np.array(Image.open(image_path).convert('RGB'))
        height, width, _ = original_image.shape
        
        # Preprocess (LAB-CLAHE)
        preprocessed_image = self.preprocessor.enhance_contrast(original_image)
        
        # Initialize Blender
        blender = PatchBlender(shape=(height, width))
        
        # Context manager for torch gradient tracking
        class DummyContext:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass

        context = torch.no_grad() if HAS_TORCH else DummyContext()
        use_torch_inference = HAS_TORCH and isinstance(self.model, torch.nn.Module)
        
        with context:
            # Sliding Window Inference
            for patch_rgb, coordinates in self.extractor.extract(preprocessed_image):
                # Transform to tensor/array
                patch_data = self.preprocessor.transform_patch(patch_rgb)
                
                # Run Model
                if use_torch_inference:
                    if not isinstance(patch_data, torch.Tensor):
                        patch_tensor = torch.from_numpy(patch_data).float()
                    else:
                        patch_tensor = patch_data
                    patch_tensor = patch_tensor.to(self.device)
                    
                    if self.use_tta:
                        pred_prob = tta_forward(self.model, patch_tensor)
                    else:
                        pred_prob = torch.sigmoid(self.model(patch_tensor.unsqueeze(0))).squeeze()
                        
                    pred_prob_np = pred_prob.cpu().numpy()
                else:
                    # Pure NumPy / ONNX path
                    if HAS_TORCH and isinstance(patch_data, torch.Tensor):
                        patch_data_np = patch_data.cpu().numpy()
                    else:
                        patch_data_np = patch_data
                        
                    if self.use_tta:
                        pred_prob_np = tta_forward_np(self.model, patch_data_np)
                    else:
                        # unsqueeze equivalent
                        input_feed = np.expand_dims(patch_data_np, axis=0)
                        raw_out = self.model(input_feed)
                        # sigmoid equivalent
                        pred_prob_np = np.squeeze(1 / (1 + np.exp(-raw_out)))
                    
                # Add to blender
                blender.add(pred_prob_np, coordinates)
            
        # Merge and Threshold
        confidence_map = blender.merge()
        binary_mask = (confidence_map > self.confidence_threshold).astype(np.uint8) * 255
        
        # Extract connected component contours and bounding boxes
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        valid_contours = []
        for ctr in contours:
            # Filter out tiny noise components (area > 5 pixels)
            if cv2.contourArea(ctr) > 5:
                x, y, w, h = cv2.boundingRect(ctr)
                bounding_boxes.append([x, y, x + w, y + h])
                valid_contours.append(ctr)
                
        # Generate transparent overlay (original blended with solid color mask)
        overlay = original_image.copy()
        overlay[binary_mask > 0] = self.overlay_color
        overlay_blended = cv2.addWeighted(overlay, self.overlay_alpha, original_image, 1 - self.overlay_alpha, 0)
        
        # Generate full visualization with bounding boxes and contour boundaries
        visualization = original_image.copy()
        # Draw bounding boxes
        for box in bounding_boxes:
            cv2.rectangle(visualization, (box[0], box[1]), (box[2], box[3]), self.box_color, self.box_thickness)
        # Draw contours outline
        cv2.drawContours(visualization, valid_contours, -1, self.contour_color, self.contour_thickness)
        
        return {
            "original_image": original_image,
            "confidence_map": confidence_map,
            "binary_mask": binary_mask,
            "overlay": overlay_blended,
            "bounding_boxes": bounding_boxes,
            "contours": valid_contours,
            "visualization": visualization
        }
