from __future__ import annotations
import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from ..preprocess import Preprocessor, PatchExtractor, ImageScaler
from ..postprocess import PatchBlender
from .tta import tta_forward, tta_forward_np
from ..utils import sigmoid_np


class CrackInferencePipeline:
    """
    Pipeline for running inference on large images.
    Uses the backends package for device-aware inference execution.
    """

    def __init__(self, model=None, device: str = "cuda",
                 patch_size: int = 512, overlap_ratio: float = 0.2,
                 confidence_threshold: float = 0.5, use_tta: bool = False,
                 preprocessor: Optional[Preprocessor] = None, use_clahe: bool = True,
                 clahe_clip_limit: float = 2.0,
                 overlay_alpha: float = 0.4,
                 overlay_color: tuple = (255, 0, 0),
                 box_color: tuple = (0, 255, 0),
                 box_thickness: int = 2,
                 contour_color: tuple = (0, 0, 255),
                 contour_thickness: int = 2,
                 blend_mode: str = "average",
                 batch_size: int = 1,
                 scale_factor: Optional[float] = None,
                 max_dim: Optional[int] = None,
                 scaler: Optional[ImageScaler] = None,
                 filter_background: bool = False,
                 patch_std_threshold: Optional[float] = None):

        self.backend = self._resolve_backend(model, device)

        self.device = getattr(self.backend, "device", device)
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        self.use_tta = use_tta
        self.blend_mode = blend_mode
        self.batch_size = batch_size

        if patch_std_threshold is not None:
            self.patch_std_threshold = patch_std_threshold
        elif filter_background:
            self.patch_std_threshold = 12.0
        else:
            self.patch_std_threshold = None

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

        if scaler is not None:
            self.scaler = scaler
        elif scale_factor is not None or max_dim is not None:
            self.scaler = ImageScaler(scale_factor=scale_factor, max_dim=max_dim)
        else:
            self.scaler = None

        self.transform = self.preprocessor.transform

    @staticmethod
    def _resolve_backend(model, device: str):
        """Resolve a model instance or path to an inference backend."""
        if model is None:
            return None

        if hasattr(model, "_backend") and hasattr(model._backend, "backend_type"):
            return model._backend

        if hasattr(model, "backend_type"):
            return model

        if isinstance(model, str) and model.lower().endswith(".onnx"):
            from findcrack.backends import get_backend
            return get_backend(model_path=model, device=device, backend="onnx")

        if HAS_TORCH and isinstance(model, torch.nn.Module):
            from findcrack.backends import PyTorchBackend
            return PyTorchBackend(model, device)

        return model

    @classmethod
    def from_checkpoint(cls, model_class, checkpoint_path: str, **kwargs):
        """
        Helper function to load model from a checkpoint.
        """
        if not HAS_TORCH:
            raise ImportError(
                "Loading a model from a checkpoint requires PyTorch. Please install PyTorch or "
                "install findcrack with standard extras: pip install findcrack[standard]"
            )
        model = model_class(n_channels=3, n_classes=1)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return cls(model, **kwargs)

    @classmethod
    def from_pretrained(cls, variant: str, device: str = "cuda", local_checkpoint: bool = False, **kwargs):
        """
        Helper function to load pretrained model from the model registry.
        """
        from ..models import load_model
        model = load_model(variant, device=device, local_checkpoint=local_checkpoint)
        return cls(model, device=device, **kwargs)

    def _prepare_input(self, patch_data, batch_size: int):
        """Convert patch data to the format expected by the backend."""
        is_pytorch = getattr(self.backend, "backend_type", None) == "pytorch"

        if is_pytorch and HAS_TORCH and isinstance(patch_data, torch.Tensor):
            if not isinstance(patch_data, torch.Tensor):
                patch_tensor = torch.from_numpy(patch_data).float()
            else:
                patch_tensor = patch_data
            return patch_tensor.to(self.device), True
        else:
            if HAS_TORCH and isinstance(patch_data, torch.Tensor):
                patch_np = patch_data.cpu().numpy()
            else:
                patch_np = patch_data
            return np.expand_dims(patch_np.astype(np.float32), axis=0), False

    def _run_inference(self, input_data, is_torch: bool):
        """Run backend inference and return probability map as numpy."""
        if self.use_tta:
            if is_torch:
                pred = tta_forward(self.backend.model, input_data.squeeze(0))
                return pred.cpu().numpy()
            else:
                return tta_forward_np(self.backend, input_data.squeeze(0))
        else:
            raw_out = self.backend(input_data)
            if is_torch:
                return torch.sigmoid(raw_out).squeeze().cpu().numpy()
            else:
                return np.squeeze(sigmoid_np(raw_out))

    def predict(self, image_input: str | Path | np.ndarray | Image.Image) -> dict:
        """
        Runs full inference pipeline on an image (path, numpy array, or PIL Image).
        Returns a dictionary with:
        - original_image: Original RGB image (numpy array).
        - confidence_map: Float probability map [0.0 - 1.0].
        - binary_mask: Binary segmentation mask [0 or 255].
        - overlay: Original image with a colored transparent overlay on the cracks.
        - bounding_boxes: List of [xmin, ymin, xmax, ymax] boxes for detected crack components.
        - contours: List of segmentation contours for detected cracks.
        - visualization: Original image with bounding boxes drawn and contours outlined.
        """
        if isinstance(image_input, (str, Path)):
            original_image = np.array(Image.open(image_input).convert('RGB'))
        elif isinstance(image_input, Image.Image):
            original_image = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                original_image = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                original_image = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                original_image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        height, width, _ = original_image.shape

        if self.scaler is not None:
            processed_image, is_scaled = self.scaler.downscale(original_image)
        else:
            processed_image, is_scaled = original_image, False

        proc_height, proc_width, _ = processed_image.shape

        if isinstance(self.patch_size, int):
            ph, pw = self.patch_size, self.patch_size
        else:
            ph, pw = self.patch_size

        is_pytorch_backend = getattr(self.backend, "backend_type", None) == "pytorch"

        if proc_height <= ph and proc_width <= pw:
            preprocessed_image = self.preprocessor.enhance_contrast(processed_image)
            patch_data = self.preprocessor.transform_patch(preprocessed_image)

            input_data, is_torch = self._prepare_input(patch_data, batch_size=1)
            confidence_map = self._run_inference(input_data, is_torch=is_pytorch_backend and is_torch)

            if confidence_map.shape != (proc_height, proc_width):
                confidence_map = cv2.resize(confidence_map, (proc_width, proc_height), interpolation=cv2.INTER_LINEAR)
        else:
            preprocessed_image = self.preprocessor.enhance_contrast(processed_image)
            blender = PatchBlender(shape=preprocessed_image.shape[:2], blend_mode=self.blend_mode)

            is_yolo_seg = getattr(self.backend, "is_yolo_seg", False)
            actual_batch_size = 1 if (self.use_tta or is_yolo_seg) else self.batch_size

            batch_patches = []
            batch_coords = []

            for patch_rgb, coordinates in self.extractor.extract(preprocessed_image):
                time.sleep(0.001)

                if self.patch_std_threshold is not None:
                    if not PatchExtractor.is_active_patch(patch_rgb, std_threshold=self.patch_std_threshold):
                        ph_c, pw_c = patch_rgb.shape[:2]
                        blender.add(np.zeros((ph_c, pw_c), dtype=np.float32), coordinates)
                        continue

                patch_data = self.preprocessor.transform_patch(patch_rgb)

                if actual_batch_size == 1:
                    input_data, is_torch = self._prepare_input(patch_data, batch_size=1)
                    pred_prob_np = self._run_inference(input_data, is_torch=is_pytorch_backend and is_torch)
                    blender.add(pred_prob_np, coordinates)
                else:
                    batch_patches.append(patch_data)
                    batch_coords.append(coordinates)

                    if len(batch_patches) == actual_batch_size:
                        self._process_batch(batch_patches, batch_coords, blender)
                        batch_patches = []
                        batch_coords = []

            if len(batch_patches) > 0:
                self._process_batch(batch_patches, batch_coords, blender)

            confidence_map = blender.merge()
            del blender

        if is_scaled and self.scaler is not None:
            confidence_map = self.scaler.upscale_map(confidence_map, (height, width))

        if confidence_map.shape != (height, width):
            confidence_map = cv2.resize(confidence_map, (width, height), interpolation=cv2.INTER_LINEAR)

        binary_mask = (confidence_map > self.confidence_threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        valid_contours = []
        for ctr in contours:
            if cv2.contourArea(ctr) > 5:
                x, y, w, h = cv2.boundingRect(ctr)
                bounding_boxes.append([x, y, x + w, y + h])
                valid_contours.append(ctr)

        overlay = original_image.copy()
        overlay[binary_mask > 0] = self.overlay_color
        overlay_blended = cv2.addWeighted(overlay, self.overlay_alpha, original_image, 1 - self.overlay_alpha, 0)

        visualization = original_image.copy()
        for box in bounding_boxes:
            cv2.rectangle(visualization, (box[0], box[1]), (box[2], box[3]), self.box_color, self.box_thickness)
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

    def _process_batch(self, batch_patches, batch_coords, blender):
        is_pytorch_backend = getattr(self.backend, "backend_type", None) == "pytorch"

        if is_pytorch_backend and HAS_TORCH:
            tensors = []
            for p in batch_patches:
                if not isinstance(p, torch.Tensor):
                    tensors.append(torch.from_numpy(p).float())
                else:
                    tensors.append(p)
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                logits = self.backend(batch_tensor)
            pred_probs = torch.sigmoid(logits)

            for i, coordinates in enumerate(batch_coords):
                pred_prob_np = pred_probs[i, 0].cpu().numpy()
                blender.add(pred_prob_np, coordinates)
        else:
            arrays = []
            for p in batch_patches:
                if HAS_TORCH and isinstance(p, torch.Tensor):
                    arrays.append(p.cpu().numpy())
                else:
                    arrays.append(p)
            batch_np = np.stack(arrays).astype(np.float32)
            raw_out = self.backend(batch_np)
            pred_probs_np = sigmoid_np(raw_out)

            for i, coordinates in enumerate(batch_coords):
                pred_prob_np = pred_probs_np[i, 0]
                blender.add(pred_prob_np, coordinates)
