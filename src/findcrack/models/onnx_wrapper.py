import numpy as np
import cv2
from findcrack.utils import sigmoid_np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class ONNXModelWrapper(nn.Module):
        """
        Wraps an ONNX Runtime InferenceSession inside a PyTorch nn.Module.
        This allows running inference on ONNX models using the exact same code
        and APIs as PyTorch models, supporting both CPU/GPU tensor operations
        and test-time augmentation (TTA) pipelines.
        
        Automatically handles input resizing and YOLOv8-seg post-processing.
        """
        def __init__(self, model_path: str, device: str = "cpu"):
            super().__init__()
            import onnxruntime as ort
            
            # Select execution providers based on the target device
            if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            
            # Detect if this is a YOLO-seg model with NMS built-in
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.is_yolo_seg = False
            
            if len(self.output_names) >= 2:
                out0_shape = self.session.get_outputs()[0].shape
                out1_shape = self.session.get_outputs()[1].shape
                # YOLOv8-seg E2E output structure: [1, 300, 38] and [1, 32, 128, 128]
                if len(out0_shape) == 3 and out0_shape[2] == 38 and len(out1_shape) == 4 and out1_shape[1] == 32:
                    self.is_yolo_seg = True
                    
            self.output_name = self.output_names[0]
            
            # Get expected input dimensions
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) == 4:
                h_dim, w_dim = 2, 3
                if isinstance(input_shape[3], int) and input_shape[3] in (1, 3):
                    # NHWC format
                    h_dim, w_dim = 1, 2
                
                h_expected = input_shape[h_dim]
                w_expected = input_shape[w_dim]
                self.expected_size = (
                    h_expected if isinstance(h_expected, int) else None,
                    w_expected if isinstance(w_expected, int) else None
                )
            else:
                self.expected_size = (None, None)

        def _postprocess_yolo_seg(self, output0: np.ndarray, output1: np.ndarray, target_shape: tuple) -> np.ndarray:
            h_img, w_img = target_shape
            combined_mask = np.zeros((h_img, w_img), dtype=np.float32)
            
            boxes = output0[0, :, 0:4]          # shape (300, 4)
            scores = output0[0, :, 4]           # shape (300,)
            coeffs = output0[0, :, 6:38]        # shape (300, 32)
            protos = output1[0]                 # shape (32, 128, 128)
            
            keep = scores > 0.25
            if not np.any(keep):
                return np.expand_dims(np.expand_dims(np.full((h_img, w_img), -10.0, dtype=np.float32), axis=0), axis=0)
                
            valid_boxes = boxes[keep]
            valid_coeffs = coeffs[keep]
            
            N = valid_boxes.shape[0]
            masks_in = np.matmul(valid_coeffs, protos.reshape(32, -1)).reshape(N, 128, 128)
            
            for i in range(N):
                box = valid_boxes[i]
                mask = sigmoid_np(masks_in[i])
                mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                
                in_h = self.expected_size[0] or 512.0
                in_w = self.expected_size[1] or 512.0
                
                x1, y1, x2, y2 = box
                x1 = int(round(x1 * w_img / in_w))
                y1 = int(round(y1 * h_img / in_h))
                x2 = int(round(x2 * w_img / in_w))
                y2 = int(round(y2 * h_img / in_h))
                
                x1 = max(0, min(x1, w_img))
                y1 = max(0, min(y1, h_img))
                x2 = max(0, min(x2, w_img))
                y2 = max(0, min(y2, h_img))
                
                crop_mask = np.zeros_like(mask_resized)
                crop_mask[y1:y2, x1:x2] = 1.0
                mask_resized = mask_resized * crop_mask
                
                combined_mask = np.maximum(combined_mask, mask_resized)
                
            p = np.clip(combined_mask, 1e-6, 1.0 - 1e-6)
            logits = np.log(p / (1.0 - p))
            return np.expand_dims(np.expand_dims(logits, axis=0), axis=0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_size = (x.shape[2], x.shape[3]) # (H, W)
            x_device = x.device
            
            # 1. Resize input to expected shape if model requires a fixed input size
            if self.expected_size[0] and self.expected_size[1]:
                target_size = self.expected_size
                if orig_size != target_size:
                    x = torch.nn.functional.interpolate(
                        x, size=target_size, mode="bilinear", align_corners=False
                    )
            
            # 2. Convert PyTorch tensor to NumPy array
            x_np = x.detach().cpu().numpy().astype(np.float32)
            
            # 3. Run inference
            if self.is_yolo_seg:
                outputs = self.session.run(self.output_names, {self.input_name: x_np})
                logit_np = self._postprocess_yolo_seg(outputs[0], outputs[1], orig_size)
                out_tensor = torch.from_numpy(logit_np).to(x_device)
                return out_tensor
            else:
                outputs = self.session.run([self.output_name], {self.input_name: x_np})
                out_tensor = torch.from_numpy(outputs[0]).to(x_device)
                
                if out_tensor.shape[2:] != orig_size:
                    out_tensor = torch.nn.functional.interpolate(
                        out_tensor, size=orig_size, mode="bilinear", align_corners=False
                    )
                return out_tensor
else:
    class ONNXModelWrapper:
        """
        Wraps an ONNX Runtime InferenceSession for pure NumPy inference.
        
        Automatically handles input resizing and YOLOv8-seg post-processing.
        """
        def __init__(self, model_path: str, device: str = "cpu"):
            import onnxruntime as ort
            
            if device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.is_yolo_seg = False
            
            if len(self.output_names) >= 2:
                out0_shape = self.session.get_outputs()[0].shape
                out1_shape = self.session.get_outputs()[1].shape
                if len(out0_shape) == 3 and out0_shape[2] == 38 and len(out1_shape) == 4 and out1_shape[1] == 32:
                    self.is_yolo_seg = True
                    
            self.output_name = self.output_names[0]
            
            # Get expected input dimensions
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) == 4:
                h_dim, w_dim = 2, 3
                if isinstance(input_shape[3], int) and input_shape[3] in (1, 3):
                    h_dim, w_dim = 1, 2
                
                h_expected = input_shape[h_dim]
                w_expected = input_shape[w_dim]
                self.expected_size = (
                    h_expected if isinstance(h_expected, int) else None,
                    w_expected if isinstance(w_expected, int) else None
                )
            else:
                self.expected_size = (None, None)

        def _postprocess_yolo_seg(self, output0: np.ndarray, output1: np.ndarray, target_shape: tuple) -> np.ndarray:
            h_img, w_img = target_shape
            combined_mask = np.zeros((h_img, w_img), dtype=np.float32)
            
            boxes = output0[0, :, 0:4]          # shape (300, 4)
            scores = output0[0, :, 4]           # shape (300,)
            coeffs = output0[0, :, 6:38]        # shape (300, 32)
            protos = output1[0]                 # shape (32, 128, 128)
            
            keep = scores > 0.25
            if not np.any(keep):
                return np.expand_dims(np.expand_dims(np.full((h_img, w_img), -10.0, dtype=np.float32), axis=0), axis=0)
                
            valid_boxes = boxes[keep]
            valid_coeffs = coeffs[keep]
            
            N = valid_boxes.shape[0]
            masks_in = np.matmul(valid_coeffs, protos.reshape(32, -1)).reshape(N, 128, 128)
            
            for i in range(N):
                box = valid_boxes[i]
                mask = sigmoid_np(masks_in[i])
                mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                
                in_h = self.expected_size[0] or 512.0
                in_w = self.expected_size[1] or 512.0
                
                x1, y1, x2, y2 = box
                x1 = int(round(x1 * w_img / in_w))
                y1 = int(round(y1 * h_img / in_h))
                x2 = int(round(x2 * w_img / in_w))
                y2 = int(round(y2 * h_img / in_h))
                
                x1 = max(0, min(x1, w_img))
                y1 = max(0, min(y1, h_img))
                x2 = max(0, min(x2, w_img))
                y2 = max(0, min(y2, h_img))
                
                crop_mask = np.zeros_like(mask_resized)
                crop_mask[y1:y2, x1:x2] = 1.0
                mask_resized = mask_resized * crop_mask
                
                combined_mask = np.maximum(combined_mask, mask_resized)
                
            p = np.clip(combined_mask, 1e-6, 1.0 - 1e-6)
            logits = np.log(p / (1.0 - p))
            return np.expand_dims(np.expand_dims(logits, axis=0), axis=0)

        def __call__(self, x: np.ndarray) -> np.ndarray:
            orig_size = (x.shape[2], x.shape[3]) # (H, W)
            
            # 1. Resize input to expected shape
            if self.expected_size[0] and self.expected_size[1]:
                target_size = self.expected_size
                if orig_size != target_size:
                    x_hwc = np.transpose(x[0], (1, 2, 0))
                    x_resized = cv2.resize(
                        x_hwc, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
                    )
                    if len(x_resized.shape) == 2:
                        x_resized = np.expand_dims(x_resized, axis=-1)
                    x = np.expand_dims(np.transpose(x_resized, (2, 0, 1)), axis=0)
            
            # 2. Run inference
            if self.is_yolo_seg:
                outputs = self.session.run(self.output_names, {self.input_name: x.astype(np.float32)})
                return self._postprocess_yolo_seg(outputs[0], outputs[1], orig_size)
            else:
                outputs = self.session.run([self.output_name], {self.input_name: x.astype(np.float32)})
                out_np = outputs[0]
                
                if out_np.shape[2:] != orig_size:
                    out_hwc = np.transpose(out_np[0], (1, 2, 0))
                    out_resized = cv2.resize(
                        out_hwc, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_LINEAR
                    )
                    if len(out_resized.shape) == 2:
                        out_resized = np.expand_dims(out_resized, axis=-1)
                    out_np = np.expand_dims(np.transpose(out_resized, (2, 0, 1)), axis=0)
                    
                return out_np
