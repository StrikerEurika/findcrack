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
        Backward-compatible wrapper that delegates to the backends package.

        Retains nn.Module inheritance so isinstance checks in the pipeline
        continue to work, but actual inference is routed to the optimal
        backend (ONNX CPU or ONNX CUDA with IOBinding).
        """
        def __init__(self, model_path: str, device: str = "cpu"):
            super().__init__()
            from findcrack.backends import get_backend
            self._backend = get_backend(model_path=model_path, device=device, backend="onnx")
            self._device = self._backend.device

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x_np = x.detach().cpu().numpy()
            out_np = self._backend(x_np)
            return torch.from_numpy(out_np).to(x.device)

        @property
        def is_yolo_seg(self):
            return getattr(self._backend, "is_yolo_seg", False)

        @property
        def baked_normalization(self):
            return getattr(self._backend, "baked_normalization", False)

        @property
        def expected_size(self):
            return getattr(self._backend, "expected_size", (None, None))
else:
    class ONNXModelWrapper:
        """Backward-compatible wrapper for torch-free environments."""
        def __init__(self, model_path: str, device: str = "cpu"):
            from findcrack.backends import get_backend
            self._backend = get_backend(model_path=model_path, device=device, backend="onnx")
            self._device = self._backend.device

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return self._backend(x)

        @property
        def is_yolo_seg(self):
            return getattr(self._backend, "is_yolo_seg", False)

        @property
        def baked_normalization(self):
            return getattr(self._backend, "baked_normalization", False)

        @property
        def expected_size(self):
            return getattr(self._backend, "expected_size", (None, None))
