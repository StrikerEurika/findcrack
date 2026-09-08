import numpy as np
from findcrack.utils import sigmoid_np


class ONNXModelWrapper:
    """ONNX model wrapper that delegates to the backends package."""
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
