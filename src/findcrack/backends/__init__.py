from __future__ import annotations
from .base import Backend

try:
    from .pytorch_backend import PyTorchBackend
except Exception:
    PyTorchBackend = None

try:
    from .onnx_cpu_backend import ONNXCPUBackend
except Exception:
    ONNXCPUBackend = None

try:
    from .onnx_cuda_backend import ONNXCUDABackend
except Exception:
    ONNXCUDABackend = None

try:
    from .backend_factory import get_backend, normalize_device
except Exception:
    get_backend = None
    normalize_device = None

try:
    from .gpu_utils import get_backend_info, verify_gpu, list_available_devices
except Exception:
    get_backend_info = None
    verify_gpu = None
    list_available_devices = None

__all__ = [
    "Backend",
    "PyTorchBackend",
    "ONNXCPUBackend",
    "ONNXCUDABackend",
    "get_backend",
    "normalize_device",
    "get_backend_info",
    "verify_gpu",
    "list_available_devices",
]
