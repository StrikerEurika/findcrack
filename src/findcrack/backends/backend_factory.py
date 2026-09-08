from __future__ import annotations
from typing import Any
import os

from .base import Backend
from .onnx_cpu_backend import ONNXCPUBackend


def normalize_device(device: str) -> str:
    """Normalize common device aliases to canonical strings."""
    d = str(device).lower().strip()
    if d in ("gpu", "cuda"):
        return "cuda"
    if d in ("cpu",):
        return "cpu"
    if d in ("mps", "apple"):
        return "mps"
    return d


def _ort_providers() -> list:
    """Return execution providers exposed by the installed onnxruntime."""
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except ImportError:
        return []


def _ort_cuda_available() -> bool:
    """True when onnxruntime was built with GPU support (e.g. onnxruntime-gpu)."""
    return "CUDAExecutionProvider" in _ort_providers()


def get_backend(
    model_path: str = None,
    device: str = "cuda",
    backend: str = None,
) -> Backend:
    """Factory that selects the optimal backend for the given device and model."""

    normalized = normalize_device(device)

    if backend == "onnx" or (model_path and model_path.lower().endswith(".onnx")):
        # ONNX resource selection is driven by onnxruntime providers
        if normalized != "cpu" and _ort_cuda_available():
            try:
                from .onnx_cuda_backend import ONNXCUDABackend
                return ONNXCUDABackend(model_path)
            except (ImportError, RuntimeError) as e:
                print(
                    f"WARNING: CUDA ONNX backend unavailable ({e}). "
                    "Falling back to CPU."
                )
        elif normalized == "cuda":
            print(
                "WARNING: device='cuda' requested but no CUDAExecutionProvider "
                "found (install onnxruntime-gpu). Falling back to CPU."
            )
        return ONNXCPUBackend(model_path)

    raise ValueError(f"Unsupported backend: {backend}")
