from __future__ import annotations
from typing import Any, Dict, List, Optional


def get_backend_info(backend: Any) -> Dict[str, Any]:
    """Return a summary dict of backend capabilities and active providers.

    Works with any findcrack backend instance (ONNX CPU/CUDA).
    """
    info: Dict[str, Any] = {
        "backend_type": getattr(backend, "backend_type", "unknown"),
        "device": getattr(backend, "device", "unknown"),
        "is_gpu": getattr(backend, "is_gpu", lambda: False)(),
    }

    session = getattr(backend, "session", None)
    if session is not None:
        try:
            info["available_providers"] = list(session.get_providers())
        except Exception:
            info["available_providers"] = []
    else:
        info["available_providers"] = []

    return info


def verify_gpu(model_path: str, device: str = "cuda", backend: str = "onnx") -> Dict[str, Any]:
    """Verify that ONNX Runtime is actually using GPU for the given model.

    Returns a dict with:
        - success (bool): True if GPU is active
        - backend_type (str)
        - device (str)
        - providers (list)
        - error (str | None)

    Example:
        >>> from findcrack.backends.gpu_utils import verify_gpu
        >>> result = verify_gpu("model.onnx", device="cuda")
        >>> print(result)
    """
    from .backend_factory import get_backend

    result: Dict[str, Any] = {
        "success": False,
        "backend_type": None,
        "device": None,
        "providers": [],
        "error": None,
    }

    try:
        b = get_backend(model_path=model_path, device=device, backend=backend)
    except Exception as e:
        result["error"] = f"Backend creation failed: {e}"
        return result

    info = get_backend_info(b)
    result["backend_type"] = info["backend_type"]
    result["device"] = info["device"]
    result["providers"] = info["available_providers"]

    if info["is_gpu"] and "CUDAExecutionProvider" in info["available_providers"]:
        result["success"] = True
    else:
        result["error"] = (
            f"GPU not active. backend_type={info['backend_type']}, "
            f"device={info['device']}, providers={info['available_providers']}"
        )

    return result


def _cuda_runtime_available() -> bool:
    """Attempt a real CUDA EP session creation to test runtime availability.

    ORT's get_available_providers() is build-time only. CUDA EP can still
    fail at runtime if cuBLAS/cuDNN DLLs are missing or mismatched.
    This function registers CUDA runtime DLL dirs, creates a minimal ONNX
    model, and tries to load it with CUDAExecutionProvider to verify the
    runtime is actually functional.
    """
    try:
        from .onnx_cuda_backend import _register_cuda_dll_dirs
        _register_cuda_dll_dirs()
    except Exception:
        pass

    try:
        import onnxruntime as ort

        # Serialized minimal ONNX model bytes to avoid depending on the 'onnx' package at runtime
        model_bytes = (
            b'\x08\r:=\n\x0e\n\x01X\n\x01X\x12\x01Y"\x03Add\x12\x01gZ\x13\n\x01X\x12\x0e\n\x0c'
            b'\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x01b\x13\n\x01Y\x12\x0e\n\x0c\x08\x01\x12'
            b'\x08\n\x02\x08\x01\n\x02\x08\x01B\x04\n\x00\x10\x10'
        )

        sess = ort.InferenceSession(
            model_bytes,
            providers=["CUDAExecutionProvider"],
        )
        return "CUDAExecutionProvider" in sess.get_providers()
    except Exception:
        return False


def list_available_devices() -> List[Dict[str, Any]]:
    """List devices available for ONNX inference without requiring a model file.

    CUDA availability is determined by a real runtime check (session creation),
    not just build-time provider listing. This avoids the false-positive case
    where CUDAExecutionProvider is built in but fails to load due to missing
    cuBLAS/cuDNN DLLs.

    Returns a list of dicts with keys: name, available, providers.
    Example:
        >>> from findcrack.backends.gpu_utils import list_available_devices
        >>> list_available_devices()
        [{'name': 'cpu', 'available': True, 'providers': ['CPUExecutionProvider']},
         {'name': 'cuda', 'available': True, 'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']}]
    """
    devices: List[Dict[str, Any]] = []

    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
    except ImportError:
        providers = []

    cuda_runtime_ok = _cuda_runtime_available()

    devices.append({
        "name": "cpu",
        "available": "CPUExecutionProvider" in providers if providers else True,
        "providers": providers,
    })

    devices.append({
        "name": "cuda",
        "available": cuda_runtime_ok,
        "providers": providers if cuda_runtime_ok else [p for p in providers if p != "CUDAExecutionProvider"],
    })

    return devices
