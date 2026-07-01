from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.hub import get_dir, download_url_to_file
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from .onnx_wrapper import ONNXModelWrapper

# Load model registry detailing architectures, default arguments, remote URLs, hashes, and backend type.
REGISTRY_PATH = Path(__file__).parent / "registry.json"
with open(REGISTRY_PATH, "r") as f:
    MODEL_REGISTRY = json.load(f)


def verify_sha256(filepath: str, expected_hash: str) -> bool:
    """Verifies file integrity via SHA256 checksum."""
    if not expected_hash:
        return True
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash


def get_cache_dir() -> str:
    """Returns the package's weight caching directory."""
    if HAS_TORCH:
        hub_directory = get_dir()
    else:
        hub_directory = os.path.expanduser(os.path.join("~", ".cache", "torch", "hub"))
    model_directory = os.path.join(hub_directory, 'checkpoints', 'findcrack')
    os.makedirs(model_directory, exist_ok=True)
    return model_directory


def download_weight_file(url: str, cached_file: str, expected_hash: str = None):
    """Downloads remote weight file and verifies its checksum."""
    print(f"Downloading weights from {url}... (This is a one-time download)")
    try:
        if HAS_TORCH:
            download_url_to_file(url, cached_file, progress=True)
        else:
            import urllib.request
            print(f"Downloading {url} to {cached_file}...")
            urllib.request.urlretrieve(url, cached_file)
            print("Download completed.")
    except Exception as e:
        raise IOError(f"Failed to download weight file from {url}. Error: {e}")
        
    if expected_hash and not verify_sha256(cached_file, expected_hash):
        if os.path.exists(cached_file):
            os.remove(cached_file)
        raise ValueError(f"Checksum verification failed for {cached_file}. Cache cleared.")


def load_model(
    variant: str,
    device: str = "cpu",
    force_download: bool = False,
    architecture = None,
    kwargs: dict = None,
    backend: str = None,
    sha256: str = None
) -> Any:
    """
    Loads a model by its registry name OR directly from a remote HTTP(S) URL.
    
    Args:
        variant: Registry name (e.g. 'unet_cfd_v1') OR direct remote URL (e.g. 'https://.../model.pth').
        device: The target device for execution (e.g., 'cpu', 'cuda', 'mps').
        force_download: If True, deletes cached files and re-downloads weights.
        architecture: Model architecture class (e.g., UNet) - required if loading PyTorch weights from a URL.
        kwargs: Keyword arguments for instantiating the model class (used when loading from a URL).
        backend: Backend target ('pytorch' or 'onnx'). Automatically inferred from URL if loading from a URL.
        sha256: Optional SHA256 checksum to verify file integrity.
    """
    is_url = variant.startswith("http://") or variant.startswith("https://")
    
    if is_url:
        url = variant
        if backend is None:
            backend = "onnx" if url.lower().endswith(".onnx") else "pytorch"
            
        if backend == "pytorch" and architecture is None:
            raise ValueError(
                "You must specify the 'architecture' class (e.g., UNet, DeepCrack) "
                "when loading a PyTorch model directly from a URL."
            )
            
        config = {
            "architecture": architecture,
            "kwargs": kwargs or {},
            "backend": backend,
            "url": url,
            "sha256": sha256
        }
    else:
        if variant not in MODEL_REGISTRY:
            available_variants = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available_variants}")
        config = MODEL_REGISTRY[variant]
        backend = config.get("backend", "pytorch")
    
    # 1. Resolve caching paths
    filename = os.path.basename(config["url"])
    cached_file = os.path.join(get_cache_dir(), filename)
    
    if force_download and os.path.exists(cached_file):
        try:
            os.remove(cached_file)
        except OSError:
            pass
        
    # 2. Download weights if missing
    if not os.path.exists(cached_file):
        download_weight_file(config["url"], cached_file, config.get("sha256"))
        
    # 3. Instantiate and load weights based on backend type
    if backend == "pytorch":
        if not HAS_TORCH:
            raise ImportError(
                "Loading a PyTorch backend model requires PyTorch. Please install PyTorch or "
                "install findcrack with standard extras: pip install findcrack[standard]"
            )
        # Safe device selection
        target_device = device
        if "cuda" in str(device) and not torch.cuda.is_available():
            print("Warning: CUDA is not available. Falling back to CPU.")
            target_device = "cpu"
        elif "mps" in str(device) and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("Warning: MPS is not available. Falling back to CPU.")
            target_device = "cpu"
            
        arch_class = config["architecture"]
        if isinstance(arch_class, str):
            if arch_class == "UNet":
                from .unet import UNet
                arch_class = UNet
            elif arch_class == "DeepCrack":
                from .deepcrack import DeepCrack
                arch_class = DeepCrack
            else:
                raise ValueError(f"Unknown PyTorch architecture class name: {arch_class}")
        
        model = arch_class(**config.get("kwargs", {}))
        
        # Load weights
        state_dict = torch.load(cached_file, map_location=target_device)
        model.load_state_dict(state_dict)
        return model.to(target_device).eval()
        
    elif backend == "onnx":
        model = ONNXModelWrapper(cached_file, device=device)
        return model
        
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def list_models() -> list:
    """
    Returns a list of all available pre-trained model variants in the registry.
    """
    return list(MODEL_REGISTRY.keys())


def register_model(
    name: str,
    url: str,
    architecture = None,
    kwargs: dict = None,
    backend: str = "pytorch",
    sha256: str = None
):
    """
    Dynamically registers a custom model variant at runtime.
    This allows users to define custom remote models and use the standard load_model API.
    
    Args:
        name: Name/identifier of the variant.
        url: Remote URL to download the model weights/file from.
        architecture: The PyTorch class/type of the model (not required for ONNX).
        kwargs: Keyword arguments for instantiating the model class.
        backend: Backend framework, either 'pytorch' or 'onnx'.
        sha256: Optional SHA256 checksum to verify the downloaded file.
    """
    if backend not in ("pytorch", "onnx"):
        raise ValueError("backend must be 'pytorch' or 'onnx'")
    if backend == "pytorch" and architecture is None:
        raise ValueError("architecture class must be provided for PyTorch backend")
        
    MODEL_REGISTRY[name] = {
        "architecture": architecture,
        "kwargs": kwargs or {},
        "backend": backend,
        "url": url,
        "sha256": sha256
    }