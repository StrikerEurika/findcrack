import os
import torch
import hashlib
from torch.hub import get_dir, download_url_to_file

from .unet import UNet
from .deepcrack import DeepCrack
from .onnx_wrapper import ONNXModelWrapper

# Model registry detailing architectures, default arguments, remote URLs, hashes, and backend type.
MODEL_REGISTRY = {
    "Seg_UNET_CFD_actual_v1": {
        "metadata": {
            "loss_functions": ["TverskyLoss"],
            "input_shape": [3, 512, 512],
        },
        "architecture": UNet,
        "kwargs": {
            "n_channels": 3,
            "n_classes": 1,
            "bilinear": False
        },
        "backend": "pytorch",
        "url": "https://github.com/StrikerEurika/findcrack/releases/download/v0.1.0/Seg_UNET_CFD_actual_v1_best.pth",
        "sha256": None  # Users can supply checksums to verify integrity
    },
    "Seg_UNET_CFD_actual_v2": {
        "metadata": {
            "loss_functions": ["BCEWithLogitsLoss", "DiceLoss"],
            "input_shape": [3, 512, 512],
        },
        "architecture": UNet,
        "kwargs": {
            "n_channels": 3,
            "n_classes": 1,
            "bilinear": False
        },
        "backend": "pytorch",
        "url": "https://github.com/StrikerEurika/findcrack/releases/download/v0.1.0/Seg_UNET_CFD_actual_v2_best.pth",
        "sha256": None  # Users can supply checksums to verify integrity
    },
}

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
    hub_directory = get_dir()
    model_directory = os.path.join(hub_directory, 'checkpoints', 'findcrack')
    os.makedirs(model_directory, exist_ok=True)
    return model_directory

def download_weight_file(url: str, cached_file: str, expected_hash: str = None):
    """Downloads remote weight file and verifies its checksum."""
    print(f"Downloading weights from {url}... (This is a one-time download)")
    try:
        download_url_to_file(url, cached_file, progress=True)
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
) -> torch.nn.Module:
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
        arch_class = config["architecture"]
        model = arch_class(**config.get("kwargs", {}))
        
        # Load weights
        state_dict = torch.load(cached_file, map_location=device)
        model.load_state_dict(state_dict)
        return model.to(device).eval()
        
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