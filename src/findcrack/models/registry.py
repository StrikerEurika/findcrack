from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Any, Optional

try:
    import torch
    from torch.hub import get_dir, download_url_to_file
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from .onnx_wrapper import ONNXModelWrapper


def load_registry() -> dict:
    """
    Loads and normalizes the model registry from config/registry.yml.

    The registry is the single source of truth for every model: its
    architecture, instantiation kwargs, backend, remote release URL and the
    local checkpoint path (relative to the checkpoints directory). JSON is no
    longer supported.
    """
    yaml_path = Path(__file__).resolve().parent.parent / "config" / "registry.yml"
    if not yaml_path.exists():
        print(f"Warning: Registry not found at {yaml_path}. Using empty registry.")
        return {}

    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        raw_models = data.get("models", {})
        normalized = {}
        for key, m in raw_models.items():
            # Standardize architecture type and kwargs
            arch_cfg = m.get("architecture", {})
            if isinstance(arch_cfg, dict):
                arch = arch_cfg.get("type")
                kwargs = arch_cfg.get("kwargs", {})
            else:
                arch = arch_cfg
                kwargs = m.get("kwargs", {})

            # Standardize backend, URL, local path and checksum from artifacts
            artifacts = m.get("artifacts", {})
            backend = artifacts.get("framework", m.get("backend", "pytorch"))
            url = artifacts.get("model_url", m.get("url"))
            local_path = artifacts.get("local_path", m.get("local_path"))

            checksum = artifacts.get("checksum", {})
            sha256 = checksum.get("sha256") if isinstance(checksum, dict) else m.get("sha256")

            normalized[key] = {
                "architecture": arch,
                "kwargs": kwargs,
                "backend": backend,
                "url": url,
                "local_path": local_path,
                "sha256": sha256,
                "raw_config": m
            }
        return normalized
    except Exception as e:
        print(f"Warning: Failed to load registry.yml: {e}. Falling back to empty registry.")
        return {}


# Load model registry detailing architectures, default arguments, remote URLs, hashes, and backend type.
MODEL_REGISTRY = load_registry()


def get_checkpoints_dir() -> Path:
    """Resolves the checkpoints directory location."""
    env_dir = os.environ.get("FINDCRACK_CHECKPOINTS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    # 1. Check relative to root of the repo (climbing 4 levels from registry.py)
    try_root = Path(__file__).resolve().parents[4] / "checkpoints"
    if try_root.is_dir():
        return try_root
        
    # 2. Check relative to the package src directory
    try_pkg = Path(__file__).resolve().parents[2] / "checkpoints"
    if try_pkg.is_dir():
        return try_pkg

    # 3. Check current working directory
    try_cwd = Path.cwd() / "checkpoints"
    if try_cwd.is_dir():
        return try_cwd
        
    return try_root


def resolve_local_checkpoint(variant: str, backend: str = "pytorch") -> Optional[Path]:
    """
    Attempts to find a local checkpoint weight file for the given variant.
    Resolution order:
    1. The explicit `local_path` declared for this variant in the registry.
    2. A folder / file whose name matches the variant (case-insensitive).
    3. Any weight file containing the variant name as a substring.
    4. A fuzzy parts-based match on the variant name.
    """
    checkpoints_dir = get_checkpoints_dir()
    if not checkpoints_dir.is_dir():
        return None

    backend_exts = (".onnx",) if backend == "onnx" else (".pth", ".pt")
    variant_norm = variant.replace("-", "_").lower()
    variant_parts = [p for p in variant_norm.split("_") if p]

    # 1. Registry-declared local path (deterministic, preferred)
    config = MODEL_REGISTRY.get(variant, {})
    local_rel = config.get("local_path")
    if local_rel:
        declared = checkpoints_dir / local_rel
        if declared.is_file():
            return declared
        # Allow the declared filename to live anywhere under the checkpoints dir
        fname = Path(local_rel).name
        matches = [p for p in checkpoints_dir.glob(f"**/{fname}") if p.is_file()]
        if matches:
            return matches[0]

    # 2. Folder matching the variant name -> pick a backend-appropriate weight
    for p in checkpoints_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == variant.lower():
            weights = [w for w in p.iterdir()
                       if w.is_file() and w.suffix.lower() in backend_exts]
            if weights:
                best = [w for w in weights if "best" in w.name.lower()]
                return (best or weights)[0]

    # 3. File whose stem matches the variant name (case-insensitive)
    for p in checkpoints_dir.rglob("*"):
        if p.is_file() and p.stem.replace("-", "_").lower() == variant_norm:
            if p.suffix.lower() in backend_exts:
                return p

    # 4. File containing the variant name as a substring
    for p in checkpoints_dir.rglob("*"):
        if p.is_file() and variant.lower() in p.name.lower() and p.suffix.lower() in backend_exts:
            return p

    # 5. Fuzzy parts-based match on the file path
    best_match, best_score = None, 0
    for p in checkpoints_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in backend_exts:
            continue
        score = sum(1 for part in variant_parts if part in str(p).lower())
        if score > best_score:
            best_score, best_match = score, p
    if best_score >= 2:
        return best_match

    return None


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
    sha256: str = None,
    local_checkpoint: bool = False
) -> Any:
    """
    Dynamic model loader. Supports 3 main scenarios:

    1. Auto-download & run (no setup):
        model = load_model("ModelName")
        # Downloads and caches model from registry if missing.

    2. Use local or saved model:
        model = load_model("ModelName", local_checkpoint=True)
        # Looks for model file at path in registry.yml; no download. You can also set env var FINDCRACK_LOCAL_CHECKPOINT=1.

    3. Manual checkpoint (advanced):
        # Download the model yourself, place at path in registry.yml, then call as above with local_checkpoint=True.

    Errors and fallback:
        - If no local file and no download URL: error with message showing required path.
        - If using local_checkpoint, falls back to download/cached if local missing, unless no URL is available.

    Supports registry-based and runtime custom models (see `register_model`).


    Source-selection policy (controlled by `local_checkpoint`):
      * local_checkpoint=True  -> load weights from the local `checkpoints/`
        folder (resolved via the registry's `local_path`, then fuzzy lookup).
        Use this while developing the package with weights stored locally.
      * local_checkpoint=False -> download weights from the model's release URL
        and cache them on disk. This is the default for end users / other
        developers trying the released models.

    The same behavior can be forced via the FINDCRACK_LOCAL_CHECKPOINT
    environment variable (set to 1/true/yes).

    Args:
        variant: Registry name (e.g. 'Seg_UNET_CFD_actual_v1') OR a direct
            remote URL (e.g. 'https://.../model.pth').
        device: Target device for execution ('cpu', 'cuda', 'mps').
        force_download: If True, re-download weights ignoring any cache.
        architecture: Model class (e.g. UNet) - required when loading PyTorch
            weights from a raw URL.
        kwargs: Instantiation kwargs for the architecture (used for URL loads).
        backend: Backend ('pytorch' or 'onnx'); inferred from URL if omitted.
        sha256: Optional SHA256 checksum to verify an downloaded file.
        local_checkpoint: If True, prefer the local checkpoints folder; otherwise
            download from the registry's release URL.
    """
    is_url = variant.startswith("http://") or variant.startswith("https://")

    # Flag (or env) decides whether to look locally first or download.
    use_local = (not is_url) and (
        local_checkpoint
        or os.environ.get("FINDCRACK_LOCAL_CHECKPOINT", "0").lower() in ("1", "true", "yes")
    )

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
    
    cached_file = None

    # Try resolving to a local checkpoint weight file if flagged
    if use_local:
        local_path = resolve_local_checkpoint(variant, backend)
        if local_path:
            print(f"Using local checkpoint weight file: {local_path}")
            cached_file = str(local_path)
        else:
            print(f"Warning: Local checkpoint for '{variant}' not found in checkpoints folder as expected by registry.yml. If you want to use a local checkpoint, download weights and place them here: {get_checkpoints_dir() / (MODEL_REGISTRY.get(variant, {}).get('local_path') or '')}\nFalling back to cache/remote URL (if configured).")
            
    # Fallback to cache/remote download if not found locally or if use_local is False
    if cached_file is None:
        if not config.get("url"):
            raise ValueError(f"No remote URL configured for variant '{variant}', and no local checkpoint was found.\n"
                             f"To use this model, manually download/checkpoint and place at: {get_checkpoints_dir() / (MODEL_REGISTRY.get(variant, {}).get('local_path') or '')}")
            
        filename = os.path.basename(config["url"])
        cached_file = os.path.join(get_cache_dir(), filename)
        
        if force_download and os.path.exists(cached_file):
            try:
                os.remove(cached_file)
            except OSError:
                pass
            
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
        # Generalized SMP factory dispatch
        producer = config["raw_config"].get("producer", "").lower()
        if producer.startswith("segmentation models pytorch") or producer == "smp":
            from .smp import create_smp_model
            arch_string = config["raw_config"].get("architecture", {}).get("arch", arch_class)
            model = create_smp_model(arch_string, **config.get("kwargs", {}))
        elif isinstance(arch_class, str):
            if arch_class == "UNet":
                print("WARNING: Fallback to legacy .unet UNet. Registry configuration likely missing or incorrect. All new Unet usage must go through SMP.")
                from .unet import UNet
                model = UNet(**config.get("kwargs", {}))
            else:
                raise ValueError(f"Unknown PyTorch architecture class name: {arch_class}")
        else:
            model = arch_class(**config.get("kwargs", {}))

        # Load weights (tolerant of both raw state_dict files and wrapped
        # training checkpoints that embed the weights under a known key).
        checkpoint = torch.load(cached_file, map_location=target_device)
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        model.load_state_dict(checkpoint)
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

def print_model_paths():
    """
    Print expected paths for all models (for users who want to manually place checkpoints).
    """
    for name, cfg in MODEL_REGISTRY.items():
        print(f"Model: {name}")
        print(f"  Backend: {cfg.get('backend')}")
        print(f"  Local path: {get_checkpoints_dir() / (cfg.get('local_path') or '')}")
        print(f"  Download URL: {cfg.get('url')}")
        print()



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