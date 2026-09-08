from __future__ import annotations
import os
import hashlib
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .onnx_wrapper import ONNXModelWrapper


def _normalize_device(device: str) -> str:
    d = str(device).lower().strip()
    if d in ("gpu", "cuda"):
        return "cuda"
    if d == "cpu":
        return "cpu"
    if d in ("mps", "apple"):
        return "mps"
    return d


def load_registry() -> dict:
    """
    Loads and normalizes the model registry from config/registry.yml.
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
            arch_cfg = m.get("architecture", {})
            if isinstance(arch_cfg, dict):
                arch = arch_cfg.get("type")
                kwargs = arch_cfg.get("kwargs", {})
            else:
                arch = arch_cfg
                kwargs = m.get("kwargs", {})

            artifacts = m.get("artifacts", {})
            backend = artifacts.get("framework", m.get("backend", "onnx"))
            if backend == "ultralytics":
                backend = "onnx"
            url = artifacts.get("model_url", m.get("url"))
            local_path = artifacts.get("local_path", m.get("local_path"))
            if local_path and not Path(local_path).is_absolute():
                path_obj = Path(local_path)
                parts = path_obj.parts
                if parts and parts[0] == ".":
                    parts = parts[1:]
                if parts and parts[0] == "checkpoints":
                    parts = parts[1:]
                local_path = str(Path(*parts)) if parts else ""

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


MODEL_REGISTRY = load_registry()


def get_checkpoints_dir() -> Path:
    """Resolves the checkpoints directory location."""
    env_dir = os.environ.get("FINDCRACK_CHECKPOINTS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    try_root = Path(__file__).resolve().parents[3] / "checkpoints"
    if try_root.is_dir():
        return try_root
        
    try_pkg = Path(__file__).resolve().parents[2] / "checkpoints"
    if try_pkg.is_dir():
        return try_pkg

    try_cwd = Path.cwd() / "checkpoints"
    if try_cwd.is_dir():
        return try_cwd
        
    return try_root


def resolve_local_checkpoint(variant: str, backend: str = "onnx") -> Optional[Path]:
    """
    Attempts to find a local checkpoint weight file for the given variant.
    """
    checkpoints_dir = get_checkpoints_dir()
    if not checkpoints_dir.is_dir():
        return None

    backend_exts = (".onnx",)
    variant_norm = variant.replace("-", "_").lower()
    variant_parts = [p for p in variant_norm.split("_") if p]

    config = MODEL_REGISTRY.get(variant, {})
    local_rel = config.get("local_path")
    if local_rel:
        declared = checkpoints_dir / local_rel
        if declared.is_file():
            return declared
        fname = Path(local_rel).name
        matches = [p for p in checkpoints_dir.glob(f"**/{fname}") if p.is_file()]
        if matches:
            return matches[0]

    for p in checkpoints_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == variant.lower():
            weights = [w for w in p.iterdir()
                       if w.is_file() and w.suffix.lower() in backend_exts]
            if weights:
                best = [w for w in weights if "best" in w.name.lower()]
                return (best or weights)[0]

    for p in checkpoints_dir.rglob("*"):
        if p.is_file() and p.stem.replace("-", "_").lower() == variant_norm:
            if p.suffix.lower() in backend_exts:
                return p

    for p in checkpoints_dir.rglob("*"):
        if p.is_file() and variant.lower() in p.name.lower() and p.suffix.lower() in backend_exts:
            return p

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
    hub_directory = os.path.expanduser(os.path.join("~", ".cache", "findcrack"))
    model_directory = os.path.join(hub_directory, 'checkpoints')
    os.makedirs(model_directory, exist_ok=True)
    return model_directory


def download_weight_file(url: str, cached_file: str, expected_hash: str = None):
    """Downloads remote weight file and verifies its checksum."""
    print(f"Downloading weights from {url}... (This is a one-time download)")
    try:
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
    backend: str = None,
    sha256: str = None,
    local_checkpoint: bool = False
) -> Any:
    """
    Dynamic ONNX model loader.

    Args:
        variant: Registry name (e.g. 'ModelName'), a remote URL, or a local file path.
        device: Target device for execution ('cpu', 'cuda').
        force_download: If True, re-download weights ignoring any cache.
        backend: Backend framework ('onnx'); inferred from extension if omitted.
        sha256: Optional SHA256 checksum to verify downloaded file.
        local_checkpoint: If True, prefer the local checkpoints folder.
    """
    is_url = variant.startswith("http://") or variant.startswith("https://")

    use_local = (not is_url) and (
        local_checkpoint
        or os.environ.get("FINDCRACK_LOCAL_CHECKPOINT", "0").lower() in ("1", "true", "yes")
    )

    cached_file = None
    if is_url:
        url = variant
        if backend is None:
            backend = "onnx" if url.lower().endswith(".onnx") else "onnx"

        config = {
            "backend": backend,
            "url": url,
            "sha256": sha256
        }
    elif Path(variant).is_file() or Path(variant).is_absolute():
        _lp = Path(variant)
        if backend is None:
            backend = "onnx" if variant.lower().endswith(".onnx") else "onnx"
        config = {
            "backend": backend,
            "url": None,
            "sha256": sha256,
        }
        if not _lp.is_file():
            raise FileNotFoundError(f"Local model file not found: {variant}")
        cached_file = str(_lp.resolve())
        use_local = False
    else:
        if variant not in MODEL_REGISTRY:
            available_variants = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available_variants}")
        config = MODEL_REGISTRY[variant]
        if backend is None:
            backend = config.get("backend", "onnx")

    if backend != "onnx":
        backend = "onnx"

    if use_local:
        local_path = resolve_local_checkpoint(variant, backend)
        if local_path:
            print(f"Using local checkpoint weight file: {local_path}")
            cached_file = str(local_path)
        else:
            print(f"Warning: Local checkpoint for '{variant}' not found in checkpoints folder. Falling back to cache/remote URL.")
            
    if cached_file is None:
        if not config.get("url"):
            raise ValueError(f"No remote URL configured for variant '{variant}', and no local checkpoint was found.")
            
        filename = os.path.basename(config["url"])
        cached_file = os.path.join(get_cache_dir(), filename)
        
        if force_download and os.path.exists(cached_file):
            try:
                os.remove(cached_file)
            except OSError:
                pass
            
        if not os.path.exists(cached_file):
            download_weight_file(config["url"], cached_file, config.get("sha256"))
        
    model = ONNXModelWrapper(cached_file, device=device)
    return model


def get_model_status_map() -> dict:
    """
    Returns a mapping: variant -> {display_name, status, backend, url, local_path, sha256}.
    """
    result = {}
    for key, entry in MODEL_REGISTRY.items():
        backend = entry.get("backend", "onnx")
        local_path = entry.get("local_path")
        url = entry.get("url")
        sha256 = entry.get("sha256")
        display_name = entry.get("friendly_name") or key
        installed = False
        resolved_path = None
        try:
            resolved_path = resolve_local_checkpoint(key, backend=backend)
        except Exception:
            resolved_path = None
        if resolved_path and resolved_path.is_file():
            installed = True
        if installed:
            status = "available"
        elif url:
            status = "downloadable"
        else:
            status = "unavailable"
        result[key] = {
            "display_name": display_name,
            "status": status,
            "backend": backend,
            "url": url,
            "local_path": local_path,
            "sha256": sha256,
        }
    return result


def list_models() -> list:
    """
    Returns a list of all available pre-trained model variants in the registry.
    """
    return list(MODEL_REGISTRY.keys())


def print_model_paths():
    """
    Print expected paths for all models.
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
    backend: str = "onnx",
    sha256: str = None
):
    """
    Dynamically registers a custom ONNX model variant at runtime.
    
    Args:
        name: Name/identifier of the variant.
        url: Remote URL to download the model weights/file from.
        backend: Backend framework ('onnx').
        sha256: Optional SHA256 checksum to verify the downloaded file.
    """
    if backend != "onnx":
        raise ValueError("backend must be 'onnx'")
        
    MODEL_REGISTRY[name] = {
        "backend": backend,
        "url": url,
        "sha256": sha256
    }
