try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    smp = None

def load_smp_model(model_path_or_hf_id: str, device: str = "cuda"):
    """Loads an SMP model from a local path or Hugging Face Hub ID."""
    if not HAS_SMP:
        raise ImportError(
            "segmentation-models-pytorch is required to load SMP models. "
            "Please install PyTorch & SMP with standard extra: 'pip install findcrack[standard]' or 'uv add findcrack --extra standard'."
        )
    model = smp.from_pretrained(model_path_or_hf_id)
    model = model.to(device).eval()
    return model