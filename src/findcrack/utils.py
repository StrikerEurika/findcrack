import segmentation_models_pytorch as smp

def load_smp_model(model_path_or_hf_id: str, device: str = "cuda"):
    """Loads an SMP model from a local path or Hugging Face Hub ID."""
    model = smp.from_pretrained(model_path_or_hf_id)
    model = model.to(device).eval()
    return model