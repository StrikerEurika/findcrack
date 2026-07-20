try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    smp = None

def create_smp_model(arch: str, **kwargs):
    """
    Instantiates an SMP model by architecture string and kwargs.
    Example: arch="Unet", kwargs: encoder_name, encoder_weights, in_channels, classes, ...
    """
    if not HAS_SMP:
        raise ImportError(
            "segmentation-models-pytorch is required to create SMP models. "
            "Please install PyTorch & SMP with standard extra: 'pip install findcrack[standard]' or 'uv add findcrack --extra standard'."
        )
    ModelClass = getattr(smp, arch)
    return ModelClass(**kwargs)
