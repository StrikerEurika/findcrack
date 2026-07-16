import segmentation_models_pytorch as smp

def create_smp_model(arch: str, **kwargs):
    """
    Instantiates an SMP model by architecture string and kwargs.
    Example: arch="Unet", kwargs: encoder_name, encoder_weights, in_channels, classes, ...
    """
    ModelClass = getattr(smp, arch)
    return ModelClass(**kwargs)
