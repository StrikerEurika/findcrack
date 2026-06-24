import torch
import os
from torch.hub import download_url_to_file, get_dir

from .unet import UNet

# model variants
MODEL_ZOO = {
    "unet_cfd_v1": {
        "architecture": UNet,
        "kwargs": {
            "n_channels": 3, 
            "n_classes": 1, 
            "bilinear": False
        },
        "url": "https://github.com/YOUR_USERNAME/findcrack/releases/download/v1.0/unet_cfd_v1.pth"
    }
}

def get_pretrained_model(variant: str, device: str = "cpu"):
    """
    Get a pretrained model by its variant name.
    """

    if variant not in MODEL_ZOO:
        available_variants = list(MODEL_ZOO.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available: {available_variants}")
    
    config = MODEL_ZOO[variant]

    # init the empty architcture
    model = config["architecture"](**config["kwargs"])

    # caching directory
    hub_directory = get_dir()
    model_directory = os.path.join(hub_directory, 'checkpoints', 'findcrack')
    os.makedirs(model_directory, exist_ok=True)
    
    filename = os.path.basename(config["url"])
    cached_file = os.path.join(model_directory, filename)
    
    # Download weights if they don't exist locally
    if not os.path.exists(cached_file):
        print(f"Downloading weights for '{variant}'... (This is a one-time download)")
        download_url_to_file(config["url"], cached_file, progress=True)
        
    # 4. Load weights into the model
    state_dict = torch.load(cached_file, map_location=device)
    model.load_state_dict(state_dict)
    
    return model.to(device).eval()