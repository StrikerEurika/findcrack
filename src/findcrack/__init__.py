from .inference import CrackInferencePipeline
from .models import load_model, UNet, list_models, register_model
from .evaluation import calculate_metrics
from .preprocess import apply_lab_clahe, get_inference_transform, Preprocessor, PatchExtractor
from .postprocess import PatchBlender

import importlib.metadata

try:
    __version__ = importlib.metadata.version("findcrack")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "CrackInferencePipeline",
    "load_model",
    "UNet",

    "calculate_metrics",
    "apply_lab_clahe",
    "get_inference_transform",
    "Preprocessor",
    "PatchExtractor",
    "PatchBlender",
    "list_models",
    "register_model",
]

