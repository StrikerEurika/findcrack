from .inference import CrackInferencePipeline
from .models import load_model, UNet, DeepCrack, list_models, register_model
from .evaluation import calculate_metrics
from .preprocess import apply_lab_clahe, get_inference_transform, Preprocessor, PatchExtractor, PatchBlender

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "CrackInferencePipeline",
    "load_model",
    "UNet",
    "DeepCrack",
    "calculate_metrics",
    "apply_lab_clahe",
    "get_inference_transform",
    "Preprocessor",
    "PatchExtractor",
    "PatchBlender",
    "list_models",
    "register_model",
]

