# findcrack

`findcrack` is a deep learning crack detection package designed for pixel-level segmentation on high-resolution images. It supports U-Net and DeepCrack architectures, providing an easy-to-use API for inference, model caching, and multi-backend execution (PyTorch & ONNX).

---

## Features

- **Pre-trained Model Zoo**: Fetch pre-trained model weights (e.g., `Seg_UNET_CFD_actual_v1`, `Seg_UNET_CFD_actual_v2`) dynamically on demand.
- **Unified Backend Engine**: Seamlessly executes either PyTorch (`.pth`/`.pt`) or ONNX (`.onnx`) models using the same standard interface.
- **Sliding-Window Inference**: Efficiently process ultra-high-resolution images by dividing them into overlapping patches.
- **Gaussian & Average Blending**: Reconstructs the full image from patches using overlapping Gaussian blending filters to eliminate edge-seam artifacts.
- **Test-Time Augmentation (TTA)**: Performs multi-way augmentations (original, horizontal flip, vertical flip, and rotations) to produce highly robust prediction masks.
- **Validation Metrics**: Compute standard segmentation metrics like IoU, Dice Coefficient, Precision, Recall, and Pixel Accuracy.

---

## Installation

You can install `findcrack` directly from source or via PyPI (once published):

```bash
# Install via pip
pip install findcrack

# Or using uv
uv add findcrack
```

---

## Quickstart

Here is how to load a pre-trained model and run crack detection on a large image:

```python
import cv2
from findcrack import CrackInferencePipeline, load_model

# 1. Load a pre-trained model from the official registry (or use your own URL)
# The weights are downloaded dynamically from GitHub releases on first use.
model = load_model("Seg_UNET_CFD_actual_v1", device="cuda")

# 2. Setup the inference pipeline
pipeline = CrackInferencePipeline(
    model=model,
    device="cuda",
    patch_size=512,
    overlap_ratio=0.2,
    confidence_threhold=0.5,
    use_tta=True  # Enables multi-way Test-Time Augmentation
)

# 3. Perform inference
results = pipeline.predict("path/to/high_res_concrete.jpg")

# The results dictionary contains:
# - results["original_image"]: Original RGB image (numpy array)
# - results["confidence_map"]: Float probability map [0.0 - 1.0]
# - results["binary_mask"]: Binary segmentation mask [0 or 255]

# Save the output mask
cv2.imwrite("detected_cracks.png", results["binary_mask"])
```

---

## API Reference

### Model Loading & Caching

#### `load_model(variant: str, device: str = "cpu", force_download: bool = False, architecture = None, **kwargs)`
Loads a model variant from the local registry or directly from a remote HTTP(S) URL.

- **Parameters**:
  - `variant`: The name of a registered variant (e.g., `"Seg_UNET_CFD_actual_v1"`) or a direct HTTP(S) URL to a weights file.
  - `device`: Target execution device (`"cpu"`, `"cuda"`, or `"mps"`).
  - `force_download`: If `True`, re-downloads weights even if cached locally.
  - `architecture`: PyTorch architecture class (e.g., `UNet`, `DeepCrack`) - required only if loading a raw `.pth`/`.pt` file from a custom URL.

```python
from findcrack import load_model, UNet

# Load custom model weights directly from an external URL
model = load_model(
    variant="https://my-domain.com/custom_unet.pth",
    architecture=UNet,
    device="cuda"
)
```

#### `list_models()`
Returns a list of all pre-trained models available in the built-in registry.

#### `register_model(name: str, url: str, architecture = None, kwargs: dict = None, backend: str = "pytorch")`
Registers a custom variant dynamically at runtime.

---

### Pipeline Configuration

#### `CrackInferencePipeline(model, device: str = "cuda", patch_size: int = 512, overlap_ratio: float = 0.2, confidence_threhold: float = 0.5, use_tta: bool = False)`
Handles sliding window preprocessing, execution, TTA, and patching reconstruction.

---

## Directory Structure

```text
src/
└── findcrack/
    ├── __init__.py          # Main API endpoints (load_model, CrackInferencePipeline, etc.)
    ├── metrics.py           # Segmentation evaluation metrics (IoU, Dice, etc.)
    ├── patching.py          # Sliding window extraction and blend reconstruction
    ├── pipeline.py          # Crack Inference Pipeline wrapper
    ├── preprocess.py        # Color-space CLAHE contrast enhancement & transforms
    ├── tta.py               # Test-Time Augmentation forward pass routines
    └── models/
        ├── __init__.py      # Model module exports
        ├── unet.py          # U-Net model definition
        ├── deepcrack.py     # DeepCrack model definition
        ├── onnx_wrapper.py  # Wrapper for running ONNX models as nn.Modules
        └── zoo.py           # Remote weight registry and cached loaders
```

---

## License

This project is licensed under the MIT License.