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

By default, `findcrack` does not require PyTorch or its related packages, keeping the installation small and lightweight for ONNX-only inference.

### Base Installation (ONNX & NumPy only)
To run ONNX models without PyTorch:
```bash
# Install via pip
pip install findcrack

# Or using uv
uv add findcrack
```

### PyTorch Support (Standard Installation)
To enable PyTorch models and training support, install the `standard` extra (which includes PyTorch, torchvision, and torchaudio):
```bash
# Install via pip
pip install "findcrack[standard]"

# Or using uv
uv add findcrack --extra standard
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
    confidence_threshold=0.5,
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

#### `CrackInferencePipeline(model, device: str = "cuda", patch_size: int = 512, overlap_ratio: float = 0.2, confidence_threshold: float = 0.5, use_tta: bool = False, preprocessor = None, use_clahe: bool = True, clahe_clip_limit: float = 2.0)`
Handles sliding window preprocessing, execution, TTA, and patching reconstruction. Can be configured with a custom `Preprocessor` or custom CLAHE parameters.

---

### Image Preprocessing

#### `Preprocessor(use_clahe: bool = True, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), std: Tuple[float, float, float] = (0.229, 0.224, 0.225), additional_transforms: List = None)`
A configuration-driven preprocessing class encapsulating LAB-CLAHE contrast enhancement and Albumentations normalization/tensorization.

```python
from findcrack import Preprocessor

# Custom preprocessor setup
preprocessor = Preprocessor(
    use_clahe=True,
    clip_limit=3.0,
    tile_grid_size=(4, 4)
)

# Apply CLAHE to global image (reduces patch boundary mismatch)
enhanced_image = preprocessor.enhance_contrast(image)

# Normalize and convert a patch to a PyTorch tensor
patch_tensor = preprocessor.transform_patch(patch_rgb)
```

#### CLI Usage
You can run `findcrack.preprocess` as a script to batch apply LAB-CLAHE contrast enhancement to an image or directory:

```bash
# Process a single image
python -m findcrack.preprocess input.jpg output.jpg --clip-limit 2.0 --tile-grid-size 8 8

# Process a directory of images
python -m findcrack.preprocess path/to/input_dir path/to/output_dir
```

---

## Advanced End-to-End Usage

If you need fine-grained control over patching parameters, custom preprocessing transformations, or model thresholding, you can run the pipeline manually. This is particularly useful when handling arbitrary non-square patch sizes or injecting custom Albumentations transforms.

Here is a full example of manual patch extraction, inference, and blending on an arbitrarily-sized image:

```python
import numpy as np
import torch
from PIL import Image
import albumentations as A

from findcrack import load_model
from findcrack.preprocess import Preprocessor, PatchExtractor
from findcrack.postprocess import PatchBlender

# 1. Load pre-trained model and set to evaluation mode
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("Seg_UNET_CFD_actual_v1", device=device)
model.eval()

# 2. Instantiate preprocessor with custom Albumentations transforms
custom_preprocessor = Preprocessor(
    use_clahe=True,
    clip_limit=3.0,
    tile_grid_size=(4, 4),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    additional_transforms=[
        A.GaussianBlur(p=0.5)  # E.g., custom blur filter
    ]
)

# 3. Read input image of any arbitrary dimensions
img_path = "large_image.jpg"
image = np.array(Image.open(img_path).convert("RGB"))
height, width, _ = image.shape

# 4. Initialize patch extractor and blender for arbitrary size
patch_size = (512, 512)  # Can be a Tuple (height, width) or an integer
overlap_ratio = 0.25      # overlap percentage between consecutive steps

extractor = PatchExtractor(patch_size=patch_size, overlap_ratio=overlap_ratio)
blender = PatchBlender(shape=(height, width))

# Preprocess image contrast globally before patching to reduce boundary artifacts
enhanced_image = custom_preprocessor.enhance_contrast(image)

# 5. Extract patches and feed to the model
with torch.no_grad():
    for patch_rgb, coordinates in extractor.extract(enhanced_image):
        # Normalize and transform patch to tensor
        patch_tensor = custom_preprocessor.transform_patch(patch_rgb).to(device)
        
        # Forward pass (adding batch dimension and squeezing logits)
        logits = model(patch_tensor.unsqueeze(0))
        pred_prob = torch.sigmoid(logits).squeeze()
        
        # Add predicted patch probabilities back to the blender
        blender.add(pred_prob.cpu().numpy(), coordinates)

# 6. Merge/blend the overlapping patch maps into the full-size confidence map
confidence_map = blender.merge()

# 7. Convert confidence map to a binary mask using custom thresholding
confidence_threshold = 0.5
binary_mask = (confidence_map > confidence_threshold).astype(np.uint8) * 255
```

---

## Directory Structure

```text
src/
└── findcrack/
    ├── __init__.py          # Main API endpoints (load_model, CrackInferencePipeline, etc.)
    ├── evaluation/          # Evaluation tools and metrics package
    │   ├── __init__.py      # Evaluation module exports
    │   └── metrics.py       # Segmentation evaluation metrics (IoU, Dice, etc.)
    ├── inference/           # Inference pipeline and test-time augmentation (TTA)
    │   ├── __init__.py      # Inference exports
    │   ├── pipeline.py      # Crack Inference Pipeline wrapper
    │   └── tta.py           # Test-Time Augmentation forward pass routines
    ├── postprocess/         # Postprocessing and patch blending package
    │   ├── __init__.py      # Postprocess exports
    │   └── blending.py      # PatchBlender class for merging patches
    ├── preprocess/          # Image preprocessing & patching package
    │   ├── __init__.py      # Preprocess module exports
    │   ├── __main__.py      # CLI for running preprocessing scripts
    │   ├── clahe.py         # LAB-CLAHE contrast enhancement
    │   ├── patching.py      # Sliding window patch extraction
    │   ├── preprocessor.py  # Unified Preprocessor class wrapper
    │   └── transforms.py    # Image normalization and albumentations setup
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