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

### Virtual Environment Setup & Activation

If you are developing or running scripts within a local virtual environment:

- **Creating the virtual environment**:
  ```bash
  # Using standard Python
  python -m venv .venv
  
  # Or using uv (recommended)
  uv venv
  ```

- **Activating the virtual environment**:
  - **Linux/macOS (Bash/Zsh)**:
    ```bash
    source .venv/bin/activate
    ```
  - **Windows (Command Prompt)**:
    ```cmd
    .venv\Scripts\activate.bat
    ```
  - **Windows (PowerShell)**:
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
  - **Windows (Git Bash / Bash)**:
    ```bash
    source .venv/Scripts/activate
    ```


---

## Quickstart

Here is how to load a pre-trained model and run crack detection on a large image:

```python
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from findcrack import CrackInferencePipeline, load_model

# 1. Load a pre-trained model from the official registry (or use your own URL)
# The weights are downloaded dynamically from GitHub releases on first use.
model = load_model("Det_YOLOv26n-seg_crack-dataset_v1", device="cpu")

# 2. Setup the inference pipeline
pipeline = CrackInferencePipeline(
    model=model,
    device="cpu",
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
# - results["overlay"]: Original image with a colored transparent overlay on the cracks
# - results["bounding_boxes"]: List of [xmin, ymin, xmax, ymax] coordinates for detected crack components
# - results["contours"]: List of segmentation contours for detected cracks
# - results["visualization"]: Original image with bounding boxes drawn and contours outlined

# Save the output mask, overlay, and visual bounding boxes
# cv2.imwrite("detected_cracks.png", results["binary_mask"])
# cv2.imwrite("detected_cracks_overlay.png", results["overlay"])
# cv2.imwrite("detected_cracks_visualization.png", results["visualization"])

# Create a figure with 3 subplots
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, wspace=0.3, hspace=0.3)

# 1. Binary Mask
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(results["binary_mask"], cmap='gray')
ax1.set_title('Binary Mask')
ax1.axis('off')

# 2. Overlay
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(results["overlay"])
ax2.set_title('Overlay (Cracks)')
ax2.axis('off')

# 3. Visualization with Bounding Boxes and Contours
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(results["visualization"])
ax3.set_title('Visualization (Boxes + Contours)')
ax3.axis('off')

# Use tight_layout only once at the end
plt.tight_layout()

# Optional: Save the plot as well
plt.savefig('predicted_images_plot.png', dpi=150, bbox_inches='tight')

# Display the plot
plt.show()

# Close the figure to free memory
plt.close()
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

#### `CrackInferencePipeline(model, device: str = "cuda", patch_size: int = 512, overlap_ratio: float = 0.2, confidence_threshold: float = 0.5, use_tta: bool = False, preprocessor = None, use_clahe: bool = True, clahe_clip_limit: float = 2.0, overlay_alpha: float = 0.4, overlay_color: tuple = (255, 0, 0), box_color: tuple = (0, 255, 0), box_thickness: int = 2, contour_color: tuple = (0, 0, 255), contour_thickness: int = 2)`
Handles sliding window preprocessing, execution, TTA, and patching reconstruction. Can be configured with a custom `Preprocessor` or custom CLAHE parameters.

- **Parameters**:
  - `model`: Loaded PyTorch model or ONNX wrapper model.
  - `device`: Device to execute on (`"cpu"`, `"cuda"`, etc.).
  - `patch_size`: Size of the sliding window patch (e.g. `512` or `256`). Inputs will automatically scale to match the model's expected shape if mismatching.
  - `overlap_ratio`: Overlap percentage between consecutive sliding windows (e.g. `0.2`).
  - `confidence_threshold`: Probability threshold to label a pixel as a crack.
  - `use_tta`: Toggle Test-Time Augmentation (flips and rotations).
  - `use_clahe`: Apply LAB-CLAHE contrast enhancement globally.
  - `overlay_alpha`: Transparency level for the output overlay mask.
  - `overlay_color`: RGB tuple color for the overlay mask (default red: `(255, 0, 0)`).
  - `box_color`: RGB tuple color for the bounding boxes (default green: `(0, 255, 0)`).
  - `contour_color`: RGB tuple color for the segmentation contours (default blue: `(0, 0, 255)`).

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
model = load_model("Det_YOLOv26n-seg_crack-dataset_v1", device=device)
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

## Demos and Real-World Validation

`findcrack` includes a modular demo suite to quickly validate end-to-end inference using mock data or real images.

### Folder Layout
```text
demo/
├── generator.py     # Utilities for generating mock concrete test images
├── runner.py        # Handles model resolution (and fallback), pipeline runs, and output saving
└── demo.py          # Main CLI orchestration entrypoint
```

### How to Run:
```bash
# 1. Run the default model variant on a generated mock image
uv run python demo/demo.py

# 2. Specify a model variant from the registry (e.g. your YOLOv8/v11 segmentation model)
uv run python demo/demo.py --model Det_YOLOv26n-seg_crack-dataset_v1

# 3. Process a real image file using a specific model
uv run python demo/demo.py --model Det_YOLOv26n-seg_crack-dataset_v1 --image path/to/cracks.jpg

# example running
uv run python demo/demo.py --model Det_YOLOv26n-seg_crack-dataset_v1 --image "./demo/images/CFD_001.jpg"
```

All predictions are saved inside the root-level `output/` directory:
- `output/<image_name>_mask.png`: Binary prediction mask.
- `output/<image_name>_overlay.png`: Transparent overlay highlighting cracks on the original image.
- `output/<image_name>_visualization.png`: Original image with bounding boxes drawn and contour boundaries outlined.

---

## Directory Structure

```text
demo/                    # Modular demo scripts and mock generators
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