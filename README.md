# findcrack

Deep learning crack detection package. Pixel-level segmentation on high-res images. Supports U-Net, DeepCrack, custom ONNX. Fast, robust, modular, minimal deps (ONNX mode).

---

## What
Deep learning package. Pixel-level crack segmentation. Detects cracks in concrete images.

## Why
Standardize crack detection for research, industrial, civil. Fast. Loads models dynamically. PyTorch optional.

## Features
- Pretrained model registry
- Patch inference (sliding window)
- Seamless ONNX/PyTorch backend
- Gaussian/average blend (no seams)
- Test-time augmentation (robust mask)
- Metrics: IoU, Dice, Precision, Recall, Pixel Accuracy
- Custom image preprocess (CLAHE)
- Patch extraction, blending
- Register custom model, CLI + notebook support

---

## Install

### Minimal (ONNX/NumPy)
```sh
pip install findcrack
```
or
```sh
uv add findcrack
```

Only ONNX, NumPy. No PyTorch.

### Standard (PyTorch model/training):
```sh
pip install "findcrack[standard]"
```
or
```sh
uv add findcrack --extra standard
```

### Virtual Env
```sh
python -m venv .venv
source .venv/bin/activate     # Linux/mac
.venv\Scripts\activate.bat    # Windows
```
Or `uv venv`

---

## Usage

### Basic Inference
```python
from findcrack import CrackInferencePipeline, load_model

model = load_model("Det_YOLOv26n-seg_crack-dataset_v1", device="cpu")
pipeline = CrackInferencePipeline(
    model=model, device="cpu", patch_size=512, overlap_ratio=0.2, confidence_threshold=0.5, use_tta=True
)
results = pipeline.predict("path/to/image.jpg")
# results: dict with images, masks, overlay, bounding boxes, contours, etc.
```

#### Visualize
result keys:
- `binary_mask` - segmentation mask
- `overlay` - cracks overlay
- `visualization` - boxes and contours

Save e.g. `cv2.imwrite('out.png', results["overlay"])`

#### Custom Model
```python
from findcrack import load_model, UNet
model = load_model(variant="/path/to/.pth", architecture=UNet, device="cpu")
```

Register custom model:
```python
from findcrack import register_model
register_model(name, url, architecture=UNet, kwargs={...}, backend="pytorch")
```

List models:
```python
from findcrack import list_models
print(list_models())
```

---

## Pipeline
- Sliding window patch infer
- Overlap, blend
- TTA: flips + rotation
- CLAHE (contrast)

---

## Directory Structure
- `demo/` - CLIs, mock, runner
- `src/findcrack/`
  - `models/` - U-Net, DeepCrack, ONNX, registry
  - `inference/` - pipeline, TTA
  - `evaluation/` - metrics
  - `preprocess/` - CLAHE, patch, transforms
  - `postprocess/` - blending

---

## API Entry
From `__init__.py`:
- `CrackInferencePipeline`
- `load_model`, `UNet`
- `calculate_metrics`
- `apply_lab_clahe`, `get_inference_transform`
- `Preprocessor`, `PatchExtractor`, `ImageScaler`
- `PatchBlender`
- `list_models`, `register_model`, `get_model_status_map`

---

## Script Usage
Batch CLAHE preprocess CLI:
```sh
python -m findcrack.preprocess input.jpg output.jpg --clip-limit 2.0 --tile-grid-size 8 8
```

---

## Demo
Run demo pipeline with model, image:
```sh
uv run python demo/demo.py --model Det_YOLOv26n-seg_crack-dataset_v1 --image path/to/image.jpg
```
Look for results in `output/`.

---

## Advanced
Manual patch extraction, blending—see readme + src for more.

---

## Requirements
- Python >=3.11
- Core: albumentations, onnxruntime, opencv-python, pillow, numpy, matplotlib, scikit-image, scipy, pandas, seaborn, scikit-learn, tqdm, ipykernel
- `standard`/`torch` extra: torch, torchaudio, torchvision, segmentation-models-pytorch

---

## Dev Setup
- install `uv`
- `uv pip install -e .[standard]`

---

## Common Errors
- Incorrect path: FileNotFoundError
- No PyTorch (for `.pth`): must install with `[standard]`
- CUDA/MPS unavailable: use `device="cpu"`
- Image must be RGB, high-res
- Out of memory: lower patch_size

---

## License
MIT
