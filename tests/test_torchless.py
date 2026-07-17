import sys
import unittest
import subprocess
import os

class TestTorchless(unittest.TestCase):
    def test_torchless_execution(self):
        script = """
import sys

# Raise ImportError for torch and related libraries
class BlockedImportFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname in ("torch", "torchvision", "torchaudio") or fullname.startswith(("torch.", "torchvision.", "torchaudio.")):
            raise ImportError(f"Blocked import of {fullname}")
        return None

sys.meta_path.insert(0, BlockedImportFinder())

# Verify import raises ImportError
try:
    import torch
    print("FAILED_TO_BLOCK_TORCH")
    sys.exit(1)
except ImportError:
    pass

# Import findcrack components
try:
    from findcrack import Preprocessor, CrackInferencePipeline, load_model, list_models, UNet
    from findcrack.models.onnx_wrapper import ONNXModelWrapper
    from findcrack.inference.tta import tta_forward_np
except Exception as e:
    print(f"FAILED_IMPORT: {e}")
    sys.exit(1)

# Check HAS_TORCH is False
from findcrack.preprocess.preprocessor import HAS_TORCH
if HAS_TORCH:
    print("FAILED_HAS_TORCH_IS_TRUE")
    sys.exit(1)

# Test UNet raises ImportError on instantiation
try:
    UNet()
    print("FAILED_UNET_NO_ERROR")
    sys.exit(1)
except ImportError:
    pass

# Test Preprocessor on NumPy
import numpy as np
preprocessor = Preprocessor(use_clahe=True)
img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
enhanced, transformed = preprocessor(img)

if not isinstance(transformed, np.ndarray):
    print("FAILED_PREPROCESSOR_NOT_NUMPY")
    sys.exit(1)
if transformed.shape != (3, 100, 100):
    print("FAILED_PREPROCESSOR_SHAPE")
    sys.exit(1)

# Test dummy ONNX Model and pipeline with TTA
def dummy_model(x):
    # input is (B, C, H, W). returns (B, 1, H, W)
    return np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

pipeline = CrackInferencePipeline(dummy_model, device="cpu", patch_size=50, use_tta=True)

# Create a temp file and predict
import tempfile
import os
from PIL import Image

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_img_path = os.path.join(tmpdir, "test.jpg")
    Image.fromarray(img).save(tmp_img_path)
    
    results = pipeline.predict(tmp_img_path)
    if results["original_image"].shape != img.shape:
        print("FAILED_PIPELINE_ORIGINAL_IMAGE")
        sys.exit(1)
    if results["confidence_map"].shape != (100, 100):
        print("FAILED_PIPELINE_CONFIDENCE_MAP")
        sys.exit(1)
    if results["binary_mask"].shape != (100, 100):
        print("FAILED_PIPELINE_BINARY_MASK")
        sys.exit(1)

print("SUCCESS")
"""
        env = os.environ.copy()
        # Ensure PYTHONPATH includes our src directory
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
        
        # Log outputs for debugging
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        self.assertEqual(result.returncode, 0, f"Torchless test failed with code {result.returncode}")
        self.assertIn("SUCCESS", result.stdout)

if __name__ == "__main__":
    unittest.main()
