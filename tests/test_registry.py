import os
import tempfile
import unittest
from pathlib import Path
from findcrack import load_model, list_models, CrackInferencePipeline
from findcrack.models.registry import resolve_local_checkpoint, get_checkpoints_dir, print_model_paths


class TestModelZoo(unittest.TestCase):
    def setUp(self):
        from findcrack.models.registry import MODEL_REGISTRY
        if "Seg_YOLO26n-seg-v1_crack-seg" not in MODEL_REGISTRY:
            MODEL_REGISTRY["Seg_YOLO26n-seg-v1_crack-seg"] = {
                "backend": "onnx",
                "url": None,
                "local_path": "ultralytics/yolo/YOLO26n-seg_v6_baseline/weights/best.onnx",
                "sha256": None,
                "raw_config": {}
            }

    def test_list_models(self):
        models = list_models()
        self.assertIsInstance(models, list)
        self.assertTrue(len(models) > 0)
        print_model_paths()

    def test_resolve_local_checkpoint(self):
        checkpoints_dir = get_checkpoints_dir()
        self.assertTrue(checkpoints_dir.is_dir(), f"Checkpoints dir {checkpoints_dir} does not exist.")
        
        path_yolo = resolve_local_checkpoint("Seg_YOLO26n-seg-v1_crack-seg", backend="onnx")
        self.assertIsNotNone(path_yolo, "Failed to resolve local checkpoint for Seg_YOLO26n-seg-v1_crack-seg")
        self.assertTrue(path_yolo.is_file())
        self.assertTrue(path_yolo.suffix.lower() == ".onnx")

    def test_load_model_nonexistent_local_fallback_error(self):
        with self.assertRaises(ValueError):
            load_model("nonexistent_model_variant_abc", local_checkpoint=True)

    def test_load_model_requires_onnx_backend(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                load_model("nonexistent_variant", device="cpu", backend="pytorch")
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
