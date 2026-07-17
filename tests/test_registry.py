import os
import unittest
from pathlib import Path
from findcrack import load_model, list_models, CrackInferencePipeline
from findcrack.models.registry import resolve_local_checkpoint, get_checkpoints_dir, print_model_paths

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestModelZoo(unittest.TestCase):
    def test_list_models(self):
        models = list_models()
        # This just checks model listing
        self.assertIsInstance(models, list)
        self.assertTrue(len(models) > 0)
        # Demo print of all valid model paths for info/debug
        print_model_paths()

    def test_resolve_local_checkpoint(self):
        checkpoints_dir = get_checkpoints_dir()
        self.assertTrue(checkpoints_dir.is_dir(), f"Checkpoints dir {checkpoints_dir} does not exist.")
        
        # Test resolution for Seg_Unet-v1_CrackTree260 (not in registry, but folder exists)
        path = resolve_local_checkpoint("Seg_Unet-v1_CrackTree260", backend="pytorch")
        self.assertIsNotNone(path, "Failed to resolve local checkpoint for Seg_Unet-v1_CrackTree260")
        self.assertTrue(path.is_file())
        self.assertTrue(path.name.endswith(".pth") or path.name.endswith(".pt"))

        # Test resolution for Seg_Unet-v1_CFD
        path_v1 = resolve_local_checkpoint("Seg_Unet-v1_CFD", backend="pytorch")
        self.assertIsNotNone(path_v1, "Failed to resolve local checkpoint for Seg_Unet-v1_CFD")
        self.assertTrue(path_v1.is_file())
        self.assertEqual(path_v1.name, "Seg_Unet-v1_CFD.pt")

        # Test resolution for Seg_YOLO26n-seg-v1_crack-seg
        path_yolo = resolve_local_checkpoint("Seg_YOLO26n-seg-v1_crack-seg", backend="onnx")
        self.assertIsNotNone(path_yolo, "Failed to resolve local checkpoint for Seg_YOLO26n-seg-v1_crack-seg")
        self.assertTrue(path_yolo.is_file())
        self.assertEqual(path_yolo.name, "Seg_YOLO26n-seg-v1_crack-seg.onnx")

    @unittest.skipIf(not HAS_TORCH, "PyTorch not available")
    def test_load_model_local(self):
        # Load local pytorch model
        model = load_model("Seg_Unet-v1_CFD", device="cpu", local_checkpoint=True)
        self.assertIsNotNone(model)
        self.assertTrue(isinstance(model, torch.nn.Module))
        self.assertFalse(model.training)

    @unittest.skipIf(not HAS_TORCH, "PyTorch not available")
    def test_pipeline_from_pretrained_local(self):
        # Create pipeline using local model loading
        pipeline = CrackInferencePipeline.from_pretrained(
            "Seg_Unet-v1_CFD", 
            device="cpu", 
            local_checkpoint=True
        )
        self.assertIsNotNone(pipeline)
        self.assertTrue(isinstance(pipeline.model, torch.nn.Module))

    def test_load_model_nonexistent_local_fallback_error(self):
        # A nonexistent variant should raise ValueError
        with self.assertRaises(ValueError):
            load_model("nonexistent_model_variant_abc", local_checkpoint=True)


if __name__ == "__main__":
    unittest.main()
