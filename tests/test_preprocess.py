import unittest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from findcrack.preprocess import apply_lab_clahe, get_inference_transform, Preprocessor
from findcrack.inference import CrackInferencePipeline

class TestPreprocess(unittest.TestCase):
    def test_apply_lab_clahe(self):
        # Create a dummy RGB image (100x100x3)
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Test default
        enhanced = apply_lab_clahe(img)
        self.assertEqual(enhanced.shape, img.shape)
        self.assertEqual(enhanced.dtype, img.dtype)

        # Test custom params
        enhanced_custom = apply_lab_clahe(img, clip_limit=3.0, tile_grid_size=(4, 4))
        self.assertEqual(enhanced_custom.shape, img.shape)
        self.assertEqual(enhanced_custom.dtype, img.dtype)

        # Test Grayscale 2D
        gray_2d = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        enhanced_gray_2d = apply_lab_clahe(gray_2d)
        self.assertEqual(enhanced_gray_2d.shape, gray_2d.shape)
        self.assertEqual(enhanced_gray_2d.dtype, gray_2d.dtype)
        
        # Test Grayscale 3D (1 channel)
        gray_3d = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        enhanced_gray_3d = apply_lab_clahe(gray_3d)
        self.assertEqual(enhanced_gray_3d.shape, gray_3d.shape)
        self.assertEqual(enhanced_gray_3d.dtype, gray_3d.dtype)
        
        # Test RGBA
        rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        enhanced_rgba = apply_lab_clahe(rgba)
        self.assertEqual(enhanced_rgba.shape, rgba.shape)
        self.assertEqual(enhanced_rgba.dtype, rgba.dtype)
        # Verify alpha channel is preserved
        np.testing.assert_array_equal(rgba[:, :, 3], enhanced_rgba[:, :, 3])

    @unittest.skipIf(not HAS_TORCH, "PyTorch not available")
    def test_get_inference_transform(self):
        transform = get_inference_transform()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        transformed = transform(image=img)
        self.assertIn("image", transformed)
        self.assertIsInstance(transformed["image"], torch.Tensor)
        # Normalization output should be float32 tensor
        self.assertEqual(transformed["image"].dtype, torch.float32)
        # Shape should be (C, H, W)
        self.assertEqual(transformed["image"].shape, (3, 100, 100))

        # Test custom mean/std
        custom_transform = get_inference_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transformed_custom = custom_transform(image=img)
        self.assertEqual(transformed_custom["image"].shape, (3, 100, 100))

    def test_preprocessor_class(self):
        # Create preprocessor with defaults
        preprocessor = Preprocessor()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Test enhance_contrast
        enhanced = preprocessor.enhance_contrast(img)
        self.assertEqual(enhanced.shape, img.shape)
        
        # Test transform_patch
        tensor = preprocessor.transform_patch(img)
        if HAS_TORCH:
            self.assertIsInstance(tensor, torch.Tensor)
        else:
            self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.shape, (3, 100, 100))
        
        # Test __call__
        enhanced_call, tensor_call = preprocessor(img)
        self.assertEqual(enhanced_call.shape, img.shape)
        if HAS_TORCH:
            self.assertIsInstance(tensor_call, torch.Tensor)
        else:
            self.assertIsInstance(tensor_call, np.ndarray)
        self.assertEqual(tensor_call.shape, (3, 100, 100))
        
        # Test disabled CLAHE
        preprocessor_no_clahe = Preprocessor(use_clahe=False)
        enhanced_no_clahe = preprocessor_no_clahe.enhance_contrast(img)
        np.testing.assert_array_equal(enhanced_no_clahe, img)

    @unittest.skipIf(not HAS_TORCH, "PyTorch not available")
    def test_pipeline_integration(self):
        # Define a mock model for testing pipeline initialization
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
            def forward(self, x):
                return self.conv(x)
                
        model = DummyModel()
        
        # Test default initialization
        pipeline = CrackInferencePipeline(model, device="cpu")
        self.assertIsInstance(pipeline.preprocessor, Preprocessor)
        self.assertTrue(pipeline.preprocessor.use_clahe)
        
        # Test with custom preprocessor
        custom_preprocessor = Preprocessor(use_clahe=False, mean=(0.5, 0.5, 0.5))
        pipeline_custom = CrackInferencePipeline(model, device="cpu", preprocessor=custom_preprocessor)
        self.assertIs(pipeline_custom.preprocessor, custom_preprocessor)
        self.assertFalse(pipeline_custom.preprocessor.use_clahe)
        self.assertEqual(pipeline_custom.preprocessor.mean, (0.5, 0.5, 0.5))

    def test_pipeline_integration_numpy(self):
        # A dummy callable representing the ONNX wrapper
        def dummy_model(x):
            # Input x is (B, C, H, W). Returns (B, 1, H, W)
            return np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            
        # Test default initialization with numpy path
        pipeline = CrackInferencePipeline(dummy_model, device="cpu", patch_size=50)
        self.assertIsInstance(pipeline.preprocessor, Preprocessor)
        self.assertTrue(pipeline.preprocessor.use_clahe)

        # Test predict with numpy path
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # Create temp file
        import tempfile
        import os
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img_path = os.path.join(tmpdir, "test.jpg")
            Image.fromarray(img).save(tmp_img_path)
            
            # Predict
            results = pipeline.predict(tmp_img_path)
            self.assertEqual(results["original_image"].shape, img.shape)
            self.assertEqual(results["confidence_map"].shape, (100, 100))
            self.assertEqual(results["binary_mask"].shape, (100, 100))
            self.assertEqual(results["overlay"].shape, img.shape)
            self.assertEqual(results["visualization"].shape, img.shape)
            self.assertIsInstance(results["bounding_boxes"], list)
            self.assertIsInstance(results["contours"], list)

    def test_patch_extraction(self):
        from findcrack import PatchExtractor
        
        # Test patch extraction
        extractor = PatchExtractor(patch_size=50, overlap_ratio=0.0)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        patches = list(extractor.extract(img))
        self.assertEqual(len(patches), 4)  # 2x2 grid

        # Test patch extraction with image smaller than patch size (should raise ValueError)
        img_small = np.zeros((30, 30, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            list(extractor.extract(img_small))

    def test_pipeline_direct_prediction_for_small_images(self):
        # Image is 40x40, patch size is 64. Direct prediction should kick in instead of patching.
        def dummy_model(x):
            # x is shape (B, C, H, W). Return (B, 1, H, W)
            return np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            
        pipeline = CrackInferencePipeline(dummy_model, device="cpu", patch_size=64)
        
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        
        import tempfile
        import os
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img_path = os.path.join(tmpdir, "test_small.jpg")
            Image.fromarray(img).save(tmp_img_path)
            
            results = pipeline.predict(tmp_img_path)
            self.assertEqual(results["original_image"].shape, (40, 40, 3))
            self.assertEqual(results["confidence_map"].shape, (40, 40))
            self.assertEqual(results["binary_mask"].shape, (40, 40))
            self.assertEqual(results["overlay"].shape, (40, 40, 3))
            self.assertEqual(results["visualization"].shape, (40, 40, 3))

    def test_pipeline_batch_inference(self):
        # Test pipeline with batch_size > 1
        def dummy_model(x):
            return np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            
        # 100x100 image, 50x50 patch -> 4 patches. batch_size = 2.
        pipeline = CrackInferencePipeline(dummy_model, device="cpu", patch_size=50, batch_size=2)
        
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        import tempfile
        import os
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img_path = os.path.join(tmpdir, "test_batch.jpg")
            Image.fromarray(img).save(tmp_img_path)
            
            results = pipeline.predict(tmp_img_path)
            self.assertEqual(results["original_image"].shape, (100, 100, 3))
            self.assertEqual(results["confidence_map"].shape, (100, 100))

if __name__ == "__main__":
    unittest.main()
