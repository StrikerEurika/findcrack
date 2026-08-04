import unittest
import numpy as np
import cv2
from findcrack.preprocess import ImageScaler
from findcrack.inference import CrackInferencePipeline


class TestImageScaler(unittest.TestCase):
    def test_scaler_scale_factor(self):
        scaler = ImageScaler(scale_factor=0.5)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        scaled_img, is_scaled = scaler.downscale(img)
        self.assertTrue(is_scaled)
        self.assertEqual(scaled_img.shape, (50, 100, 3))

        # Upscale probability map
        prob_map = np.random.rand(50, 100).astype(np.float32)
        upscaled_map = scaler.upscale_map(prob_map, (100, 200))
        self.assertEqual(upscaled_map.shape, (100, 200))

        # Upscale binary mask
        mask = np.zeros((50, 100), dtype=np.uint8)
        upscaled_mask = scaler.upscale_mask(mask, (100, 200))
        self.assertEqual(upscaled_mask.shape, (100, 200))

    def test_scaler_max_dim(self):
        scaler = ImageScaler(max_dim=100)
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        scaled_img, is_scaled = scaler.downscale(img)
        self.assertTrue(is_scaled)
        # Max dim (400) reduced to 100 -> ratio 0.25 -> (50, 100, 3)
        self.assertEqual(scaled_img.shape, (50, 100, 3))

    def test_scaler_no_op(self):
        scaler = ImageScaler(scale_factor=1.0)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        scaled_img, is_scaled = scaler.downscale(img)
        self.assertFalse(is_scaled)
        self.assertEqual(scaled_img.shape, (100, 100, 3))

    def test_scaler_invalid_args(self):
        with self.assertRaises(ValueError):
            ImageScaler(scale_factor=-0.5)
        with self.assertRaises(ValueError):
            ImageScaler(max_dim=0)

    def test_pipeline_scaling_integration(self):
        def dummy_model(x):
            return np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

        # Image is 200x200, scale_factor=0.5 -> downscaled to 100x100 -> patch_size=50 -> patches
        pipeline = CrackInferencePipeline(
            dummy_model, device="cpu", patch_size=50, scale_factor=0.5
        )

        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        results = pipeline.predict(img)

        # Output results must match original full image shape
        self.assertEqual(results["original_image"].shape, (200, 200, 3))
        self.assertEqual(results["confidence_map"].shape, (200, 200))
        self.assertEqual(results["binary_mask"].shape, (200, 200))
        self.assertEqual(results["overlay"].shape, (200, 200, 3))
        self.assertEqual(results["visualization"].shape, (200, 200, 3))

    def test_pipeline_max_dim_integration(self):
        def dummy_model(x):
            return np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

        # Image 300x300, max_dim=150 -> downscaled to 150x150
        pipeline = CrackInferencePipeline(
            dummy_model, device="cpu", patch_size=50, max_dim=150
        )

        img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        results = pipeline.predict(img)

        self.assertEqual(results["original_image"].shape, (300, 300, 3))
        self.assertEqual(results["confidence_map"].shape, (300, 300))
        self.assertEqual(results["binary_mask"].shape, (300, 300))


if __name__ == "__main__":
    unittest.main()
