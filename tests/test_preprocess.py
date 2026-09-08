import unittest
import numpy as np

from findcrack.preprocess import apply_lab_clahe, get_inference_transform, Preprocessor
from findcrack.inference import CrackInferencePipeline

class TestPreprocess(unittest.TestCase):
    def test_apply_lab_clahe(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        enhanced = apply_lab_clahe(img)
        self.assertEqual(enhanced.shape, img.shape)
        self.assertEqual(enhanced.dtype, img.dtype)

        enhanced_custom = apply_lab_clahe(img, clip_limit=3.0, tile_grid_size=(4, 4))
        self.assertEqual(enhanced_custom.shape, img.shape)
        self.assertEqual(enhanced_custom.dtype, img.dtype)

        gray_2d = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        enhanced_gray_2d = apply_lab_clahe(gray_2d)
        self.assertEqual(enhanced_gray_2d.shape, gray_2d.shape)
        self.assertEqual(enhanced_gray_2d.dtype, gray_2d.dtype)
        
        gray_3d = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        enhanced_gray_3d = apply_lab_clahe(gray_3d)
        self.assertEqual(enhanced_gray_3d.shape, gray_3d.shape)
        self.assertEqual(enhanced_gray_3d.dtype, gray_3d.dtype)
        
        rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        enhanced_rgba = apply_lab_clahe(rgba)
        self.assertEqual(enhanced_rgba.shape, rgba.shape)
        self.assertEqual(enhanced_rgba.dtype, rgba.dtype)
        np.testing.assert_array_equal(rgba[:, :, 3], enhanced_rgba[:, :, 3])

    def test_get_inference_transform(self):
        transform = get_inference_transform()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        transformed = transform(image=img) if hasattr(transform, 'image') else transform(img)
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.dtype, np.float32)
        self.assertEqual(transformed.shape, (3, 100, 100))

        custom_transform = get_inference_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transformed_custom = custom_transform(img)
        self.assertEqual(transformed_custom.shape, (3, 100, 100))

    def test_preprocessor_class(self):
        preprocessor = Preprocessor()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        enhanced = preprocessor.enhance_contrast(img)
        self.assertEqual(enhanced.shape, img.shape)
        
        array = preprocessor.transform_patch(img)
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape, (3, 100, 100))
        
        enhanced_call, array_call = preprocessor(img)
        self.assertEqual(enhanced_call.shape, img.shape)
        self.assertIsInstance(array_call, np.ndarray)
        self.assertEqual(array_call.shape, (3, 100, 100))
        
        preprocessor_no_clahe = Preprocessor(use_clahe=False)
        enhanced_no_clahe = preprocessor_no_clahe.enhance_contrast(img)
        np.testing.assert_array_equal(enhanced_no_clahe, img)

    def test_pipeline_integration_numpy(self):
        def dummy_model(x):
            return np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            
        pipeline = CrackInferencePipeline(dummy_model, device="cpu", patch_size=50, use_clahe=True)
        self.assertIsInstance(pipeline.preprocessor, Preprocessor)
        self.assertTrue(pipeline.preprocessor.use_clahe)

        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        import tempfile
        import os
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img_path = os.path.join(tmpdir, "test.jpg")
            Image.fromarray(img).save(tmp_img_path)
            
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
        
        extractor = PatchExtractor(patch_size=50, overlap_ratio=0.0)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        patches = list(extractor.extract(img))
        self.assertEqual(len(patches), 4)

        img_small = np.zeros((30, 30, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            list(extractor.extract(img_small))

    def test_pipeline_direct_prediction_for_small_images(self):
        def dummy_model(x):
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
        def dummy_model(x):
            return np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            
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
