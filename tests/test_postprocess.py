import unittest
import numpy as np
from findcrack.postprocess import PatchBlender

class TestPostprocess(unittest.TestCase):
    def test_patch_blending(self):
        # Initialize blender
        blender = PatchBlender(shape=(100, 100))
        self.assertEqual(blender.prediction_map.shape, (100, 100))
        self.assertEqual(blender.count_map.shape, (100, 100))
        
        # Add two overlapping patches with value 1.0 and 2.0
        # Patch 1: top-left (50x50)
        patch1 = np.ones((50, 50), dtype=np.float32)
        blender.add(patch1, (0, 0))
        
        # Patch 2: top-left (50x50) overlapping
        patch2 = np.ones((50, 50), dtype=np.float32) * 2.0
        blender.add(patch2, (0, 0))
        
        # Merge patches (should be average: 1.5)
        merged = blender.merge()
        self.assertEqual(merged[0, 0], 1.5)
        self.assertEqual(blender.count_map[0, 0], 2)
        
        # Area with no updates should remain 0
        self.assertEqual(merged[99, 99], 0.0)

    def test_gaussian_patch_blending(self):
        # Initialize blender with gaussian blend mode
        blender = PatchBlender(shape=(100, 100), blend_mode="gaussian")
        self.assertEqual(blender.prediction_map.shape, (100, 100))
        self.assertEqual(blender.count_map.shape, (100, 100))
        
        # Add two overlapping patches with value 1.0 and 2.0
        patch1 = np.ones((50, 50), dtype=np.float32)
        blender.add(patch1, (0, 0))
        
        patch2 = np.ones((50, 50), dtype=np.float32) * 2.0
        blender.add(patch2, (0, 0))
        
        # Merge patches (should be weighted average: 1.5 since the gaussian weights are the same at identical coordinates)
        merged = blender.merge()
        self.assertAlmostEqual(merged[0, 0], 1.5)
        
        # Area with no updates should remain 0
        self.assertEqual(merged[99, 99], 0.0)

        # Test invalid blend mode
        with self.assertRaises(ValueError):
            PatchBlender(shape=(100, 100), blend_mode="invalid")

    def test_sigmoid_np_overflow_prevention(self):
        from findcrack.utils import sigmoid_np
        import warnings
        
        # Test standard values
        np.testing.assert_almost_equal(sigmoid_np(0.0), 0.5)
        np.testing.assert_almost_equal(sigmoid_np(np.array([0.0])), np.array([0.5]))
        
        # Test extremely large negative/positive values (which normally overflow np.exp)
        large_inputs = np.array([-1000.0, 1000.0, -1e5, 1e5])
        
        # Run and check that no RuntimeWarning (overflow) is raised
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            res = sigmoid_np(large_inputs)
            
        # Verify output bounds
        self.assertAlmostEqual(res[0], 0.0)
        self.assertAlmostEqual(res[1], 1.0)
        self.assertAlmostEqual(res[2], 0.0)
        self.assertAlmostEqual(res[3], 1.0)

if __name__ == "__main__":
    unittest.main()
