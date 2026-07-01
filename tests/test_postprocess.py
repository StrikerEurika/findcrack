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

if __name__ == "__main__":
    unittest.main()
