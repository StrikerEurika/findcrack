import unittest
import numpy as np
from findcrack.evaluation import calculate_metrics

class TestEvaluation(unittest.TestCase):
    def test_calculate_metrics(self):
        # Create dummy true and predicted masks
        # 4x4 array
        y_true = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        
        y_pred = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        
        # y_true has 4 positive pixels.
        # y_pred has 3 positive pixels.
        # TP = 3 (intersection of positive pixels)
        # FP = 0 (pixels predicted positive that are negative in ground truth)
        # FN = 1 (pixels predicted negative that are positive in ground truth)
        # TN = 12 (both 0)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics["IoU"], 3.0 / (3.0 + 0.0 + 1.0))
        self.assertAlmostEqual(metrics["Dice"], 6.0 / (6.0 + 0.0 + 1.0))
        self.assertAlmostEqual(metrics["Precision"], 3.0 / (3.0 + 0.0))
        self.assertAlmostEqual(metrics["Recall"], 3.0 / (3.0 + 1.0))
        self.assertAlmostEqual(metrics["Pixel Accuracy"], (3.0 + 12.0) / 16.0)

if __name__ == "__main__":
    unittest.main()
