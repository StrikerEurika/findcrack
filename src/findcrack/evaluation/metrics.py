import numpy as np

def calculate_metrics(y_true: np.ndarray, y_prediction: np.ndarray, epsilon: float = 1e-7) -> dict:
    """
    Calculates IoU, Dice, Precision, Recall, and Pixel Accuracy.
    """
    
    y_true = y_true.astype(bool)
    y_prediction = y_prediction.astype(bool)
    
    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum(y_true & y_prediction)
    FP = np.sum(~y_true & y_prediction)
    FN = np.sum(y_true & ~y_prediction)
    TN = np.sum(~y_true & ~y_prediction)
    
    return {
        "IoU": TP / (TP + FP + FN + epsilon),
        "Dice": (2 * TP) / (2 * TP + FP + FN + epsilon),
        "Precision": TP / (TP + FP + epsilon),
        "Recall": TP / (TP + FN + epsilon),
        "Pixel Accuracy": (TP + TN) / (TP + TN + FP + FN + epsilon)
    }
