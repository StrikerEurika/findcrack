import os
import time
import numpy as np
from PIL import Image

from findcrack import CrackInferencePipeline, load_model

def load_demo_model(model_name: str, has_torch: bool, device: str):
    """
    Attempts to load the model from the registry, falling back to a dummy segmentor if needed.
    """
    print(f"\n--- Loading Model '{model_name}' (Device: {device}) ---")
    try:
        model = load_model(model_name, device=device)
        print(f"Successfully loaded model: '{model_name}'")
        return model
    except Exception as e:
        print(f"Warning: Could not download/load '{model_name}' ({e}).")
        
        if has_torch:
            print("Falling back to a lightweight local PyTorch segmentor for testing.")
            import torch.nn as nn
            class DummySegmentor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
                    nn.init.constant_(self.conv.weight, -0.05)
                    nn.init.constant_(self.conv.bias, 1.0)
                def forward(self, x):
                    return self.conv(x)
            return DummySegmentor()
        else:
            print("Falling back to a lightweight NumPy segmentor for testing.")
            def dummy_onnx_model(x):
                gray = np.mean(x, axis=1, keepdims=True)
                logits = (128.0 - gray) / 10.0 - 1.0
                return logits
            return dummy_onnx_model

def execute_inference(model, device: str, image_path: str) -> dict:
    """
    Instantiates the inference pipeline and runs it on the given image.
    """
    pipeline = CrackInferencePipeline(
        model=model,
        device=device,
        patch_size=256,
        overlap_ratio=0.2,
        confidence_threshold=0.5,
        use_tta=True,
        overlay_color=(255, 0, 0),
        box_color=(0, 255, 0),
        contour_color=(0, 0, 255),
        box_thickness=3,
        contour_thickness=2
    )
    
    print("\nRunning sliding-window inference with TTA...")
    start_time = time.time()
    results = pipeline.predict(image_path)
    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.3f} seconds.")
    return results

def save_outputs(results: dict, image_path: str):
    """
    Saves the resulting mask, overlay, and box & contour visualization images.
    """
    print("\nVerifying outputs:")
    print(f"- Original Image Shape:   {results['original_image'].shape}")
    print(f"- Confidence Map Shape:   {results['confidence_map'].shape} (min: {results['confidence_map'].min():.3f}, max: {results['confidence_map'].max():.3f})")
    print(f"- Binary Mask Shape:       {results['binary_mask'].shape} (unique values: {np.unique(results['binary_mask'])})")
    
    # Ensure output folder exists at the root level of the project
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Binary Mask
    output_mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    Image.fromarray(results["binary_mask"]).save(output_mask_path)
    print(f"\nSaved binary prediction mask to:        {os.path.abspath(output_mask_path)}")
    
    # 2. Transparent Overlay
    output_overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    Image.fromarray(results["overlay"]).save(output_overlay_path)
    print(f"Saved transparent overlay to:            {os.path.abspath(output_overlay_path)}")
    
    # 3. Full Bounding Box & Contour Visualization
    output_viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    Image.fromarray(results["visualization"]).save(output_viz_path)
    print(f"Saved box & contour visualization to:    {os.path.abspath(output_viz_path)}")
    
    print(f"\nDetected {len(results['bounding_boxes'])} crack components.")
