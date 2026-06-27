import os
import sys
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw

# Ensure the local src/ directory is on the path if not installed package-wide
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from findcrack import Preprocessor, CrackInferencePipeline, load_model, register_model

def create_mock_crack_image(filename="mock_concrete.jpg", size=(1000, 1000)):
    """
    Generates a realistic mock concrete image with random lines representing cracks.
    """
    print(f"Generating mock concrete image '{filename}' ({size[0]}x{size[1]})...")
    # Base concrete gray with some noise
    img_array = np.random.normal(128, 20, (size[1], size[0], 3)).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(img_array)
    
    # Draw some "crack" lines
    draw = ImageDraw.Draw(image)
    
    # Crack 1
    draw.line([(100, 100), (300, 250), (450, 600), (800, 900)], fill=(20, 20, 20), width=5)
    # Crack 2
    draw.line([(800, 100), (700, 300), (400, 600), (200, 950)], fill=(10, 10, 10), width=4)
    # Small hair cracks
    draw.line([(300, 250), (200, 400)], fill=(40, 40, 40), width=2)
    
    image.save(filename)
    print(f"Mock image saved to {os.path.abspath(filename)}")
    return filename

def run_realworld_demo(image_path=None):
    print("====================================================")
    print("         findcrack - Real World Usage Demo          ")
    print("====================================================")
    
    # 1. Resolve image path
    is_mock = False
    if image_path is None:
        image_path = create_mock_crack_image()
        is_mock = True
    else:
        if not os.path.exists(image_path):
            print(f"Error: Specified image path '{image_path}' does not exist.")
            sys.exit(1)
        print(f"Using real input image: {os.path.abspath(image_path)}")
    
    # 2. Check if PyTorch is installed to choose our test path
    try:
        import torch
        HAS_TORCH = True
        print("\n[Detected] PyTorch is installed in this environment.")
    except ImportError:
        HAS_TORCH = False
        print("\n[Detected] PyTorch is NOT installed. Running in ONNX-only mode.")
    
    # 3. Load or create the model
    if HAS_TORCH:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n--- Loading PyTorch Model (Device: {device}) ---")
        try:
            # Load real pre-trained model from the zoo
            model = load_model("Seg_UNET_CFD_actual_v1", device=device)
            print("Successfully loaded pre-trained model: 'Seg_UNET_CFD_actual_v1'")
        except Exception as e:
            print(f"Warning: Could not download/load 'Seg_UNET_CFD_actual_v1' ({e}).")
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
            
            model = DummySegmentor()
    else:
        print("\n--- Loading ONNX/NumPy Pipeline Path ---")
        # In ONNX-only mode, we cannot load the PyTorch weights. 
        # We fall back to a NumPy function that simulates model outputs.
        def dummy_onnx_model(x):
            gray = np.mean(x, axis=1, keepdims=True)
            logits = (128.0 - gray) / 10.0 - 1.0
            return logits
            
        model = dummy_onnx_model
        device = "cpu"
        print("Using NumPy callable to simulate ONNX execution.")

    # 4. Instantiate the inference pipeline
    pipeline = CrackInferencePipeline(
        model=model,
        device=device,
        patch_size=256,
        overlap_ratio=0.2,
        confidence_threshold=0.5,
        use_tta=True,  # Test-time augmentation
        overlay_color=(255, 0, 0),  # Transparent red overlay
        box_color=(0, 255, 0),      # Green bounding boxes
        contour_color=(0, 0, 255),  # Blue contour segmentation outlines
        box_thickness=3,
        contour_thickness=2
    )
    
    # 5. Run inference and time it
    print("\nRunning sliding-window inference with TTA...")
    start_time = time.time()
    results = pipeline.predict(image_path)
    elapsed = time.time() - start_time
    
    print(f"Inference completed in {elapsed:.3f} seconds.")
    
    # 6. Verify result structures
    print("\nVerifying outputs:")
    print(f"- Original Image Shape:   {results['original_image'].shape}")
    print(f"- Confidence Map Shape:   {results['confidence_map'].shape} (min: {results['confidence_map'].min():.3f}, max: {results['confidence_map'].max():.3f})")
    print(f"- Binary Mask Shape:       {results['binary_mask'].shape} (unique values: {np.unique(results['binary_mask'])})")
    
    # Ensure output folder exists in the same directory as demo.py
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the output files
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
    
    # Clean up mock input image if it was generated
    if is_mock and os.path.exists(image_path):
        os.remove(image_path)
        
    print("\n====================================================")
    print("           Demo completed successfully!             ")
    print("====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="findcrack - Real World Usage Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to a real image of cracks to run prediction on.")
    args = parser.parse_args()
    
    run_realworld_demo(args.image)
