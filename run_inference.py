import os
import cv2
from PIL import Image
from findcrack import CrackInferencePipeline

def main():
    # Define paths
    model_path = r"D:\codes\projects\interns\intern-year-four\crack\findcrack\checkpoints\ultralytics\yolo\YOLO26n-seg_v6_baseline\weights\best.onnx"
    image_path = r"D:\codes\projects\interns\intern-year-four\crack\findcrack\assets\images\CFD_001.jpg"
    output_dir = r"D:\codes\projects\interns\intern-year-four\crack\findcrack\output"
    
    print(f"Loading model: {model_path}")
    print(f"Loading image: {image_path}")
    
    # Initialize the inference pipeline
    pipeline = CrackInferencePipeline(
        model=model_path,
        device="cuda",          # Uses GPU (will fall back to CPU if GPU is unavailable)
        patch_size=512,         # Sliding window patch dimensions (e.g. 256x256)
        overlap_ratio=0.25,      # Overlap between adjacent patches
        confidence_threshold=0.5
    )
    
    # Run prediction
    results = pipeline.predict(image_path)
    
    # Save the output files
    os.makedirs(output_dir, exist_ok=True)
    
    # Save 1: Binary prediction mask
    mask_path = os.path.join(output_dir, "CFD_001_mask.png")
    cv2.imwrite(mask_path, results["binary_mask"])
    print(f"Saved binary prediction mask to: {mask_path}")
    
    # Save 2: Overlaid cracks on original image (convert RGB to BGR for OpenCV)
    overlay_path = os.path.join(output_dir, "CFD_001_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(results["overlay"], cv2.COLOR_RGB2BGR))
    print(f"Saved transparent overlay to: {overlay_path}")
    
    # Save 3: Full box and contour visualization
    viz_path = os.path.join(output_dir, "CFD_001_visualization.png")
    cv2.imwrite(viz_path, cv2.cvtColor(results["visualization"], cv2.COLOR_RGB2BGR))
    print(f"Saved box & contour visualization to: {viz_path}")
    
    print(f"Detected {len(results['bounding_boxes'])} crack components.")

if __name__ == "__main__":
    main()
