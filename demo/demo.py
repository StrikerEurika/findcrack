import os
import sys
import argparse

# Ensure the local src/ directory is on the path if not installed package-wide
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from generator import create_mock_crack_image
from runner import load_demo_model, execute_inference, save_outputs

def main():
    parser = argparse.ArgumentParser(description="findcrack - Real World Usage Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to a real image of cracks to run prediction on.")
    parser.add_argument("--model", type=str, default="Seg_UNET_CFD_actual_v1", help="Name of the model variant in the registry or direct path/URL to an ONNX/PyTorch file.")
    args = parser.parse_args()

    # 1. Resolve image path
    is_mock = False
    if args.image is None:
        image_path = create_mock_crack_image()
        is_mock = True
    else:
        if not os.path.exists(args.image):
            print(f"Error: Specified image path '{args.image}' does not exist.")
            sys.exit(1)
        image_path = args.image
        print(f"Using real input image: {os.path.abspath(image_path)}")

    # 2. Check if PyTorch is installed to choose our test path
    try:
        import torch
        HAS_TORCH = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n[Detected] PyTorch is installed in this environment.")
    except ImportError:
        HAS_TORCH = False
        device = "cpu"
        print("\n[Detected] PyTorch is NOT installed. Running in ONNX-only mode.")

    # 3. Load Model
    model = load_demo_model(args.model, HAS_TORCH, device)

    # 4. Run Inference
    results = execute_inference(model, device, image_path)

    # 5. Save output prediction masks
    save_outputs(results, image_path)

    # Clean up mock input image if it was generated
    if is_mock and os.path.exists(image_path):
        os.remove(image_path)

    print("\n====================================================")
    print("           Demo completed successfully!             ")
    print("====================================================")

if __name__ == "__main__":
    main()
