import os
import numpy as np
from PIL import Image, ImageDraw

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
    
    # Save inside the demo folder to keep things tidy
    output_path = os.path.join(os.path.dirname(__file__), filename)
    image.save(output_path)
    print(f"Mock image saved to {os.path.abspath(output_path)}")
    return output_path
