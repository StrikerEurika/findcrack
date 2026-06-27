import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .preprocessor import Preprocessor

def main():
    parser = argparse.ArgumentParser(description="Preprocess images for crack detection using LAB-CLAHE.")
    parser.add_argument("input", type=str, help="Path to input image file or directory.")
    parser.add_argument("output", type=str, help="Path to output preprocessed image file or directory.")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit (contrast enhancement factor).")
    parser.add_argument("--tile-grid-size", type=int, nargs=2, default=[8, 8], help="CLAHE tile grid size (width height).")
    parser.add_argument("--no-clahe", action="store_true", help="Disable LAB-CLAHE contrast enhancement.")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    use_clahe = not args.no_clahe
    tile_grid = tuple(args.tile_grid_size)

    preprocessor = Preprocessor(
        use_clahe=use_clahe,
        clip_limit=args.clip_limit,
        tile_grid_size=tile_grid
    )

    def process_file(in_file: Path, out_file: Path):
        try:
            image = np.array(Image.open(in_file).convert("RGB"))
            enhanced = preprocessor.enhance_contrast(image)
            Image.fromarray(enhanced).save(out_file)
            print(f"Processed: {in_file} -> {out_file}")
        except Exception as e:
            print(f"Error processing {in_file}: {e}")

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.is_dir():
            print(f"Error: Output path must be a directory because input path is a directory.")
            exit(1)

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        for file in input_path.iterdir():
            if file.suffix.lower() in valid_exts:
                process_file(file, output_path / file.name)
    else:
        if output_path.is_dir():
            process_file(input_path, output_path / input_path.name)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            process_file(input_path, output_path)

if __name__ == "__main__":
    main()
