#!/usr/bin/env python3
"""
Analyze a folder of images: count per resolution and total uncompressed float32 RGB size.
Usage: python dataset_size.py <folder_path>
"""

import sys
from pathlib import Path
from collections import Counter
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

def analyze_dataset(folder: Path):
    image_files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print("No images found.")
        return

    resolution_counts = Counter()
    total_pixels = 0
    failed = []

    for path in image_files:
        try:
            with Image.open(path) as img:
                w, h = img.size
                resolution_counts[(w, h)] += 1
                total_pixels += w * h
        except Exception as e:
            failed.append((path, e))

    # float32 RGB = 3 channels * 4 bytes per channel
    bytes_per_pixel = 3 * 4
    total_bytes = total_pixels * bytes_per_pixel

    print(f"\nFound {len(image_files)} images ({len(failed)} failed to open)\n")
    print(f"{'Resolution':<20} {'Count':>6}")
    print("-" * 28)
    for (w, h), count in sorted(resolution_counts.items(), key=lambda x: -x[1]):
        print(f"{f'{w}x{h}':<20} {count:>6}")

    print(f"\nTotal uncompressed size (float32 RGB):")
    print(f"  {total_bytes:>15,} bytes")
    print(f"  {total_bytes / 1024**2:>15.2f} MB")
    print(f"  {total_bytes / 1024**3:>15.2f} GB")

    if failed:
        print(f"\nFailed to open {len(failed)} file(s):")
        for path, err in failed:
            print(f"  {path}: {err}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <folder_path>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.")
        sys.exit(1)

    analyze_dataset(folder)
