#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def find_unique_pixel_values(mask_dir):
    """
    Scans all image masks in a directory to find all unique pixel values.

    Args:
        mask_dir (str): Path to the directory containing mask files.

    Returns:
        list: A sorted list of all unique pixel values found across all masks.
    """
    try:
        # Filter for common image file extensions
        file_list = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.tif', '.jpg', '.jpeg'))]
        if not file_list:
            print(f"Error: No image files found in '{mask_dir}'.")
            return None
    except FileNotFoundError:
        print(f"Error: Directory not found at '{mask_dir}'.")
        return None

    all_values = set()

    print(f"Scanning {len(file_list)} masks in '{mask_dir}'...")
    for fname in tqdm(file_list, desc="Processing masks", unit="image"):
        try:
            path = os.path.join(mask_dir, fname)
            with Image.open(path) as im:
                mask_array = np.array(im)
                # Find unique values in the current mask and add them to the master set
                all_values.update(np.unique(mask_array))
        except Exception as e:
            print(f"\nWarning: Could not read or process file '{fname}': {e}")

    return sorted(list(all_values))

def main():
    parser = argparse.ArgumentParser(
        description="Find all unique pixel values in a directory of segmentation masks to identify unlabeled indices."
    )
    parser.add_argument(
        'mask_dir', 
        help="Directory containing the ground truth mask files."
    )
    args = parser.parse_args()

    all_pixel_values = find_unique_pixel_values(args.mask_dir)

    if all_pixel_values is not None:
        print("\n--- Analysis Complete ---")
        print(f"✅ All unique pixel values found across all masks: {all_pixel_values}")

        # Check for values outside the expected 0-39 range
        expected_range = set(range(40))
        found_set = set(all_pixel_values)
        
        # Values in found_set that are not in expected_range
        unlabeled_values = sorted(list(found_set - expected_range))

        if unlabeled_values:
            print("\n🚨 Found values outside the expected 0-39 range!")
            print(f"Potential unlabeled/ignored values: {unlabeled_values}")
            # Suggest the most likely candidate for the ignore_index
            print(f"\nRecommendation: Try running your original script with the argument `--ignore-index {unlabeled_values[0]}`.")
        else:
            print("\n✅ All pixel values are within the expected 0-39 range. No unexpected values found.")

if __name__ == '__main__':
    main()