import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms
from pdb import set_trace as stx

# --- Configuration ---
PE_CONFIG = "PE-Core-L14-336"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Patch Generation Settings ---
# Define multiple grid sizes (rows, cols) to create patches from.
# A 3x3 grid will generate 9 patches. A 2x2 grid generates 4.
GRID_SIZES = [(3, 3)]  # Generates a 3x3 grid of patches. Add more like [(2, 2), (4, 4)] for more patches.

# Define different scales to apply before creating grid patches.
# 1.0 is the full image. 0.875 is a slight zoom-in.
SCALES = [1.0, 0.875, 0.75] # Generates patches at 3 different scales.

# Overlap between patches as a fraction of patch size. 0.25 means 25% overlap.
# Set to 0 for no overlap.
OVERLAP = 0.25

# --- Output Directory ---
# Patches are always saved individually for feature matching.
EMBEDDINGS_DIR = f"JUNE_embed/FINAL_WED_NEW_PLATES_PE"

def load_pe_model(config: str = PE_CONFIG):
    """
    Loads the PE‐Core model and its preprocess transform.
    Returns: (model, preprocess_fn)
    """
    if config == 'PE-Spatial-G14-448':
        model = pe.VisionTransformer.from_config(config, pretrained=True).to(DEVICE).eval()
    else:
        model = pe.CLIP.from_config(config, pretrained=True).to(DEVICE).eval()
    
    preprocess = pe_transforms.get_image_transform(model.image_size)
    return model, preprocess

def preprocess_image(preprocess_fn, image: Image.Image) -> torch.Tensor:
    """Preprocesses a PIL Image to a tensor."""
    return preprocess_fn(image).unsqueeze(0).to(DEVICE)

def encode_image(model, img_tensor: torch.Tensor) -> np.ndarray:
    """Runs the vision encoder and returns a normalized 1×D numpy vector."""
    with torch.no_grad(), torch.autocast("cuda"):
        try:
            feat = model.encode_image(img_tensor)
        except AttributeError:
            try:
                feat = model.forward_features(img_tensor)
            except Exception as e:
                print(f"Error in model encoding: {e}")
                return None

    if isinstance(feat, torch.Tensor):
        arr = feat.cpu().numpy().reshape(-1)
        # Normalization is applied per patch before saving
        return arr
    else:
        print("Model output was not a tensor.")
        return None

def save_embedding(emb: np.ndarray, save_path: str):
    """Saves a 1-D embedding to .npy, creating dirs as needed and normalizing."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Normalize the embedding vector
    emb /= (np.linalg.norm(emb) + 1e-8)
    np.save(save_path, emb)

def generate_multiscale_grid_patches(
    image: Image.Image,
    scales: list = SCALES,
    grid_sizes: list = GRID_SIZES,
    overlap: float = OVERLAP
):
    """
    Generates a comprehensive set of potentially overlapping patches at multiple scales.

    Args:
        image (Image.Image): The input PIL Image.
        scales (list): A list of floats, where each float is a scaling factor for the image.
        grid_sizes (list): A list of tuples (rows, cols) defining the grid for each scale.
        overlap (float): The fraction of overlap between adjacent patches.

    Returns:
        A list of tuples, where each tuple contains (patch_image, patch_name_suffix).
    """
    patches = []
    original_width, original_height = image.size

    for scale in scales:
        scaled_width = int(original_width * scale)
        scaled_height = int(original_height * scale)

        # Center crop the image to the new scale
        x_offset = (original_width - scaled_width) // 2
        y_offset = (original_height - scaled_height) // 2
        scaled_image = image.crop((x_offset, y_offset, x_offset + scaled_width, y_offset + scaled_height))

        for grid_rows, grid_cols in grid_sizes:
            if grid_rows == 0 or grid_cols == 0:
                continue

            patch_width = scaled_width // grid_cols
            patch_height = scaled_height // grid_rows
            
            step_x = int(patch_width * (1 - overlap))
            step_y = int(patch_height * (1 - overlap))

            for r in range(grid_rows):
                for c in range(grid_cols):
                    # Calculate top-left corner of the patch
                    # For a grid, we want to sample from grid cells, not just step
                    start_x = c * patch_width
                    start_y = r * patch_height
                    
                    # Adjust for overlap for a sliding window approach
                    if c > 0: start_x -= int(c * patch_width * overlap)
                    if r > 0: start_y -= int(r * patch_height * overlap)

                    # Ensure the crop box is within the scaled image bounds
                    end_x = min(start_x + patch_width, scaled_width)
                    end_y = min(start_y + patch_height, scaled_height)
                    
                    start_x = max(0, start_x)
                    start_y = max(0, start_y)
                    
                    if end_x > start_x and end_y > start_y:
                        bbox = (start_x, start_y, end_x, end_y)
                        patch = scaled_image.crop(bbox)
                        
                        # Create a descriptive name for the patch
                        scale_percent = int(scale * 100)
                        patch_name = f"scale_{scale_percent}_grid_{r}x{c}"
                        patches.append((patch, patch_name))
    
    # Add a simple center crop as a baseline patch
    center_crop_size = min(original_width, original_height)
    center_x, center_y = original_width // 2, original_height // 2
    half_crop = center_crop_size // 2
    center_bbox = (center_x - half_crop, center_y - half_crop, center_x + half_crop, center_y + half_crop)
    patches.append((image.crop(center_bbox), "center_crop"))
    
    return patches


def precompute_embeddings(
    prototype_dir: str,
    embeddings_root: str = EMBEDDINGS_DIR,
):
    model, preprocess = load_pe_model()

    date_folders = [
        d for d in os.listdir(prototype_dir)
        if os.path.isdir(os.path.join(prototype_dir, d)) and d.isdigit()
    ]

    for date_str in tqdm(sorted(date_folders), desc="Processing Dates"):
        date_folder_path = os.path.join(prototype_dir, date_str)
        date_out_dir = os.path.join(embeddings_root, date_str)
        os.makedirs(date_out_dir, exist_ok=True)
        print(f"[{date_str}] processing")

        class_folders = [
            d for d in os.listdir(date_folder_path)
            if os.path.isdir(os.path.join(date_folder_path, d))
        ]
        
        for cls in tqdm(sorted(class_folders), desc="Processing Classes", leave=False):
            class_folder = os.path.join(date_folder_path, cls)
            class_out_dir = os.path.join(date_out_dir, cls)
            os.makedirs(class_out_dir, exist_ok=True)

            image_files = [
                os.path.join(class_folder, fn) for fn in os.listdir(class_folder)
                if fn.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            
            processed_image_count = 0

            for img_path in tqdm(sorted(image_files), desc=f"images→{cls}", leave=False):
                try:
                    img = Image.open(img_path).convert("RGB")
                    image_basename = os.path.splitext(os.path.basename(img_path))[0]

                    # Generate a large number of patches using the new function
                    patches_with_names = generate_multiscale_grid_patches(img)

                    if not patches_with_names:
                        tqdm.write(f"    ⚠️ No patches could be generated for {os.path.basename(img_path)}.")
                        continue

                    num_patches_saved = 0
                    for crop, patch_name_suffix in patches_with_names:
                        inp = preprocess_image(preprocess, crop)
                        patch_embedding = encode_image(model, inp)

                        if patch_embedding is not None:
                            save_path = os.path.join(class_out_dir, f"{image_basename}_{patch_name_suffix}.npy")
                            save_embedding(patch_embedding, save_path)
                            num_patches_saved += 1
                        else:
                            tqdm.write(f"    ⚠️ Skipping patch '{patch_name_suffix}' for {os.path.basename(img_path)} due to encoding error.")
                    
                    if num_patches_saved > 0:
                        processed_image_count += 1

                except Exception as e:
                    tqdm.write(f"    ⚠️ Error processing {os.path.basename(img_path)}: {e}")

            if processed_image_count == 0:
                tqdm.write(f"    ⚠️ No patches were saved for any image in class {cls} in {date_str}.")


if __name__ == "__main__":
    # IMPORTANT: Set the path to your dataset's prototype directory
    # PROTO_DIR = "/home/nutrition/code/yash/food-classify/seg_full_segFULL_new_patches" 
    PROTO_DIR = os.getenv("PROTO_DIR", "./prototypes") 
    print(f"📂 Using prototype directory: {PROTO_DIR}")
    precompute_embeddings(PROTO_DIR)
    print("\n✅ All done. Patch embeddings are saved in:", EMBEDDINGS_DIR)