import csv
import json
from pathlib import Path
import numpy as np
import pycocotools.mask as mask_util
import cv2
from PIL import Image
import supervision as sv
import os
from google import genai # Use this as per the new documentation
# Import for types like Content, Part, GenerateContentConfig
from google.genai import types # Use this as per the new documentation
import re


import numpy as np
from scipy.spatial.distance import mahalanobis

class DistanceSimilarity:
    def __init__(self, proto_embs: dict[str, np.ndarray]):
        """
        proto_embs: dict[class_name → prototype embedding vector]
        """
        self.proto_embs = proto_embs
        # Precompute anything for Mahalanobis
        all_feats = np.stack(list(proto_embs.values()), axis=0)
        # covariance of prototypes
        cov = np.cov(all_feats, rowvar=False) + np.eye(all_feats.shape[1]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)

    def euclidean(self, feat: np.ndarray) -> dict[str, float]:
        # smaller distance → higher “sim” via 1/(1+d)
        return {
            cls: 1.0 / (1.0 + np.linalg.norm(feat - proto))
            for cls, proto in self.proto_embs.items()
        }

    def manhattan(self, feat: np.ndarray) -> dict[str, float]:
        return {
            cls: 1.0 / (1.0 + np.abs(feat - proto).sum())
            for cls, proto in self.proto_embs.items()
        }

    def mahalanobis(self, feat: np.ndarray) -> dict[str, float]:
        # uses scipy.spatial.distance.mahalanobis
        return {
            cls: 1.0 / (1.0 + mahalanobis(feat, proto, self.inv_cov))
            for cls, proto in self.proto_embs.items()
        }
        
        
        
def load_colormap(csv_path: str):
    classes, cmap = [], {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            classes.append(row['category'])
            cmap[row['category']] = (int(row['R']), int(row['G']), int(row['B']))
    return classes, cmap


def mask_to_rle(mask: np.ndarray) -> dict: # Add type hint for clarity
    """Converts a boolean numpy mask to COCO RLE format."""
    if mask.ndim == 2: # Ensure it's a 2D mask
        mask_fortran = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(mask_fortran)
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    elif mask.ndim == 3 and mask.shape[2] == 1: # If it's (H, W, 1)
        mask_fortran = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(mask_fortran)[0] # Access the first element if encode returns a list
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    else:
        raise ValueError(f"Mask must be 2D or 3D with last dim 1, got shape {mask.shape}")



def remove_overlapping_masks(
    masks: np.ndarray,
    boxes: np.ndarray,
    names: list,
    containment_threshold: float = 0.9
):
    """
    Removes masks mostly contained inside larger ones (based on containment, not IoU),
    and updates bounding boxes and class names.

    Args:
        masks (np.ndarray): Boolean masks of shape (N, H, W)
        boxes (np.ndarray): Bounding boxes of shape (N, 4) in xyxy format
        names (List[str]): Class names for each mask
        containment_threshold (float): % of mask i inside mask j to remove i

    Returns:
        filtered_masks, filtered_boxes, filtered_names, removed_indices
    """
    N = masks.shape[0]
    keep = np.ones(N, dtype=bool)

    for i in range(N):
        if not keep[i]:
            continue
        for j in range(N):
            if i == j or not keep[j]:
                continue

            inter = np.logical_and(masks[i], masks[j]).sum()
            area_i = masks[i].sum()

            containment = inter / (area_i + 1e-6)

            # if i is mostly contained inside j and j is larger, remove i
            if containment > containment_threshold and area_i < masks[j].sum():
                keep[i] = False
                break

    removed_indices = np.where(~keep)[0].tolist()
    return masks[keep], boxes[keep], [n for k, n in enumerate(names) if keep[k]], removed_indices


def save_raw_masks(masks, out_dir):
    # check if out_dir exists, if not create it
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img.save(out_dir / f'mask_{i}.png')
    


def save_results(*, img_rgb, class_masks, confs, classes, cmap, out_dir, original_image_path = None, generate_calories = False):
    original_img = Image.fromarray(img_rgb) # Keep this for sending to Gemini
    abs_mask = np.zeros_like(img_rgb) 
    for c in classes:
        if c in confs:
            mask = class_masks[c]
            
            color = cmap.get(c)
            if color is None:
                print("No color for class %s", c)
                continue
            abs_mask[mask] = color
            
    abs_mask = Image.fromarray(abs_mask)
    abs_mask.save(out_dir/"abs_mask.png")
    print("Saved absolute mask for %s", out_dir.name)
            

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_rgb).save(out_dir/"input.jpg")
    H,W = img_rgb.shape[:2]
    combined = np.zeros((H,W), bool)
    for c in classes:
        if c in confs:
            combined |= class_masks[c]
    Image.fromarray((combined*255).astype('uint8'))\
        .save(out_dir/"combined_mask.png")
    print("Saved combined mask for %s", out_dir.name)

    ov = img_rgb.copy()
    for c in classes:
        if c in confs:
            m = class_masks[c]; col=cmap.get(c,(255,255,255))
            ov[m] = ((ov[m].astype(float)*0.5)+np.array(col)*0.5).astype('uint8')
    Image.fromarray(ov).save(out_dir/"overlay.png")
    print("Saved classification overlay for %s", out_dir.name)

    overlay_path = out_dir/"overlay.png"
    overlay_bgr = cv2.imread(str(overlay_path))
    y = 30
    for cls, conf in confs.items():
        text = f"{cls}: {conf:.2f}"
        cv2.putText(
            overlay_bgr, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            lineType=cv2.LINE_AA
        )
        y += 40
    labeled_path = out_dir / "overlay_labeled.png"
    cv2.imwrite(str(labeled_path), overlay_bgr)
    print("Saved overlay with class labels to %s", labeled_path)

    (out_dir/"cutouts"/"RGB").mkdir(parents=True, exist_ok=True)
    (out_dir/"cutouts"/"masks").mkdir(parents=True, exist_ok=True)
    for c in classes:
        if c in confs:
            m = class_masks[c]; mp=(m.astype('uint8')*255)
            Image.fromarray(mp).save(out_dir/"cutouts"/"masks"/f"{c}.png")
            cut = img_rgb.copy(); cut[~m]=0
            rgba = np.dstack([cut, mp])
            Image.fromarray(rgba).save(out_dir/"cutouts"/"RGB"/f"{c}.png")
    print("Saved cutouts for %s", out_dir.name)

    # --- MODIFICATION START ---
    dets = []
    for c in classes:
        if c in confs:
            mask_data = class_masks[c]
            rle_mask = mask_to_rle(mask_data) # Ensure single_mask_to_rle handles boolean masks correctly
            dets.append({
                'class': c,
                'confidence': round(float(confs[c]), 3),
                'segmentation_rle': rle_mask  # Add RLE mask
            })
    # --- MODIFICATION END ---

    with open(out_dir/"classes.json",'w') as f:
        json.dump({'detections':dets}, f, indent=2)
    print("Wrote classes.json for %s", out_dir.name)

    classes_json_path = str(out_dir/"classes.json")
    masks_path = sorted((out_dir/"cutouts"/"masks").glob("*.png"))
    # Pass the actual PIL image object
    if generate_calories:
        cal_table = estimate_calories(original_image_path, classes_json_path, masks_path)
        if cal_table:
            with open(classes_json_path, 'r+') as f:
                doc = json.load(f)
                doc['cal_estimation'] = cal_table
                f.seek(0); f.truncate()
                json.dump(doc, f, indent=2)
            print("Appended calorie estimations to classes.json")
            

def compute_ground_truth_label(gt_png_path, classes, class2idx, target_size=(1024,1024)):
    from PIL import Image
    rgb = np.array(Image.open(gt_png_path).convert('RGB'))
    if rgb.shape[:2] != target_size:
        rgb = np.array(Image.fromarray(rgb).resize(target_size, resample=Image.NEAREST))
    h,w,_ = rgb.shape
    lbl = np.zeros((h,w), dtype=np.int32)
    for color, cls in classes.items():
        mask = np.all(rgb == color, axis=-1)
        lbl[mask] = class2idx[cls]
    return lbl



def estimate_calories(image_path: str, classes_json_path: str, mask_paths: list[str]) -> dict:

    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        # Fallback to your specific key if the environment variable isn't set.
        # For production, always prefer environment variables.
        GEMINI_API_KEY = "" # Your provided key
        # Check if it's still the placeholder or your actual key
        if GEMINI_API_KEY == "": # Example placeholder
             print("Using a hardcoded API key. For better security, set the GOOGLE_API_KEY environment variable.")
        # else: # If it's a different key, it might have been intentional
             # print("Using API key from fallback.")


    if not GEMINI_API_KEY: # Final check
        print("API key (GOOGLE_API_KEY or fallback) is not set.")
        return {"error": "API key not configured"}

    try:
        # Client Initialization for Gemini Developer API
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Successfully initialized genai.Client for Gemini Developer API.")
    except Exception as e:
        print(f"Failed to initialize genai.Client: {e}", exc_info=True)
        return {"error": f"Client initialization failed: {str(e)}"}

    uploaded_file_sdk_objects = [] # To store SDK's file objects for cleanup

    try:
        # 1) Upload image, detections JSON, and each cutout mask
        print(f"Uploading main image: {image_path}...")
        main_image_file_obj = client.files.upload(file=image_path)
        uploaded_file_sdk_objects.append(main_image_file_obj)
        print(f"Uploaded main image: Name='{main_image_file_obj.name}', URI='{main_image_file_obj.uri}'")

        detected_classes = None
        try:
            with open(classes_json_path, 'r') as f:
                detected_classes = json.load(f)
            print(f"Successfully loaded classes JSON from: {classes_json_path}")
        except FileNotFoundError:
            print(f"Error: Classes JSON file not found at {classes_json_path}")
            return {"error": f"Classes JSON file not found: {classes_json_path}"}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {classes_json_path}: {e}")
            return {"error": f"Failed to decode classes JSON: {str(e)}"}
        
        
        for i, mp in enumerate(mask_paths):
            print(f"Uploading mask {i+1}/{len(mask_paths)}: {mp}...")
            mask_file_obj = client.files.upload(file=mp)
            uploaded_file_sdk_objects.append(mask_file_obj)
            print(f"Uploaded mask '{mp}': Name='{mask_file_obj.name}', URI='{mask_file_obj.uri}'")

        # 2) Prepare parts for the prompt content
        model_prompt_parts = [
            types.Part.from_uri(file_uri=main_image_file_obj.uri, mime_type=main_image_file_obj.mime_type)
        ]
        for sdk_file_obj in uploaded_file_sdk_objects[1:]: # Skip the main image
            model_prompt_parts.append(types.Part.from_uri(file_uri=sdk_file_obj.uri, mime_type=sdk_file_obj.mime_type))

        prompt_text_content = f"""
Analyze all the files provided.
The first uploaded file is the main image of the food.
Item Details: {json.dumps(detected_classes, indent=2)}
The subsequent files are individual cutout mask images. Each mask corresponds to one of the food items detailed in the JSON file. It is crucial to assume that the order of these mask files matches the order of the food items listed in the provided JSON file.

Based on all these inputs (the main image, the item details, and their respective masks), your task is to:
For each food item identified in the JSON file:
1.  Use its corresponding mask to understand its specific region and volume in the main image.
2.  Provide an estimated "Calorie Count" (e.g., "150 kcal").
3.  Provide an estimated "Weight Count" (e.g., "100 g").

Return the result as a single JSON array of objects. Each object in the array must represent a food item and strictly contain the following fields:
- "Item Name": The name of the food item (this should directly correspond to, or be clearly derived from, the class/name in the input detections JSON).
- "Calorie Count": The estimated calorie count as a string (e.g., "250 kcal", "Not determinable").
- "Weight Count": The estimated weight as a string (e.g., "120 g", "Not determinable").

Important Output Instructions:
- Only output the JSON array.
- Do not include any other text, explanations, introductory sentences, or markdown formatting (like ```json ... ``` or ``` ... ```) around the JSON output.
- The entire response should be a valid JSON array.

Example of expected JSON output format:
[
  {{
    "Item Name": "Apple",
    "Calorie Count": "95 kcal",
    "Weight Count": "180 g"
  }},
  {{
    "Item Name": "Banana",
    "Calorie Count": "105 kcal",
    "Weight Count": "118 g"
  }}
]
        """
        model_prompt_parts.append(types.Part.from_text(text=prompt_text_content))

        request_contents = [types.Content(role="user", parts=model_prompt_parts)]

        # Model Name: From the google-genai SDK documentation examples.
        model_name_to_use = "gemini-2.0-flash" # Or other compatible models like "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"

        generation_config = types.GenerateContentConfig(
            response_mime_type="text/plain"
        )

        print(f"Generating content from model '{model_name_to_use}'...")
        response = client.models.generate_content(
            model=model_name_to_use,
            contents=request_contents,
            config=generation_config
        )

        # --- Extract, clean, and parse JSON response ---
        full_text = ""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and \
               hasattr(candidate.content, 'parts') and candidate.content.parts:
                full_text = "".join(p.text for p in candidate.content.parts if hasattr(p, "text")).strip()
            elif hasattr(candidate, 'text') and candidate.text:
                 full_text = candidate.text.strip()
                 print("Extracted text from response.candidates[0].text")
            else:
                 print("Could not find text in response.candidates[0].content.parts or response.candidates[0].text.")
        
        if not full_text and hasattr(response, 'text'):
            print("Falling back to or using response.text for extracting content.")
            full_text = response.text.strip()
        
        if not full_text:
            print("Model returned no parsable text content (candidates or response.text).")
            error_detail_msg = "Model returned empty text. "
            if hasattr(response, 'prompt_feedback'):
                 error_detail_msg += f"Prompt feedback: {response.prompt_feedback}. "
            if 'candidate' in locals() and candidate: # Check if candidate object was formed
                if hasattr(candidate, 'finish_reason'):
                    fr_val = candidate.finish_reason
                    fr_str = str(fr_val.name) if hasattr(fr_val, 'name') else str(fr_val)
                    error_detail_msg += f"Finish reason: {fr_str}. "
                if hasattr(candidate, 'safety_ratings'):
                    sr_strs = [f"({(sr.category.name if hasattr(sr.category, 'name') else sr.category)}: {(sr.probability.name if hasattr(sr.probability, 'name') else sr.probability)})" for sr in candidate.safety_ratings]
                    error_detail_msg += f"Safety ratings: {sr_strs}."
            print(error_detail_msg)
            return {"error": "Model returned empty text.", "details": error_detail_msg.strip()}

        print(f"Raw response text from model:\n---\n{full_text}\n---")

        cleaned_text = re.sub(r'^```(?:json)?\s*', '', full_text, flags=re.IGNORECASE | re.DOTALL)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        cleaned_text = cleaned_text.strip()

        json_match = re.fullmatch(r'(\[.*\]|\{.*\})', cleaned_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            print("Strict full JSON block match failed. Trying to find embedded JSON.")
            json_search_embedded = re.search(r'(\[.*\]|\{.*\})', cleaned_text, re.DOTALL)
            if json_search_embedded:
                json_text = json_search_embedded.group(1)
                print(f"Found embedded JSON starting with: {json_text[:150]}...")
            else:
                print(f"No JSON array or object found in model's response after cleaning.\nCleaned text was:\n---\n{cleaned_text}\n---")
                return {"error": "No JSON found in model response", "response_text": cleaned_text}
        
        try:
            parsed_json = json.loads(json_text)
            print("Successfully parsed model output as JSON.")
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"Failed to parse extracted text as JSON: {e}\nExtracted JSON attempt was:\n---\n{json_text}\n---")
            return {"error": f"JSON parsing failed: {str(e)}", "json_text_attempt": json_text}

    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True) # exc_info=True logs stack trace
        return {"error": f"General processing error: {str(e)}"}
    finally:
        if uploaded_file_sdk_objects:
            print(f"Attempting to clean up {len(uploaded_file_sdk_objects)} uploaded file(s)...")
        for sdk_file_obj in uploaded_file_sdk_objects:
            if sdk_file_obj and hasattr(sdk_file_obj, 'name'): # Ensure it's a valid file object
                try:
                    client.files.delete(name=sdk_file_obj.name) # Use client from the try block
                    print(f"Successfully deleted temporary file: {sdk_file_obj.name}")
                except Exception as del_e:
                    print(f"Error deleting temporary file {sdk_file_obj.name}: {del_e}")
            else:
                print(f"Skipping deletion for an item that might not be a valid SDK file object: {sdk_file_obj}")


def save_gsam2_annotations(
    img_bgr: np.ndarray,
    dets_raw: sv.Detections,
    names: list[str],
    scores: list[float] | np.ndarray,
    xyxy: np.ndarray,
    masks: np.ndarray,
    img_p: Path,
    out: Path
):
    """
    Saves GSAM2 box+label annotations, mask overlay, and raw JSON results.

    - Resizes the input image to match mask dimensions to avoid index errors.

    Args:
        img_bgr (np.ndarray): Original image in BGR format (can be any resolution).
        dets_raw (sv.Detections): Detections object containing boxes and masks.
        names (List[str]): Class names for each detection.
        scores (List[float] or np.ndarray): Confidence scores for each detection.
        xyxy (np.ndarray): Bounding boxes (N,4) in xyxy format.
        masks (np.ndarray): Segmentation masks (N,H,W).
        img_p (Path): Path to the input image.
        out (Path): Output directory.
    """
    import json

    # ensure output dir exists
    out.mkdir(parents=True, exist_ok=True)

    # convert scores to list
    scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    # resize scene to match masks
    mask_h, mask_w = masks.shape[1], masks.shape[2]
    scene = cv2.resize(img_bgr, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)

    # 1) Box and (optional) label annotation
    box_img = sv.BoxAnnotator().annotate(scene=scene.copy(), detections=dets_raw)
    labels_txt = [f"{n} {s:.2f}" for n, s in zip(names, scores_list)]
    if len(labels_txt) == len(dets_raw):
        box_img = sv.LabelAnnotator().annotate(scene=box_img, detections=dets_raw, labels=labels_txt)
    else:
        print(
            f"Warning: Skipping label annotation, "
            f"{len(labels_txt)} labels vs {len(dets_raw)} detections for {img_p.name}"
        )
    cv2.imwrite(str(out / "groundingdino_annotated_image.jpg"), box_img)

    # 2) Mask overlay on annotated boxes
    mask_overlay = sv.MaskAnnotator(opacity=0.4).annotate(scene=box_img, detections=dets_raw)
    cv2.imwrite(str(out / "grounded_sam2_annotated_image_with_mask.jpg"), mask_overlay)

    # 3) Save raw JSON of annotations
    annotations = []
    for name, bbox, mask, score in zip(names, xyxy.tolist(), masks, scores_list):
        rle = mask_to_rle(mask)
        annotations.append({
            "class_name": name,
            "bbox": bbox,
            "segmentation": rle,
            "score": float(score)
        })
    raw_dict = {
        "image_path": str(img_p),
        "box_format": "xyxy",
        "img_width": mask_w,
        "img_height": mask_h,
        "annotations": annotations
    }
    with open(out / "gsam2_raw_results.json", 'w') as f:
        json.dump(raw_dict, f, indent=4)
    print(f"Saved GSAM2 results (JSON, images) for {img_p.name}")
