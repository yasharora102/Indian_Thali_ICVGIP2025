import csv
import json
from pathlib import Path
import numpy as np
import pycocotools.mask as mask_util
import cv2
from PIL import Image
import supervision as sv
import os
from google import genai 
from google.genai import types
import re
from scipy.spatial.distance import mahalanobis

class DistanceSimilarity:
    def __init__(self, proto_embs: dict[str, np.ndarray]):
        self.proto_embs = proto_embs
        all_feats = np.stack(list(proto_embs.values()), axis=0)
        cov = np.cov(all_feats, rowvar=False) + np.eye(all_feats.shape[1]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)

    def euclidean(self, feat: np.ndarray) -> dict[str, float]:
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

def mask_to_rle(mask: np.ndarray) -> dict:
    if mask.ndim == 2:
        mask_fortran = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(mask_fortran)
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    elif mask.ndim == 3 and mask.shape[2] == 1:
        mask_fortran = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(mask_fortran)[0]
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    else:
        raise ValueError(f"Mask must be 2D or 3D with last dim 1, got shape {mask.shape}")

def remove_overlapping_masks(masks: np.ndarray, boxes: np.ndarray, names: list, containment_threshold: float = 0.9):
    N = masks.shape[0]
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]: continue
        for j in range(N):
            if i == j or not keep[j]: continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            area_i = masks[i].sum()
            containment = inter / (area_i + 1e-6)
            if containment > containment_threshold and area_i < masks[j].sum():
                keep[i] = False
                break
    removed_indices = np.where(~keep)[0].tolist()
    return masks[keep], boxes[keep], [n for k, n in enumerate(names) if keep[k]], removed_indices

def save_raw_masks(masks, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img.save(out_dir / f'mask_{i}.png')

def save_results(*, img_rgb, class_masks, confs, classes, cmap, out_dir, original_image_path = None, generate_calories = False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_rgb).save(out_dir/"input.jpg")
    H,W = img_rgb.shape[:2]
    combined = np.zeros((H,W), bool)
    for c in classes:
        if c in confs:
            combined |= class_masks[c]
    Image.fromarray((combined*255).astype('uint8')).save(out_dir/"combined_mask.png")
    
    ov = img_rgb.copy()
    for c in classes:
        if c in confs:
            m = class_masks[c]; col=cmap.get(c,(255,255,255))
            ov[m] = ((ov[m].astype(float)*0.5)+np.array(col)*0.5).astype('uint8')
    Image.fromarray(ov).save(out_dir/"overlay.png")

    dets = []
    for c in classes:
        if c in confs:
            rle_mask = mask_to_rle(class_masks[c])
            dets.append({
                'class': c,
                'confidence': round(float(confs[c]), 3),
                'segmentation_rle': rle_mask
            })
    with open(out_dir/"classes.json",'w') as f:
        json.dump({'detections':dets}, f, indent=2)

    if generate_calories:
        masks_path = sorted((out_dir/"cutouts"/"masks").glob("*.png"))
        cal_table = estimate_calories(original_image_path, str(out_dir/"classes.json"), masks_path)
        if cal_table:
            with open(str(out_dir/"classes.json"), 'r+') as f:
                doc = json.load(f)
                doc['cal_estimation'] = cal_table
                f.seek(0); f.truncate()
                json.dump(doc, f, indent=2)

def compute_ground_truth_label(gt_png_path, classes, class2idx, target_size=(1024,1024)):
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
        print("API key (GOOGLE_API_KEY) is not set.")
        return {"error": "API key not configured"}

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        return {"error": f"Client initialization failed: {str(e)}"}

    uploaded_file_sdk_objects = []
    try:
        main_image_file_obj = client.files.upload(file=image_path)
        uploaded_file_sdk_objects.append(main_image_file_obj)

        with open(classes_json_path, 'r') as f:
            detected_classes = json.load(f)
        
        for mp in mask_paths:
            mask_file_obj = client.files.upload(file=mp)
            uploaded_file_sdk_objects.append(mask_file_obj)

        model_prompt_parts = [types.Part.from_uri(file_uri=main_image_file_obj.uri, mime_type=main_image_file_obj.mime_type)]
        for sdk_file_obj in uploaded_file_sdk_objects[1:]:
            model_prompt_parts.append(types.Part.from_uri(file_uri=sdk_file_obj.uri, mime_type=sdk_file_obj.mime_type))

        prompt_text_content = f"""
        Analyze all the files provided. Item Details: {json.dumps(detected_classes, indent=2)}
        Provide estimated "Calorie Count" and "Weight Count" for each item.
        Return strictly a JSON array of objects with fields: "Item Name", "Calorie Count", "Weight Count".
        """
        model_prompt_parts.append(types.Part.from_text(text=prompt_text_content))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=model_prompt_parts)],
            config=types.GenerateContentConfig(response_mime_type="text/plain")
        )

        full_text = response.text.strip()
        cleaned_text = re.sub(r'^```(?:json)?\s*', '', full_text, flags=re.IGNORECASE | re.DOTALL)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.IGNORECASE | re.DOTALL).strip()
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            return {"error": "JSON parsing failed", "text": cleaned_text}

    except Exception as e:
        return {"error": f"General processing error: {str(e)}"}
    finally:
        for sdk_file_obj in uploaded_file_sdk_objects:
             try: client.files.delete(name=sdk_file_obj.name)
             except: pass

def save_gsam2_annotations(img_bgr, dets_raw, names, scores, xyxy, masks, img_p, out):
    import json
    out.mkdir(parents=True, exist_ok=True)
    mask_h, mask_w = masks.shape[1], masks.shape[2]
    scene = cv2.resize(img_bgr, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
    box_img = sv.BoxAnnotator().annotate(scene=scene.copy(), detections=dets_raw)
    cv2.imwrite(str(out / "groundingdino_annotated_image.jpg"), box_img)
    annotations = []
    for name, bbox, mask, score in zip(names, xyxy.tolist(), masks, scores.tolist() if hasattr(scores, 'tolist') else list(scores)):
        rle = mask_to_rle(mask)
        annotations.append({"class_name": name, "bbox": bbox, "segmentation": rle, "score": float(score)})
    with open(out / "gsam2_raw_results.json", 'w') as f:
        json.dump({"image_path": str(img_p), "annotations": annotations}, f, indent=4)
