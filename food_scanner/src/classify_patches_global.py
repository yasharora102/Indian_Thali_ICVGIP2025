
# import argparse
# import json
# from pathlib import Path
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# from collections import defaultdict, Counter
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from PIL import Image, ImageDraw
# import cv2
# import pycocotools.mask as mask_util
# from config import ExperimentConfig
# from natsort import natsorted
# # Import the dataset's class-to-index mapping to ensure consistency

# IDX2CLASS = {
#     # 0: "background",
#     # 1: "Aloo Dry fry",
#     # 2: "Avakaya Muddha Papu Rice",
#     # 3: "Baby-Corn & Capsicum-Dry",
#     # 4: "Cabbage Pakodi",
#     # 5: "Cabbage fry",
#     # 6: "Capsicum Paneer Curry",
#     # 7: "Chakar-Pongal",
#     # 8: "Chole-Masala",
#     # 9: "Cluster Beans Curry",
#     # 10: "Cucumber-Raitha",
#     # 11: "Gobi Masala Curry",
#     # 12: "Gutti Vankaya Curry",
#     # 13: "Jeera Rice",
#     # 14: "Mixed Curry",
#     # 15: "Muskmelon",
#     # 16: "Rajma",
#     # 17: "Rasgulla",
#     # 18: "Sambar",
#     # 19: "Tomato Rasam",
#     # 20: "Vankaya-Ali-Karam",
#     # 21: "Veg-Biriyani",
#     # 22: "aloo-curry",
#     # 23: "curd",
#     # 24: "dal",
#     # 25: "fresh-chutney",
#     # 26: "green-salad",
#     # 27: "Moong-Beans-Curry",
#     # 28: "khichdi",
#     # 29: "lemon-rice",
#     # 30: "live-roti-with-ghee",
#     # 31: "non-spicy-curry-bottle-gourd",
#     # 32: "papad",
#     # 33: "plain-rice",
#     # 34: "watermelon",
#     # 35: "Aloo-Fry",
#     # 36: "Banana",
#     # 37: "Mix-Fruit",
#     # 38: "Non-Spicy-Baby-Corn & Capsicum-Dry",
#     # 39: "Sweet",
#     # 40: "Tomato-Rice",
#     # 41: "fried-papad-rings",
#     # 42: "gravy",
#     # 43: "ivy-gourd-fry",
#     # 44: "mango-pickle",
#     # 45: "papad-chat",
#     # 46: "pepper-rasam",
#     # 47: "pineapple",
#     # 48: "corn-fry",
#     # 49: "paneer-curry",
#     # 50: "semiya"
#     0: "background",
#   1: "Bottle-gourd-curry",
#   2: "aloo-capsicum",
#   3: "aloo-curry",
#   4: "aloo-fry",
#   5: "beans-curry",
#   6: "beetroot-kobari",
#   7: "beetroot-poriyal",
#   8: "bisi-bele-bath",
#   9: "boondi",
#   10: "cabbage-dry",
#   11: "channa-brinjal",
#   12: "chicken-dum-biryani",
#   13: "chutney",
#   14: "curd",
#   15: "dondakaya-fry",
#   16: "kakarakaya-fry",
#   17: "kofta-curry",
#   18: "leaf-dal",
#   19: "mango-pickle",
#   20: "masoor-dal",
#   21: "mirchi-ka-salan",
#   22: "muddha-pappu",
#   23: "non-spicy-curry",
#   24: "non-spicy-dal",
#   25: "pachi-pulusu",
#   26: "papad",
#   27: "payasam",
#   28: "phulka",
#   29: "raita",
#   30: "rajma",
#   31: "rasam",
#   32: "salad",
#   33: "sambar",
#   34: "steamed-rice",
#   35: "tomato-pappu",
#   36: "veg-dum-briyani",
#   37: "veg-pulao",
#   38: "Watermelon",
#   39: "Papaya",
#   40: "Banana",
#   41: "Muskmelon"
# }
# CLASS2IDX = {v: k for k, v in IDX2CLASS.items()}



# def load_prototypes(proto_base: Path, allowed: set[str], use_mean: bool):
#     """
#     Load prototype embeddings only for the classes listed in `allowed`,
#     assuming a global directory layout: proto_base/ClassName/*.npy
#     """
#     raw = {}
#     for cls_name in allowed:
#         cls_dir = proto_base / cls_name
#         if not cls_dir.is_dir():
#             continue
#         for npy in cls_dir.glob('*.npy'):
#             arr = np.load(npy)
#             raw.setdefault(cls_name, []).append(arr)

#     if not raw:
#         raise RuntimeError(f"No prototypes found for allowed classes {allowed} in {proto_base}")

#     proto_list, labels = [], []
#     for cls, arrs in raw.items():
#         stacked = np.stack(arrs, axis=0)
#         if use_mean:
#             proto_list.append(stacked.mean(axis=0))
#             labels.append(cls)
#         else:
#             for v in stacked:
#                 proto_list.append(v)
#                 labels.append(cls)

#     proto_mat = np.stack(proto_list, axis=0)
#     return proto_mat, labels


# def classify_one(
#     stem_dir: Path,
#     proto_base: Path,
#     use_mean: bool,
#     knn_k: int,
#     pooling: str,
#     conf_thresh: float,
#     orig_dir: Path,
#     patch_size: int,
#     out_root: Path,
#     allowed: set[str]
# ):
#     stem = stem_dir.name

#     # Load prototypes based on menu filters
#     proto_mat, proto_labels = load_prototypes(proto_base, allowed, use_mean)
#     knn = NearestNeighbors(n_neighbors=knn_k, metric='cosine').fit(proto_mat)

#     out_dir = out_root / stem
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Load masks & metadata
#     metadata = json.load(open(stem_dir / 'metadata.json'))
#     masks = np.load(stem_dir / 'masks.npy').astype(bool)

#     # Gather patch embeddings
#     emb_list, centers, mask_ids = [], [], []
#     for e in metadata:
#         emb_list.append(np.load(stem_dir / e['embedding']))
#         centers.append(tuple(e['center']))
#         mask_ids.append(e['mask_id'])
#     if not emb_list:
#         return stem
#     X = np.stack(emb_list, axis=0)

#     # Batch k-NN
#     dists, idxs = knn.kneighbors(X, return_distance=True)
#     sims = 1 - dists

#     # Per-patch classification data
#     patch_votes, patch_sims = [], []
#     for nn_labels, nn_sims in zip(idxs, sims):
#         lbls = [proto_labels[i] for i in nn_labels]
#         vote = Counter(lbls).most_common(1)[0][0]
#         patch_votes.append(vote)
#         sd = defaultdict(float)
#         for lbl, s in zip(lbls, nn_sims):
#             sd[lbl] += s
#         patch_sims.append(sd)

#     # Save individual mask visuals
#     mask_vis_dir = out_dir / 'mask_visuals'
#     mask_vis_dir.mkdir(exist_ok=True)
#     for idx, m in enumerate(masks):
#         mask_img = (m.astype(np.uint8) * 255).astype(np.uint8)
#         Image.fromarray(mask_img).save(mask_vis_dir / f'mask_{idx}.png')

#     # Full-mask overlay
#     img_files = list(orig_dir.glob(f"{stem}.*"))
#     if img_files:
#         base_img = Image.open(img_files[0]).convert('RGB').resize((1024,1024))
#         overlay = base_img.copy()
#         draw = ImageDraw.Draw(overlay)
#         for m in masks:
#             ys, xs = np.where(m)
#             if ys.size:
#                 draw.rectangle((xs.min(), ys.min(), xs.max(), ys.max()), outline='red', width=2)
#         overlay.save(out_dir / 'full_mask_overlay.png')

#     # Aggregate and classify per-mask
#     per_mask_votes = defaultdict(list)
#     per_mask_sims = defaultdict(lambda: defaultdict(float))
#     all_boxes = []
#     half = patch_size // 2

#     for m_id, center, vote, sd in zip(mask_ids, centers, patch_votes, patch_sims):
#         cy, cx = center
#         y0 = max(0, min(cy-half, 1024-patch_size))
#         x0 = max(0, min(cx-half, 1024-patch_size))
#         x1, y1 = x0 + patch_size, y0 + patch_size
#         all_boxes.append((x0, y0, x1, y1))

#         per_mask_votes[m_id].append(vote)
#         for lbl, s in sd.items():
#             per_mask_sims[m_id][lbl] += s

#     detections = []
#     H, W = masks.shape[1], masks.shape[2]
#     pred_maj = np.zeros((H, W), dtype=np.uint8)
#     pred_pool = np.zeros((H, W), dtype=np.uint8)

#     for m_id, votes in per_mask_votes.items():
#         sims_dict = per_mask_sims[m_id]
#         res = {}
#         if pooling in ('majority', 'both'):
#             mc, cnt = Counter(votes).most_common(1)[0]
#             res['majority'] = {'class': mc, 'conf': cnt / len(votes)}
#         if pooling in ('pooled', 'both'):
#             pc = max(sims_dict, key=sims_dict.get)
#             res['pooled'] = {'class': pc, 'conf': sims_dict[pc] / (knn_k * len(votes))}
#         if res and max(r['conf'] for r in res.values()) >= conf_thresh:
#             detections.append({'mask_id': m_id, **res})
#             if 'majority' in res:
#                 pred_maj[masks[m_id]] = CLASS2IDX[res['majority']['class']]
#             if 'pooled' in res:
#                 pred_pool[masks[m_id]] = CLASS2IDX[res['pooled']['class']]

#     # Save
#     with open(out_dir / 'detections.json', 'w') as f:
#         json.dump(detections, f, indent=2)
#     Image.fromarray(pred_maj).save(out_dir / 'pred_labelIds_majority.png')
#     Image.fromarray(pred_pool).save(out_dir / 'pred_labelIds_pooled.png')

#     # Patch overlay
#     if all_boxes and img_files:
#         vis = Image.open(img_files[0]).convert('RGB').resize((1024,1024))
#         draw = ImageDraw.Draw(vis)
#         for rect in all_boxes:
#             draw.rectangle(rect, outline='blue', width=2)
#         vis.save(out_dir / 'patch_overlay_classified.png')

#     return stem


# def main():
#     parser = argparse.ArgumentParser(description="Classify with menu-based filtering of global prototypes")
#     parser.add_argument('--config',          required=True)
#     parser.add_argument('--input_dir',       required=True)
#     parser.add_argument('--orig_images_dir', required=True)
#     parser.add_argument('--output_dir',      required=True)
#     parser.add_argument('--prototype_dir',   required=True, help='Global dir with ClassName subfolders')
#     parser.add_argument('--menu_json',       default='/home/nutrition/WED_menu.json', help='JSON mapping date→[classes]')
#     parser.add_argument('--knn_k',           type=int, default=3, choices=[1,3,4,5])
#     parser.add_argument('--pooling',         choices=['majority','pooled','both'], default='both')
#     parser.add_argument('--conf_threshold',  type=float, default=0.0)
#     parser.add_argument('--patch_size',      type=int, default=64)
#     parser.add_argument('--workers',         type=int, default=4)
#     args = parser.parse_args()

#     cfg = ExperimentConfig.load(args.config)
#     proto_base = Path(args.prototype_dir)
#     menu = json.load(open(args.menu_json))

#     inp = Path(args.input_dir)
#     orig = Path(args.orig_images_dir)
#     out = Path(args.output_dir)
#     stems = sorted([d for d in inp.iterdir() if d.is_dir()])

#     with ProcessPoolExecutor(max_workers=args.workers) as exe:
#         futures = {}
#         for stem_dir in stems:
#             stem = stem_dir.name
#             date = stem[:8]
#             allowed = set(menu.get(date, []))
#             futures[exe.submit(
#                 classify_one,
#                 stem_dir,
#                 proto_base,
#                 cfg.embeddings.use_mean,
#                 args.knn_k,
#                 args.pooling,
#                 args.conf_threshold,
#                 orig,
#                 args.patch_size,
#                 out,
#                 allowed
#             )] = stem

#         for fut in as_completed(futures):
#             name = futures[fut]
#             try:
#                 fut.result()
#                 print(f"[SUCCESS] {name}")
#             except Exception as e:
#                 print(f"[FAIL]    {name}: {e}")

# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw
import cv2
import pycocotools.mask as mask_util
from config import ExperimentConfig
from natsort import natsorted

# ── Class maps ────────────────────────────────────────────────────────
IDX2CLASS = {
  0: "background",
  1: "Bottle-gourd-curry",
  2: "aloo-capsicum",
  3: "aloo-curry",
  4: "aloo-fry",
  5: "beans-curry",
  6: "beetroot-kobari",
  7: "beetroot-poriyal",
  8: "bisi-bele-bath",
  9: "boondi",
  10: "cabbage-dry",
  11: "channa-brinjal",
  12: "chicken-dum-biryani",
  13: "chutney",
  14: "curd",
  15: "dondakaya-fry",
  16: "kakarakaya-fry",
  17: "kofta-curry",
  18: "leaf-dal",
  19: "mango-pickle",
  20: "masoor-dal",
  21: "mirchi-ka-salan",
  22: "muddha-pappu",
  23: "non-spicy-curry",
  24: "non-spicy-dal",
  25: "pachi-pulusu",
  26: "papad",
  27: "payasam",
  28: "phulka",
  29: "raita",
  30: "rajma",
  31: "rasam",
  32: "salad",
  33: "sambar",
  34: "steamed-rice",
  35: "tomato-pappu",
  36: "veg-dum-briyani",
  37: "veg-pulao",
  38: "Watermelon",
  39: "Papaya",
  40: "Banana",
  41: "Muskmelon"
}
CLASS2IDX = {v: k for k, v in IDX2CLASS.items()}

# ── Metadata → mask union bboxes ──────────────────────────────────────
def merge_bboxes_from_metadata(metadata):
    """
    Build a union bbox per mask_id from per-patch bboxes in metadata.json.
    Returns: dict[mask_id] -> (x0, y0, x1, y1)
    """
    boxes = {}
    for e in metadata:
        mid = int(e["mask_id"])
        x0, y0, x1, y1 = map(int, e["bbox"])
        if mid not in boxes:
            boxes[mid] = [x0, y0, x1, y1]
        else:
            b = boxes[mid]
            b[0] = min(b[0], x0); b[1] = min(b[1], y0)
            b[2] = max(b[2], x1); b[3] = max(b[3], y1)
    return {k: tuple(v) for k, v in boxes.items()}

def _contains(a, b, pad=2):
    """
    True if bbox `a` contains bbox `b` with a small tolerance `pad`.
    a,b = (x0,y0,x1,y1)
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 + pad <= bx0) and (ay0 + pad <= by0) and (ax1 - pad >= bx1) and (ay1 - pad >= by1)

def find_plate_via_metadata_containment(metadata, min_children=4, min_fraction=0.30, pad=2):
    """
    Identify a 'plate' mask purely via metadata bboxes:
    - Compute union bbox for each mask_id
    - Count how many other mask bboxes each bbox fully contains
    - Exclude the mask with the MOST children if:
        children >= min_children AND children/(num_masks-1) >= min_fraction
    Returns: set of mask_ids to exclude (empty or {one_id})
    """
    boxes = merge_bboxes_from_metadata(metadata)
    mids = list(boxes.keys())
    if len(mids) <= 1:
        return set()

    child_counts = {m: 0 for m in mids}
    for i in mids:
        for j in mids:
            if i == j:
                continue
            if _contains(boxes[i], boxes[j], pad=pad):
                child_counts[i] += 1

    best_id = max(child_counts, key=lambda k: child_counts[k])
    best_cnt = child_counts[best_id]
    total_others = max(1, len(mids) - 1)
    frac = best_cnt / total_others

    excluded = set()
    if best_cnt >= min_children and frac >= min_fraction:
        excluded.add(best_id)
    return excluded

# ── Prototypes ────────────────────────────────────────────────────────
def load_prototypes(proto_base: Path, allowed: set[str], use_mean: bool):
    """
    Load prototype embeddings only for classes in `allowed`.
    Directory layout: proto_base/ClassName/*.npy
    """
    raw = {}
    for cls_name in allowed:
        cls_dir = proto_base / cls_name
        if not cls_dir.is_dir():
            continue
        for npy in cls_dir.glob('*.npy'):
            arr = np.load(npy)
            raw.setdefault(cls_name, []).append(arr)

    if not raw:
        raise RuntimeError(f"No prototypes found for allowed classes {allowed} in {proto_base}")

    proto_list, labels = [], []
    for cls, arrs in raw.items():
        stacked = np.stack(arrs, axis=0)
        if use_mean:
            proto_list.append(stacked.mean(axis=0))
            labels.append(cls)
        else:
            for v in stacked:
                proto_list.append(v)
                labels.append(cls)

    proto_mat = np.stack(proto_list, axis=0)
    return proto_mat, labels

# ── Core ──────────────────────────────────────────────────────────────
def classify_one(
    stem_dir: Path,
    proto_base: Path,
    use_mean: bool,
    knn_k: int,
    pooling: str,
    conf_thresh: float,
    orig_dir: Path,
    patch_size: int,
    out_root: Path,
    allowed: set[str],
    plate_min_children: int,
    plate_min_fraction: float,
    contain_pad: int
):
    stem = stem_dir.name

    # Prototypes (menu-filtered)
    proto_mat, proto_labels = load_prototypes(proto_base, allowed, use_mean)
    knn = NearestNeighbors(n_neighbors=knn_k, metric='cosine').fit(proto_mat)

    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load masks & metadata
    metadata = json.load(open(stem_dir / 'metadata.json'))
    masks = np.load(stem_dir / 'masks.npy').astype(bool)
    _, H, W = masks.shape

    # ── Exclude plate via metadata containment ──
    excluded = find_plate_via_metadata_containment(
        metadata,
        min_children=plate_min_children,
        min_fraction=plate_min_fraction,
        pad=contain_pad
    )
    if excluded:
        print(f"[{stem}] Excluding plate-like mask(s) via containment: {sorted(list(excluded))}")

    # Gather patch embeddings (skip excluded masks)
    emb_list, centers, mask_ids = [], [], []
    for e in metadata:
        m_id = int(e['mask_id'])
        if m_id in excluded:
            continue
        emb_list.append(np.load(stem_dir / e['embedding']))
        centers.append(tuple(e['center']))
        mask_ids.append(m_id)

    if not emb_list:
        # Nothing to classify after dropping plate → still dump empty artifacts for traceability
        with open(out_dir / 'detections.json', 'w') as f:
            json.dump([], f, indent=2)
        Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(out_dir / 'pred_labelIds_majority.png')
        Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(out_dir / 'pred_labelIds_pooled.png')
        return stem

    X = np.stack(emb_list, axis=0)

    # Batch k-NN
    dists, idxs = knn.kneighbors(X, return_distance=True)
    sims = 1 - dists

    # Per-patch classification summary
    patch_votes, patch_sims = [], []
    for nn_labels, nn_sims in zip(idxs, sims):
        lbls = [proto_labels[i] for i in nn_labels]
        vote = Counter(lbls).most_common(1)[0][0]
        patch_votes.append(vote)
        sd = defaultdict(float)
        for lbl, s in zip(lbls, nn_sims):
            sd[lbl] += s
        patch_sims.append(sd)

    # Save individual (non-excluded) mask visuals
    mask_vis_dir = out_dir / 'mask_visuals'
    mask_vis_dir.mkdir(exist_ok=True)
    for idx, m in enumerate(masks):
        if idx in excluded:
            continue
        mask_img = (m.astype(np.uint8) * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(mask_vis_dir / f'mask_{idx}.png')

    # Full-mask overlay (skip excluded)
    img_files = list(orig_dir.glob(f"{stem}.*"))
    if img_files:
        base_img = Image.open(img_files[0]).convert('RGB').resize((1024, 1024))
        overlay = base_img.copy()
        draw = ImageDraw.Draw(overlay)
        for idx, m in enumerate(masks):
            if idx in excluded:
                continue
            ys, xs = np.where(m)
            if ys.size:
                draw.rectangle((xs.min(), ys.min(), xs.max(), ys.max()), outline='red', width=2)
        overlay.save(out_dir / 'full_mask_overlay.png')

    # Aggregate & classify per-mask (non-excluded)
    per_mask_votes = defaultdict(list)
    per_mask_sims = defaultdict(lambda: defaultdict(float))
    all_boxes = []
    half = patch_size // 2

    for m_id, center, vote, sd in zip(mask_ids, centers, patch_votes, patch_sims):
        cy, cx = center
        y0 = max(0, min(cy - half, 1024 - patch_size))
        x0 = max(0, min(cx - half, 1024 - patch_size))
        x1, y1 = x0 + patch_size, y0 + patch_size
        all_boxes.append((x0, y0, x1, y1))

        per_mask_votes[m_id].append(vote)
        for lbl, s in sd.items():
            per_mask_sims[m_id][lbl] += s

    detections = []
    pred_maj = np.zeros((H, W), dtype=np.uint8)
    pred_pool = np.zeros((H, W), dtype=np.uint8)

    for m_id, votes in per_mask_votes.items():
        sims_dict = per_mask_sims[m_id]
        res = {}
        if pooling in ('majority', 'both'):
            mc, cnt = Counter(votes).most_common(1)[0]
            res['majority'] = {'class': mc, 'conf': cnt / len(votes)}
        if pooling in ('pooled', 'both'):
            pc = max(sims_dict, key=sims_dict.get)
            res['pooled'] = {'class': pc, 'conf': sims_dict[pc] / (knn_k * len(votes))}
        if res and max(r['conf'] for r in res.values()) >= conf_thresh:
            detections.append({'mask_id': m_id, **res})
            if 'majority' in res:
                pred_maj[masks[m_id]] = CLASS2IDX[res['majority']['class']]
            if 'pooled' in res:
                pred_pool[masks[m_id]] = CLASS2IDX[res['pooled']['class']]

    # Save results
    with open(out_dir / 'detections.json', 'w') as f:
        json.dump(detections, f, indent=2)
    Image.fromarray(pred_maj).save(out_dir / 'pred_labelIds_majority.png')
    Image.fromarray(pred_pool).save(out_dir / 'pred_labelIds_pooled.png')

    # Patch overlay
    if all_boxes and img_files:
        vis = Image.open(img_files[0]).convert('RGB').resize((1024, 1024))
        draw = ImageDraw.Draw(vis)
        for rect in all_boxes:
            draw.rectangle(rect, outline='blue', width=2)
        vis.save(out_dir / 'patch_overlay_classified.png')

    return stem

# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Global classifier with menu filter and metadata-based plate exclusion")
    parser.add_argument('--config',          required=True)
    parser.add_argument('--input_dir',       required=True)
    parser.add_argument('--orig_images_dir', required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--prototype_dir',   required=True, help='Global dir with ClassName subfolders')
    parser.add_argument('--menu_json',       default='./config/menu.json', help='JSON mapping date→[classes]')

    parser.add_argument('--knn_k',           type=int, default=3, choices=[1, 3, 4, 5])
    parser.add_argument('--pooling',         choices=['majority', 'pooled', 'both'], default='both')
    parser.add_argument('--conf_threshold',  type=float, default=0.0)
    parser.add_argument('--patch_size',      type=int, default=64)
    parser.add_argument('--workers',         type=int, default=4)

    # Plate exclusion (metadata containment)
    parser.add_argument('--plate_min_children', type=int, default=4,
                        help='Min number of other mask bboxes contained to mark as plate.')
    parser.add_argument('--plate_min_fraction', type=float, default=0.30,
                        help='Min fraction of (num_masks-1) contained.')
    parser.add_argument('--contain_pad',        type=int, default=2,
                        help='Tolerance (pixels) when testing bbox containment.')

    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    proto_base = Path(args.prototype_dir)
    menu = json.load(open(args.menu_json))

    inp = Path(args.input_dir)
    orig = Path(args.orig_images_dir)
    out = Path(args.output_dir)
    stems = natsorted([d for d in inp.iterdir() if d.is_dir()])

    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {}
        for stem_dir in stems:
            stem = stem_dir.name
            date = stem[:8]
            allowed = set(menu.get(date, []))
            futures[exe.submit(
                classify_one,
                stem_dir,
                proto_base,
                cfg.embeddings.use_mean,
                args.knn_k,
                args.pooling,
                args.conf_threshold,
                orig,
                args.patch_size,
                out,
                allowed,
                args.plate_min_children,
                args.plate_min_fraction,
                args.contain_pad
            )] = stem

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                print(f"[SUCCESS] {name}")
            except Exception as e:
                print(f"[FAIL]    {name}: {e}")

if __name__ == '__main__':
    main()
