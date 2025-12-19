


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
# Import the dataset's class-to-index mapping to ensure consistency

IDX2CLASS = {
    #  0: "background",
    # 1: "Aloo Dry fry",
    # 2: "Avakaya Muddha Papu Rice",
    # 3: "Baby-Corn & Capsicum-Dry",
    # 4: "Cabbage Pakodi",
    # 5: "Cabbage fry",
    # 6: "Capsicum Paneer Curry",
    # 7: "Chakar-Pongal",
    # 8: "Chole-Masala",
    # 9: "Cluster Beans Curry",
    # 10: "Cucumber-Raitha",
    # 11: "Gobi Masala Curry",
    # 12: "Gutti Vankaya Curry",
    # 13: "Jeera Rice",
    # 14: "Mixed Curry",
    # 15: "Muskmelon",
    # 16: "Rajma",
    # 17: "Rasgulla",
    # 18: "Sambar",
    # 19: "Tomato Rasam",
    # 20: "Vankaya-Ali-Karam",
    # 21: "Veg-Biriyani",
    # 22: "aloo-curry",
    # 23: "curd",
    # 24: "dal",
    # 25: "fresh-chutney",
    # 26: "green-salad",
    # 27: "Moong-Beans-Curry",
    # 28: "khichdi",
    # 29: "lemon-rice",
    # 30: "live-roti-with-ghee",
    # 31: "non-spicy-curry-bottle-gourd",
    # 32: "papad",
    # 33: "plain-rice",
    # 34: "watermelon",
    # 35: "Aloo-Fry",
    # 36: "Banana",
    # 37: "Mix-Fruit",
    # 38: "Non-Spicy-Baby-Corn & Capsicum-Dry",
    # 39: "Sweet",
    # 40: "Tomato-Rice",
    # 41: "fried-papad-rings",
    # 42: "gravy",
    # 43: "ivy-gourd-fry",
    # 44: "mango-pickle",
    # 45: "papad-chat",
    # 46: "pepper-rasam",
    # 47: "pineapple",
    # 48: "corn-fry",
    # 49: "paneer-curry",
    # 50: "semiya"
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


def load_prototypes(proto_base: Path, date: str, use_mean: bool):
    proto_dir = proto_base / date
    # print(f"[load_protos] Looking for prototypes in {proto_dir}")
    if not proto_dir.is_dir():
        raise RuntimeError(f"No prototypes for date {date} in {proto_base}")
    raw = {}
    for f in proto_dir.rglob('*.npy'):
        cls = f.stem.split('_',1)[0]
        arr = np.load(f)
        raw.setdefault(cls, []).append(arr)
    proto_list, labels = [], []
    for cls, arrs in raw.items():
        arrs = np.stack(arrs,0)
        if use_mean:
            proto_list.append(arrs.mean(axis=0)); labels.append(cls)
        else:
            for v in arrs:
                proto_list.append(v); labels.append(cls)
    if not proto_list:
        raise RuntimeError(f"No prototypes loaded for date {date}")
    return np.stack(proto_list), labels


def classify_one(stem_dir: Path, proto_base: Path, use_mean: bool,
                 knn_k: int, pooling: str, conf_thresh: float,
                 orig_dir: Path, patch_size: int, out_root: Path):
    stem = stem_dir.name
    date = stem[:8]
    # Load prototypes for this date
    proto_mat, proto_labels = load_prototypes(proto_base, date, use_mean)
    knn = NearestNeighbors(n_neighbors=knn_k, metric='cosine').fit(proto_mat)

    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata and masks
    metadata = json.load(open(stem_dir/'metadata.json'))
    masks    = np.load(stem_dir/'masks.npy').astype(bool)

    # Prepare patch embeddings
    emb_list, centers, mask_ids = [], [], []
    for e in metadata:
        emb_list.append(np.load(stem_dir/e['embedding']))
        centers.append(tuple(e['center']))
        mask_ids.append(e['mask_id'])
    if not emb_list:
        return stem
    X = np.stack(emb_list,0)

    # k-NN and similarity
    dists, idxs = knn.kneighbors(X, return_distance=True)
    sims = 1 - dists

    # Per-patch votes & sims
    patch_votes, patch_sims = [], []
    for nn_labels, nn_sims in zip(idxs, sims):
        lbls = [proto_labels[i] for i in nn_labels]
        patch_votes.append(Counter(lbls).most_common(1)[0][0])
        sd = defaultdict(float)
        for lbl, s in zip(lbls, nn_sims): sd[lbl] += s
        patch_sims.append(sd)

    # Visual outputs: per-mask and full overlay
    mask_vis_dir = out_dir/'mask_visuals'; mask_vis_dir.mkdir(exist_ok=True)
    for idx, m in enumerate(masks):
        Image.fromarray((m.astype(np.uint8)*255)).save(mask_vis_dir/f'mask_{idx}.png')
    img_files = list(orig_dir.glob(f"{stem}.*"))
    if img_files:
        base_img = Image.open(img_files[0]).convert('RGB').resize((1024,1024))
        full_ov   = base_img.copy(); d = ImageDraw.Draw(full_ov)
        for m in masks:
            ys, xs = np.where(m)
            if ys.size: d.rectangle((xs.min(), ys.min(), xs.max(), ys.max()), outline='red', width=2)
        full_ov.save(out_dir/'full_mask_overlay.png')

    # Aggregate by mask
    per_mask_votes = defaultdict(list)
    per_mask_sims  = defaultdict(lambda: defaultdict(float))
    per_mask_boxes = defaultdict(list)
    all_boxes = []
    half = patch_size//2

    for (m_id, center, vote, sd) in zip(mask_ids, centers, patch_votes, patch_sims):
        cy, cx = center
        y0 = max(0, min(cy-half, 1024-patch_size)); x0 = max(0, min(cx-half, 1024-patch_size))
        x1, y1 = x0+patch_size, y0+patch_size
        per_mask_boxes[m_id].append((x0,y0,x1,y1)); all_boxes.append((x0,y0,x1,y1))
        per_mask_votes[m_id].append(vote)
        for lbl, s in sd.items(): per_mask_sims[m_id][lbl] += s

    # Classify per mask
    detections = []
    H,W = masks.shape[1], masks.shape[2]
    pred_maj = np.zeros((H,W),dtype=np.uint8)
    pred_pool= np.zeros((H,W),dtype=np.uint8)

    for m_id, votes in per_mask_votes.items():
        sims_dict = per_mask_sims[m_id]
        res = {}
        if pooling in ('majority','both'):
            mc, cnt = Counter(votes).most_common(1)[0]
            res['majority'] = {'class': mc, 'conf': cnt/len(votes)}
        if pooling in ('pooled','both'):
            pc = max(sims_dict, key=sims_dict.get)
            res['pooled'] = {'class': pc, 'conf': sims_dict[pc]/(knn_k*len(votes))}
        if res and max(r['conf'] for r in res.values()) >= conf_thresh:
            detections.append({'mask_id': m_id, **res})
            if 'majority' in res: pred_maj[masks[m_id]] = CLASS2IDX[res['majority']['class']]
            if 'pooled'   in res: pred_pool[masks[m_id]] = CLASS2IDX[res['pooled']['class']]

    # Save outputs
    with open(out_dir/'detections.json','w') as f: json.dump(detections,f,indent=2)
    Image.fromarray(pred_maj).save(out_dir/'pred_labelIds_majority.png')
    Image.fromarray(pred_pool).save(out_dir/'pred_labelIds_pooled.png')
    if all_boxes and img_files:
        vis = Image.open(img_files[0]).convert('RGB').resize((1024,1024)); dd = ImageDraw.Draw(vis)
        for r in all_boxes: dd.rectangle(r, outline='blue', width=2)
        vis.save(out_dir/'patch_overlay_classified.png')

    return m_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',           required=True)
    parser.add_argument('--input_dir',        required=True)
    parser.add_argument('--orig_images_dir',  required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--prototype_dir')
    parser.add_argument('--global_proto_dir')
    parser.add_argument('--knn_k',            type=int, default=3, choices=[1,3,4,5])
    parser.add_argument('--pooling',          choices=['majority','pooled','both'], default='both')
    parser.add_argument('--conf_threshold',   type=float, default=0.0)
    parser.add_argument('--patch_size',       type=int, default=64)
    parser.add_argument('--workers',          type=int, default=4)
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    proto_base = Path(args.global_proto_dir) if args.global_proto_dir else Path(args.prototype_dir)
    print(f"[main] Using prototypes from {proto_base}")
    inp = Path(args.input_dir)
    orig_root = Path(args.orig_images_dir)
    out = Path(args.output_dir)
    stems = natsorted([d for d in inp.iterdir() if d.is_dir()])
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(
            classify_one,
            stem,
            proto_base,
            cfg.embeddings.use_mean,
            args.knn_k,
            args.pooling,
            args.conf_threshold,
            orig_root,
            args.patch_size,
            out
        ): stem.name for stem in stems}
        for fut in as_completed(futures):
            name = futures[fut]
            try: fut.result(); print(f"[SUCCESS] {name}")
            except Exception as e: print(f"[FAIL]    {name}: {e}")

if __name__=='__main__':
    main()
