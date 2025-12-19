#!/usr/bin/env python3
"""
semantic_segmentation_eval.py

Compute class‐level and pixel‐level metrics for multi‐class semantic segmentation
(using index‐based maps instead of RGB colormaps).
"""
import argparse
import logging
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # <-- Added tqdm

# -----------------------------------------------------------------------------
# Your index→class mapping:
idx2class = {
    # 0: "background",
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
    # 11: "Dal-Fry",
    # 12: "Gobi Masala Curry",
    # 13: "Gutti Vankaya Curry",
    # 14: "Jeera Rice",
    # 15: "Leaf Dal",
    # 16: "Mixed Curry",
    # 17: "ivy-dal-gourd-curry",
    # 18: "Muskmelon",
    # 19: "pepper-rasam",
    # 20: "Rajma",
    # 21: "Rasgulla",
    # 22: "Sambar",
    # 23: "Tomato Rasam",
    # 24: "Vankaya-Ali-Karam",
    # 25: "Veg-Biriyani",
    # 26: "aloo-curry",
    # 27: "curd",
    # 28: "dal",
    # 29: "fresh-chutney",
    # 30: "green-salad",
    # 31: "Moong-Beans-Curry",
    # 32: "khichdi",
    # 33: "lemon-rice",
    # 34: "live-roti-with-ghee",
    # 35: "non-spicy-curry-bottle-gourd",
    # 36: "non-spicy-dal",
    # 37: "papad",
    # 38: "plain-rice",
    # 39: "watermelon",
    # 40: "Aloo-Fry",
    # 41: "Banana",
    # 42: "Mix-Fruit",
    # 43: "Non-Spicy-Baby-Corn & Capsicum-Dry",
    # 44: "Sweet",
    # 45: "Tomato-Rice",
    # 46: "fried-papad-rings",
    # 47: "gravy",
    # 48: "ivy-gourd-fry",
    # 49: "mango-pickle",
    # 50: "papad-chat",
    # 51: "pepper-rasam",
    # 52: "pineapple",
    # 53: "corn-fry",
    # 54: "horse-gram-curry",
    # 55: "paneer-curry",
    # 56: "semiya"
    # 0: "background",
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
NUM_CLASSES = len(idx2class)

# -----------------------------------------------------------------------------
# Globals for worker processes
global_gt_dir = None
global_min_area = None
global_iou_thr = None
global_pooling = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic segmentation outputs (index maps)"
    )
    parser.add_argument(
        '--parent_dir', required=True, type=Path,
        help="Directory containing one subfolder per sample, each with abs_mask.png"
    )
    parser.add_argument(
        '--gt_dir', required=True, type=Path,
        help="Directory containing ground‐truth index maps named with the same stem"
    )
    parser.add_argument(
        '--output_dir', default=Path('output'), type=Path,
        help="Directory to save metrics and heatmaps"
    )
    parser.add_argument(
        '--min_area', type=int, default=2200,
        help="Min blob area to keep (in pixels)"
    )
    parser.add_argument(
        '--iou_threshold', type=float, default=0.6,
        help="IoU threshold for matching blobs"
    )
    parser.add_argument(
        '--cmap', default='OrRd', type=str,
        help="Matplotlib colormap for confusion‐heatmap"
    )
    parser.add_argument(
        '--workers', type=int, default=cpu_count(),
        help="Number of parallel worker processes"
    )
    parser.add_argument('--pooling',         choices=['majority','pooled'])

    return parser.parse_args()


def init_worker(gt_dir: str, min_area: int, iou_thr: float, pooling: str):
    """Initialize globals in each worker process."""
    global global_gt_dir, global_min_area, global_iou_thr, global_pooling
    global_gt_dir = Path(gt_dir)
    global_min_area = min_area
    global_iou_thr = iou_thr
    global_pooling = pooling
    


def clean_small_blobs(id_map: np.ndarray, min_area: int) -> np.ndarray:
    """Zero‐out connected components smaller than min_area."""
    out = id_map.copy()
    for cls in np.unique(id_map):
        if cls == 0:
            continue
        lbl = label(id_map == cls)
        for region in regionprops(lbl):
            if region.area < min_area:
                out[tuple(region.coords.T)] = 0
    return out


def extract_blobs(id_map: np.ndarray, min_area: int):
    """Return a list of {'mask','class','area'} for each CC ≥ min_area."""
    blobs = []
    lbl = label(id_map)
    for region in regionprops(lbl):
        cls = id_map[tuple(region.coords[0])]
        if cls == 0 or region.area < min_area:
            continue
        mask = np.zeros_like(id_map, dtype=bool)
        mask[tuple(region.coords.T)] = True
        blobs.append({'mask': mask, 'class': int(cls), 'area': region.area})
    return blobs


def calculate_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    uni = np.logical_or(m1, m2).sum()
    return inter / uni if uni else 0.0


def process_subdir(subdir: Path):
    """
    For each sample folder:
      - load GT index map and PR index map (abs_mask.png),
      - clean small blobs in PR,
      - build blob lists, match them via IoU,
      - accumulate both class‐level (Counter) and pixel‐level (hist) stats.
    """
    # derive stem (e.g. "20250617_163025") from subdir name
    stem = subdir.name.split('_', 2)[:2]
    stem = "_".join(stem)

    # find corresponding GT file in global_gt_dir or from subdir
    # gt_path = subdir / 'gt_labelIds.png'
        gt_path = os.path.join(gt_dir, stem + '_gtFine_labelIds.png')
    if not os.path.exists(gt_path):
        logging.warning(f"Skipping {subdir.name}: no GT file at {gt_path}")
        return None

    # pr_path = subdir / 'pred_labelIds.png'
    if global_pooling == 'majority':
        pr_path = subdir / 'pred_labelIds_majority.png'
    elif global_pooling == 'pooled':
        pr_path = subdir / 'pred_labelIds_pooled.png'
    else:
        raise ValueError(f"Invalid pooling method: {global_pooling}")
    
    if not pr_path.exists():
        logging.warning(f"Skipping {subdir.name}: no PR file at {pr_path}")
        return None

    # load index maps directly
    gt_ids = np.array(Image.open(gt_path).convert('L').resize((1024,1024),Image.Resampling.NEAREST ), dtype=np.int32)
    pr_ids = np.array(Image.open(pr_path).convert('L'), dtype=np.int32)
    pr_ids = clean_small_blobs(pr_ids, global_min_area)

    gt_blobs = extract_blobs(gt_ids, global_min_area)
    pr_blobs = extract_blobs(pr_ids, global_min_area)

    conf = Counter()
    matched = set()
    # match predicted to GT via highest IoU
    for p in pr_blobs:
        best_iou, best_idx = 0.0, -1
        for idx, g in enumerate(gt_blobs):
            iou = calculate_iou(p['mask'], g['mask'])
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= global_iou_thr:
            conf[(gt_blobs[best_idx]['class'], p['class'])] += 1
            matched.add(best_idx)
        else:
            # assign to GT class with max overlap if IoU below threshold
            max_ov, best_cls = 0, 0
            for g in gt_blobs:
                ov = np.logical_and(p['mask'], g['mask']).sum()
                if ov > max_ov:
                    max_ov, best_cls = ov, g['class']
            conf[(best_cls, p['class'])] += 1

    # count GT blobs never matched → false negatives
    for idx, g in enumerate(gt_blobs):
        if idx not in matched:
            conf[(g['class'], 0)] += 1

    # pixel‐level histogram
    mask = (gt_ids >= 0) & (gt_ids < NUM_CLASSES)
    hist = np.bincount(
        NUM_CLASSES * gt_ids[mask].ravel() + pr_ids[mask].ravel(),
        minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    return conf, hist


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # collect all subfolders under parent_dir
    subdirs = [d for d in args.parent_dir.iterdir() if d.is_dir()]

    total_conf = Counter()
    total_hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    # spin up workers, wrapping imap_unordered with tqdm for progress monitoring
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(str(args.gt_dir), args.min_area, args.iou_threshold, args.pooling)
    ) as pool:
        for result in tqdm(pool.imap_unordered(process_subdir, subdirs),
                           total=len(subdirs),
                           desc="Processing samples"):
            if result is None:
                continue
            conf, hist = result
            total_conf.update(conf)
            total_hist += hist

    logging.info("Finished processing all samples.")

    # --- Class‐level confusion matrix (filtered) ---
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for (gt, pr), cnt in total_conf.items():
        cm[gt, pr] = cnt

    present = [
        i for i in range(NUM_CLASSES)
        if cm[i, :].sum() + cm[:, i].sum() > 0
    ]
    labels = [idx2class[i] for i in present]
    cm_filt = cm[np.ix_(present, present)]
    cm_df = pd.DataFrame(cm_filt, index=labels, columns=labels)
    cm_df.to_csv(args.output_dir / 'confusion_matrix.csv')

    # heatmap (no colorbar)
    plt.figure(figsize=(max(8, 0.3 * len(labels)), max(6, 0.3 * len(labels))))
    sns.heatmap(
        cm_df, annot=True, fmt='d', cmap=args.cmap,
        xticklabels=labels, yticklabels=labels,
        cbar=False, annot_kws={"size": max(4, 12 - 0.1 * len(labels))}
    )
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(args.output_dir / 'confusion_heatmap.png')
    plt.close()

    # classification report
    y_true, y_pred = [], []
    for (gt, pr), cnt in total_conf.items():
        if gt in present and pr in present:
            y_true += [gt] * cnt
            y_pred += [pr] * cnt
    report = classification_report(
        y_true, y_pred,
        labels=present,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        args.output_dir / 'classification_report.csv'
    )

    # overall metrics
    overall = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    with open(args.output_dir / 'overall_metrics.txt', 'w') as f:
        for k, v in overall.items():
            f.write(f"{k}: {v:.4f}\n")

    # --- Pixel‐level metrics ---
    acc = np.trace(total_hist) / total_hist.sum()
    acc_cls = np.diag(total_hist) / np.maximum(1, total_hist.sum(axis=1))
    iou = np.diag(total_hist) / (
        total_hist.sum(axis=1) +
        total_hist.sum(axis=0) -
        np.diag(total_hist)
    )
    pixel_df = pd.DataFrame({
        'Class': [idx2class[i] for i in range(NUM_CLASSES)],
        'PixelAccuracy': acc_cls,
        'IoU': iou
    })
    pixel_df.loc[len(pixel_df)] = ['Mean', acc, np.nanmean(iou)]
    pixel_df.to_csv(args.output_dir / 'pixel_metrics.csv', index=False)

    # log any GT classes never predicted
    missing = set(range(NUM_CLASSES)) - {gt for (gt, _) in total_conf.keys()}
    if missing:
        logging.info("GT classes with no matched predictions:")
        for c in sorted(missing):
            fn_count = sum(
                cnt for (gt, pr), cnt in total_conf.items()
                if gt == c and pr == 0
            )
            logging.info(f"  - {idx2class[c]} (ID {c}): FN={fn_count}")


if __name__ == '__main__':
    main()



# python eval_blob_indexbased.py \
#     --parent_dir ./results/multiscale_seg_v1_RERUN \
#         --gt_dir ./data/test/masks \
#             --output_dir results/seg_v1_results \
#                 --min_area 2200 --iou_threshold 0.6 --cmap OrRd --workers 4