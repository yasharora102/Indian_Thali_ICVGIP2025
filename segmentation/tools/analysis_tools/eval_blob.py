#!/usr/bin/env python3
"""
semantic_segmentation_eval_parallel.py

Compute class‐level and pixel‐level metrics for multi‐class semantic segmentation
(using index‐based maps instead of RGB colormaps), with parallel file loading.
"""
import argparse
import logging
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage.measure import label, regionprops
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    accuracy_score
)
# -----------------------------------------------------------------------------
# Your index→class mapping:
idx2class = { 0: "background",
    1: "Aloo Dry fry",
    2: "Avakaya Muddha Papu Rice",
    3: "Baby-Corn & Capsicum-Dry",
    4: "Cabbage Pakodi",
    5: "Cabbage fry",
    6: "Capsicum Paneer Curry",
    7: "Chakar-Pongal",
    8: "Chole-Masala",
    9: "Cluster Beans Curry",
    10: "Cucumber-Raitha",
    11: "Gobi Masala Curry",
    12: "Gutti Vankaya Curry",
    13: "Jeera Rice",
    14: "Mixed Curry",
    15: "Muskmelon",
    16: "Rajma",
    17: "Rasgulla",
    18: "Sambar",
    19: "Tomato Rasam",
    20: "Vankaya-Ali-Karam",
    21: "Veg-Biriyani",
    22: "aloo-curry",
    23: "curd",
    24: "dal",
    25: "fresh-chutney",
    26: "green-salad",
    27: "Moong-Beans-Curry",
    28: "khichdi",
    29: "lemon-rice",
    30: "live-roti-with-ghee",
    31: "non-spicy-curry-bottle-gourd",
    32: "papad",
    33: "plain-rice",
    34: "watermelon",
    35: "Aloo-Fry",
    36: "Banana",
    37: "Mix-Fruit",
    38: "Non-Spicy-Baby-Corn & Capsicum-Dry",
    39: "Sweet",
    40: "Tomato-Rice",
    41: "fried-papad-rings",
    42: "gravy",
    43: "ivy-gourd-fry",
    44: "mango-pickle",
    45: "papad-chat",
    46: "pepper-rasam",
    47: "pineapple",
    48: "corn-fry",
    49: "paneer-curry",
    50: "semiya" }
NUM_CLASSES = len(idx2class)

# -----------------------------------------------------------------------------
# Globals for worker processes
global_min_area = None
global_iou_thr = None
global_pooling = None


def init_worker(min_area: int, iou_thr: float, pooling: str):
    global global_min_area, global_iou_thr, global_pooling
    global_min_area = min_area
    global_iou_thr = iou_thr
    global_pooling = pooling


def clean_small_blobs(id_map: np.ndarray, min_area: int) -> np.ndarray:
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


def process_pair(args):
    gt_path, pr_path = args
    # load index maps
    gt_ids = np.array(
        Image.open(gt_path)
             .resize((1024, 1024), Image.Resampling.NEAREST),
        dtype=np.int32
    )
    pr_img = Image.open(pr_path).resize((1024, 1024), Image.Resampling.NEAREST)
    pr_ids = np.array(pr_img, dtype=np.int32)
    # clean small blobs
    # pr_ids = clean_small_blobs(pr_ids, global_min_area)

    # extract blobs
    gt_blobs = extract_blobs(gt_ids, global_min_area)
    pr_blobs = extract_blobs(pr_ids, global_min_area)

    # class-level matching
    conf = Counter()
    matched = set()
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
            max_ov, best_cls = 0, 0
            for g in gt_blobs:
                ov = np.logical_and(p['mask'], g['mask']).sum()
                if ov > max_ov:
                    max_ov, best_cls = ov, g['class']
            conf[(best_cls, p['class'])] += 1
    for idx, g in enumerate(gt_blobs):
        if idx not in matched:
            conf[(g['class'], 0)] += 1

    # pixel-level histogram
    mask = (gt_ids >= 0) & (gt_ids < NUM_CLASSES)
    hist = np.bincount(
        NUM_CLASSES * gt_ids[mask].ravel() + pr_ids[mask].ravel(),
        minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    return conf, hist


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic segmentation outputs with parallel file loading"
    )
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Directory containing ground‐truth index maps")
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory containing predicted index maps (same names as GT)")
    parser.add_argument("--output_dir", default=Path('output'), type=Path,
                        help="Directory to save metrics and heatmaps")
    parser.add_argument("--min_area", type=int, default=2200,
                        help="Min blob area to keep (in pixels)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for matching blobs")
    parser.add_argument("--cmap", default='OrRd', type=str,
                        help="Matplotlib colormap for confusion‐heatmap")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="Number of parallel worker processes")
    parser.add_argument("--pooling", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # collect file pairs
    gt_files = sorted([f for f in args.gt_dir.iterdir() if f.suffix in ['.png', '.jpg']])
    pred_files = sorted([f for f in args.pred_dir.iterdir() if f.suffix in ['.png', '.jpg']])
    assert len(gt_files) == len(pred_files), "GT and Pred directories must contain same number of files"

    tasks = [(str(gt), str(args.pred_dir / str('_'.join(gt.name.split('_')[:-2])+ "_leftImg8bit.png"))) for gt in gt_files]
    gt = [f for f in gt_files]
    # print(str('_'.join(gt[0].name.split('_')[:-2])+ "_leftImg8bit.png"))

    total_conf = Counter()
    total_hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.min_area, args.iou_threshold, args.pooling)
    ) as pool:
        for result in tqdm(pool.imap_unordered(process_pair, tasks),
                           total=len(tasks), desc="Processing masks"):
            if result is None:
                continue
            conf, hist = result
            total_conf.update(conf)
            total_hist += hist

    logging.info("Finished processing all mask pairs.")

    # (rest of aggregation and saving metrics unchanged)
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
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        # 'total_hist_sum': total_hist.sum(),
        # "correct": np.trace(total_hist),
    }
    
    print("Overall Metrics:")
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")
        
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
