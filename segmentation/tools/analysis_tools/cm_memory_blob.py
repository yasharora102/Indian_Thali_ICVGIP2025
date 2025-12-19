#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from collections import Counter
from skimage.measure import label, regionprops

# No ColorMapper needed: masks are already index-based

def load_label_map(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_mask(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        # assume first channel holds class index
        arr = arr[..., 0]
    return arr.astype(np.int32)


def build_confusion_matrix(gt_dir, pred_dir, num_classes, ignore_index=None):
    # only consider predictions matching *_leftImg8bit.png
    preds = sorted([f for f in os.listdir(pred_dir) if f.endswith('_leftImg8bit.png')])
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for f in tqdm(preds, desc='Pixel CM', unit='img'):
        prefix = f[:-len('_leftImg8bit.png')]
        gt_path = os.path.join(gt_dir, prefix + '_gtFine_labelIds.png')
        pred_path = os.path.join(pred_dir, f)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT not found: {gt_path}")

        gt = load_mask(gt_path).ravel()
        pd = load_mask(pred_path).ravel()
        if ignore_index is not None:
            mask = gt != ignore_index
            gt, pd = gt[mask], pd[mask]
        pd = np.clip(pd, 0, num_classes - 1)
        inds = gt * num_classes + pd
        cm += np.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)
        del gt, pd, inds
    return cm


def metrics_from_confusion(cm, labels):
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum>0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support>0)
    f1 = np.divide(2 * precision * recall,
                   (precision + recall),
                   out=np.zeros_like(tp),
                   where=(precision + recall)>0)
    report = {}
    for i, label in enumerate(labels):
        report[label] = {
            'precision': float(precision[i]),
            'recall':    float(recall[i]),
            'f1-score':  float(f1[i]),
            'support':   int(support[i])
        }
    return report


def clean_small_blobs(id_map, min_area):
    out = id_map.copy()
    for cls in np.unique(id_map):
        if cls == 0:
            continue
        lbl = label(id_map == cls)
        for reg in regionprops(lbl):
            if reg.area < min_area:
                out[tuple(reg.coords.T)] = 0
    return out


def extract_blobs(id_map, min_area):
    blobs = []
    lbl_img = label(id_map)
    for reg in regionprops(lbl_img):
        cls = id_map[tuple(reg.coords[0])]
        if cls == 0 or reg.area < min_area:
            continue
        mask = np.zeros_like(id_map, dtype=bool)
        mask[tuple(reg.coords.T)] = True
        blobs.append({'mask': mask, 'class': int(cls), 'area': reg.area})
    return blobs


def calculate_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    uni   = np.logical_or(m1, m2).sum()
    return inter / uni if uni else 0.0


def build_blob_confusion(gt_dir, pred_dir, num_classes, min_area, iou_thr):
    preds = sorted([f for f in os.listdir(pred_dir) if f.endswith('_leftImg8bit.png')])
    conf = Counter()
    for f in tqdm(preds, desc='Blob CM', unit='img'):
        prefix = f[:-len('_leftImg8bit.png')]
        gt_path = os.path.join(gt_dir, prefix + '_gtFine_labelIds.png')
        pred_path = os.path.join(pred_dir, f)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT not found: {gt_path}")

        gt_ids = load_mask(gt_path)
        pr_ids = load_mask(pred_path)
        pr_ids = clean_small_blobs(pr_ids, min_area)

        gt_blobs = extract_blobs(gt_ids, min_area)
        pr_blobs = extract_blobs(pr_ids, min_area)
        matched = set()

        for p in pr_blobs:
            best_iou, best_idx = 0.0, -1
            for idx, g in enumerate(gt_blobs):
                iou = calculate_iou(p['mask'], g['mask'])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= iou_thr:
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

    return conf


def plot_beautiful(cm, labels, out_path=None, show=True, color_theme='OrRd', normalize=False, run_name='cm_memory_blob'):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n*0.7), max(8, n*0.7)), dpi=200)
    if normalize:
        # Row-normalize to percentage
        row_sums = cm.sum(axis=1, keepdims=True)
        pct = np.divide(
            cm.astype(np.float32),
            row_sums,
            out=np.zeros_like(cm, dtype=np.float32),
            where=row_sums > 0
        ) * 100
        im = ax.imshow(pct, vmin=0, vmax=100, cmap=color_theme, aspect='equal')
        ax.set_title(f'{run_name} (Normalized %)', fontsize=18)
    else:
        im = ax.imshow(cm, cmap=color_theme, aspect='equal')
        ax.set_title(f'{run_name} (Raw Counts)', fontsize=18)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticklabels(labels, fontsize=12)
    plt.subplots_adjust(left=0.25, bottom=0.3, right=0.85, top=0.9)
    ax.set_xlabel('Predicted', fontsize=16)
    ax.set_ylabel('Ground Truth', fontsize=16)
    
    # Annotate each cell
    for i in range(n):
        for j in range(n):
            if normalize:
                # display percentages with one decimal
                cell_text = f'{pct[i, j]:.1f}%'
                threshold = 60  # percentage threshold for text color decision
                val = pct[i, j]
            else:
                cell_text = f'{int(cm[i, j])}'
                threshold = cm.max() * 0.6
                val = cm[i, j]
            c = 'white' if val > threshold else 'black'
            ax.text(j, i, cell_text, ha='center', va='center', color=c, fontsize=10)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', required=True, help='Directory of GT masks')
    parser.add_argument('--pred-dir', required=True, help='Directory of pred masks')
    parser.add_argument('--label-map', default='label_new.txt', help='Text file of labels, one per line')
    parser.add_argument('--ignore-index', type=int, default=None, help='Index to ignore')
    parser.add_argument('--min-area', type=int, default=2200, help='Min blob area to keep')
    parser.add_argument('--iou-threshold', type=float, default=0.6, help='IoU threshold for blob matching')
    parser.add_argument('--out-dir', default=None, help='Directory to save outputs')
    parser.add_argument('--color-theme', default='OrRd', help='Matplotlib colormap name')
    parser.add_argument('--run-name', default='cm_memory_blob', help='Name of the run for logging')
    args = parser.parse_args()

    if args.out_dir and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Pixel-level metrics
    labels = load_label_map(args.label_map)
    num_classes = len(labels)
    cm_pix = build_confusion_matrix(args.gt_dir, args.pred_dir, num_classes, args.ignore_index)
    keep = (cm_pix.sum(axis=1) + cm_pix.sum(axis=0)) > 0
    cm_pix_f = cm_pix[keep][:, keep]
    labels_pix = [lbl for lbl, k in zip(labels, keep) if k]
    report_pix = metrics_from_confusion(cm_pix_f, labels_pix)
    with open(os.path.join(args.out_dir, 'pixel_report.json'), 'w') as f:
        json.dump(report_pix, f, indent=2)
    plot_beautiful(cm_pix_f, labels_pix,
                   out_path=os.path.join(args.out_dir, 'pixel_cm.png'),
                   show=False,
                   color_theme=args.color_theme,
                   normalize=True,
                   run_name=args.run_name)

    # Blob-level metrics
    conf_blob = build_blob_confusion(args.gt_dir, args.pred_dir,
                                    num_classes, args.min_area, args.iou_threshold)
    cm_blob = np.zeros((num_classes, num_classes), dtype=int)
    for (gt, pr), cnt in conf_blob.items():
        cm_blob[gt, pr] = cnt
    keep_b = (cm_blob.sum(axis=1) + cm_blob.sum(axis=0)) > 0
    cm_blob_f = cm_blob[keep_b][:, keep_b]
    labels_blob = [labels[i] for i, k in enumerate(keep_b) if k]
    report_blob = {f"{gt}->{pr}": cnt for (gt, pr), cnt in conf_blob.items()}
    with open(os.path.join(args.out_dir, 'blob_report.json'), 'w') as f:
        json.dump(report_blob, f, indent=2)
    plot_beautiful(cm_blob_f, labels_blob,
                   out_path=os.path.join(args.out_dir, 'blob_cm.png'),
                   show=False,
                   color_theme=args.color_theme,
                   normalize=False,
                   run_name=args.run_name)

if __name__ == '__main__':
    main()
