#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


def load_label_map(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_mask(path):
    # Load only the first channel and downcast to uint16 to save memory
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint16)


# def build_confusion_matrix(gt_dir, pred_dir, num_classes, ignore_index=None):
#     """
#     Streams through each mask pair and accumulates the confusion matrix without
#     storing all pixel vectors.
#     """
#     file_list = sorted([f for f in os.listdir(pred_dir)
#                         if f.endswith('_leftImg8bit.png')])
#     cm = np.zeros((num_classes, num_classes), dtype=np.int64)

#     for fname in tqdm(file_list, desc="Building CM", unit="image"):
#         prefix = fname[:-len('_leftImg8bit.png')]
#         gt_path = os.path.join(gt_dir, prefix + '_gtFine_labelIds.png')
#         pred_path = os.path.join(pred_dir, fname)

#         if not os.path.exists(gt_path):
#             raise FileNotFoundError(f"GT file not found: {gt_path}")

#         gt = load_mask(gt_path).ravel()
#         pd = load_mask(pred_path).ravel()

#         if ignore_index is not None:
#             mask = (gt != ignore_index)
#             gt = gt[mask]
#             pd = pd[mask]

#         pd = np.clip(pd, 0, num_classes - 1)
#         inds = gt * num_classes + pd
#         cm += np.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)

#         # free memory
#         del gt, pd, inds

#     return cm


def build_confusion_matrix(gt_dir, pred_dir, num_classes, ignore_index=None):
    """
    Streams through each mask pair and accumulates the confusion matrix without
    storing all pixel vectors.
    """
    file_list = sorted([f for f in os.listdir(pred_dir)
                        if f.endswith('_leftImg8bit.png')])
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fname in tqdm(file_list, desc="Building CM", unit="image"):
        prefix = fname[:-len('_leftImg8bit.png')]
        gt_path = os.path.join(gt_dir, prefix + '_gtFine_labelIds.png')
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT file not found: {gt_path}")

        gt = load_mask(gt_path).ravel()
        pd = load_mask(pred_path).ravel()

        if ignore_index is not None:
            mask = (gt != ignore_index)
            gt = gt[mask]
            pd = pd[mask]

        # The np.clip line has been removed as per your confirmation.
        inds = gt * num_classes + pd
        cm += np.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)

        # free memory
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
            'support':   int(support[i]),
        }

    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    macro_f1 = np.mean(f1)
    total_support = int(support.sum())
    report['macro avg'] = {
        'precision': float(macro_p),
        'recall':    float(macro_r),
        'f1-score':  float(macro_f1),
        'support':   total_support
    }

    weights = support / support.sum() if total_support > 0 else np.zeros_like(support)
    weighted_p = float((precision * weights).sum())
    weighted_r = float((recall * weights).sum())
    weighted_f1 = float((f1 * weights).sum())
    report['weighted avg'] = {
        'precision': weighted_p,
        'recall':    weighted_r,
        'f1-score':  weighted_f1,
        'support':   total_support
    }

    return report


def plot_beautiful(cm, labels, out_path=None, show=True, color_theme='OrRd', run_name=''):
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(
        cm.astype(np.float32),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float32),
        where=row_sums > 0
    ) * 100

    n = len(labels)
    cell_size = 0.7
    fig_size = (max(8, n * cell_size), max(8, n * cell_size))
    fig, ax = plt.subplots(figsize=fig_size, dpi=200)

    im = ax.imshow(pct, vmin=0, vmax=100, cmap=color_theme,
                   aspect='equal', interpolation='nearest')

    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, ha='center', va='top', fontsize=10)
    ax.set_yticklabels(labels, fontsize=12)

    plt.subplots_adjust(left=0.25, bottom=0.3, top=0.9, right=0.85)
    ax.set_xlabel('Predicted Label', fontsize=16, labelpad=20)
    ax.set_ylabel('Ground Truth Label', fontsize=16, labelpad=20)
    ax.set_title(f'Confusion Matrix (%) {run_name}', fontsize=18, weight='bold', pad=24)

    for i in range(n):
        for j in range(n):
            val = pct[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=10)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Streamlined confusion matrix + report generator')
    parser.add_argument('--gt-dir', required=True, help='Directory of GT masks')
    parser.add_argument('--pred-dir', required=True, help='Directory of pred masks')
    parser.add_argument('--label-map', default='label_new_full.txt',
                        help='Text file of labels (one per line)')
    parser.add_argument('--ignore-index', type=int, default=None,
                        help='Class index to ignore')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save outputs')
    parser.add_argument('--color-theme', default='OrRd',
                        help='Matplotlib colormap name')
    parser.add_argument("--run-name", required=True,
                        help="Name of the run for the confusion matrix title")
    args = parser.parse_args()

    if args.out_dir and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    labels = load_label_map(args.label_map)
    num_classes = len(labels)

    cm_full = build_confusion_matrix(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        num_classes=num_classes,
        ignore_index=args.ignore_index
    )

    row_sum = cm_full.sum(axis=1)
    col_sum = cm_full.sum(axis=0)
    keep = (row_sum + col_sum) > 0

    filtered_cm = cm_full[keep][:, keep]
    filtered_labels = [lbl for lbl, k in zip(labels, keep) if k]

    report = metrics_from_confusion(filtered_cm, filtered_labels)

    if args.out_dir:
        json_path = os.path.join(args.out_dir, 'classification_report.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved classification report to {json_path}")
    else:
        print(json.dumps(report, indent=2))

    out_png = os.path.join(args.out_dir, 'confusion_matrix.png') if args.out_dir else None
    plot_beautiful(filtered_cm, filtered_labels,
                   out_path=out_png, color_theme=args.color_theme, run_name=args.run_name)
    if out_png:
        print(f"Saved confusion matrix heatmap to {out_png}")


if __name__ == '__main__':
    main()
