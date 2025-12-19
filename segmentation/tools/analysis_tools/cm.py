#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report
import json


def load_label_map(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_mask(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def build_confusion_and_lists(gt_dir, pred_dir, num_classes, ignore_index=None):
    preds = sorted([f for f in os.listdir(pred_dir)
                    if f.endswith('_leftImg8bit.png')])
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_gt = []
    all_pd = []

    for pred_name in preds:
        prefix = pred_name[:-len('_leftImg8bit.png')]
        gt_name = prefix + '_gtFine_labelIds.png'
        gt_path = os.path.join(gt_dir, gt_name)
        pred_path = os.path.join(pred_dir, pred_name)

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT file not found: {gt_path}")

        gt = load_mask(gt_path).flatten()
        pd = load_mask(pred_path).flatten()

        if ignore_index is not None:
            mask = (gt != ignore_index)
            gt = gt[mask]
            pd = pd[mask]

        pd = np.clip(pd, 0, num_classes - 1)
        
        all_gt.append(gt)
        all_pd.append(pd)
        
        print(f"Max GT value before inds calculation: {gt.max() if gt.size > 0 else 'N/A'}")
        print(f"Max PD value after clip before inds calculation: {pd.max() if pd.size > 0 else 'N/A'}")
        print(f"Num classes: {num_classes}")

        inds = num_classes * gt + pd
        cm += np.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)

    all_gt = np.concatenate(all_gt)
    all_pd = np.concatenate(all_pd)
    return cm, all_gt, all_pd


def plot_beautiful(cm, labels, out_path=None, show=True, color_theme='OrRd'):
    # Row-normalize to percentage
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(
        cm.astype(np.float32),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float32),
        where=row_sums > 0
    ) * 100

    # Identify size
    n = len(labels)
    cell_size = 0.7  # inches per cell
    fig_size = (max(8, n * cell_size), max(8, n * cell_size))
    fig, ax = plt.subplots(figsize=fig_size, dpi=200)

    # heatmap without colorbar
    im = ax.imshow(pct, vmin=0, vmax=100, cmap=color_theme,
                   aspect='equal', interpolation='nearest')

    # grid lines
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    # ticks and labels: placed at bottom, rotated 90 deg for clarity
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, ha='center', va='top', fontsize=10)
    ax.set_yticklabels(labels, fontsize=12)

    # adjust margins to accommodate rotated labels
    plt.subplots_adjust(left=0.25, bottom=0.3, top=0.9, right=0.85)

    ax.set_xlabel('Predicted Label', fontsize=16, labelpad=20)
    ax.set_ylabel('Ground Truth Label', fontsize=16, labelpad=20)
    ax.set_title('Confusion Matrix (%)', fontsize=18, weight='bold', pad=24)

    # annotations
    for i in range(n):
        for j in range(n):
            value = pct[i, j]
            text_color = 'white' if value > 60 else 'black'
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                    color=text_color, fontsize=10)

    # Invert the y-axis so that the first row is at the top
    # ax.invert_yaxis()
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()



def main():
    parser = argparse.ArgumentParser(
        description='Compute & beautify confusion matrix for segmentation')
    parser.add_argument('--gt-dir', required=True, help='Directory of GT masks')
    parser.add_argument('--pred-dir', required=True, help='Directory of pred masks')
    parser.add_argument('--label-map', default='/home2/yasharora120/segmentation_benchmark/mmsegmentation/tools/analysis_tools/label_new.txt', help='Text file of labels')
    parser.add_argument('--ignore-index', type=int, default=None, help='Index to ignore')
    parser.add_argument('--out-dir', default=None, help='Path to save heatmap PNG')
    parser.add_argument('--color-theme', default='OrRd', help='Matplotlib colormap name')
    args = parser.parse_args()

    # make sure output directory exists
    if args.out_dir and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    labels = load_label_map(args.label_map)
    cm_full, all_gt, all_pd = build_confusion_and_lists(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        num_classes=len(labels),
        ignore_index=args.ignore_index
    )

    rowsum = cm_full.sum(axis=1)
    colsum = cm_full.sum(axis=0)
    keep = (rowsum + colsum) > 0
    filtered_cm = cm_full[keep][:, keep]
    filtered_labels = [lbl for lbl, k in zip(labels, keep) if k]

    # print('Filtered classes:')
    # print(filtered_labels)
    # print('Filtered confusion matrix:')
    # print(filtered_cm)

    # Classification report for kept classes
    gt_filtered = all_gt[np.isin(all_gt, np.where(keep)[0])]
    pd_filtered = all_pd[np.isin(all_gt, np.where(keep)[0])]
    label_indices = np.where(keep)[0]
    print('\nClassification Report:')
    report = classification_report(
        gt_filtered, pd_filtered,
        labels=label_indices,
        target_names=filtered_labels,
        zero_division=0,
        output_dict=True
    )
    if args.out_dir:
        out_json = os.path.join(args.out_dir, 'classification_report.json')
        with open(out_json, 'w') as f:
            json.dump(report, f, indent=2)
    # print(classification_report(
    #     gt_filtered, pd_filtered,
    #     labels=label_indices,
    #     target_names=filtered_labels,
    #     zero_division=0
    # ))
    print(report)
    # Save the confusion matrix as a PNG
    out_plot = os.path.join(args.out_dir, 'confusion_matrix.png') if args.out_dir else None
    plot_beautiful(filtered_cm, filtered_labels, out_path=out_plot, color_theme=args.color_theme)
    print(f'Confusion matrix saved to: {out_plot}' if out_plot else 'Confusion matrix not saved, no output directory specified.')
    print('Done!')
if __name__ == '__main__':
    main()