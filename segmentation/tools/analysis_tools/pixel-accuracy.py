import os
import json
import argparse
from PIL import Image
import numpy as np
from prettytable import PrettyTable
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def load_mask(path):
    return np.array(Image.open(path).resize((1024, 1024), Image.Resampling.NEAREST), dtype=np.int32)


def calculate_per_image(args):
    gt_path, pred_path, num_classes, ignore_index = args
    gt = load_mask(gt_path)
    pred = load_mask(pred_path)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt_path} and {pred_path}")

    valid_mask = (gt != ignore_index) if ignore_index is not None else np.ones_like(gt, dtype=bool)
    gt = gt[valid_mask]
    pred = pred[valid_mask]

    correct = np.zeros(num_classes, dtype=np.uint64)
    total = np.zeros(num_classes, dtype=np.uint64)

    for cls in range(num_classes):
        cls_mask = (gt == cls)
        total[cls] = cls_mask.sum()
        correct[cls] = (pred[cls_mask] == cls).sum()

    return correct, total


def main(args):
    with open(args.class_map, 'r') as f:
        index_to_class = json.load(f)
    num_classes = len(index_to_class)

    gt_files = sorted([f for f in os.listdir(args.gt_dir) if f.endswith((".png", ".jpg"))])
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith((".png", ".jpg"))])

    assert len(gt_files) == len(pred_files), "Mismatch in number of ground truth and prediction files."

    tasks = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(args.gt_dir, gt_file)
        pred_path = os.path.join(args.pred_dir, pred_file)
        tasks.append((gt_path, pred_path, num_classes, args.ignore_index))

    total_correct = np.zeros(num_classes, dtype=np.uint64)
    total_pixels = np.zeros(num_classes, dtype=np.uint64)

    with Pool(processes=args.workers or cpu_count()) as pool:
        for correct, total in tqdm(pool.imap_unordered(calculate_per_image, tasks), total=len(tasks), desc="Evaluating"):
            total_correct += correct
            total_pixels += total

    # Generate PrettyTable
    table = PrettyTable()
    table.field_names = ["Class Index", "Class Name", "Accuracy (%)", "Pixels"]

    total_accuracy = 0
    valid_classes = 0

    for idx in range(num_classes):
        class_name = index_to_class.get(str(idx), f"Class {idx}")
        pixels = total_pixels[idx]
        if pixels == 0:
            acc = "N/A"
        else:
            acc_val = (total_correct[idx] / pixels) * 100
            acc = f"{acc_val:.2f}"
            total_accuracy += acc_val
            valid_classes += 1

        table.add_row([idx, class_name, acc, pixels])

    print(table)
    
    print("simple total pixels:", total_pixels.sum())
    print("simple correct pixels:", total_correct.sum())


    if valid_classes > 0:
        mpa = total_accuracy / valid_classes
        print(f"\n➡️  Overall Pixel Accuracy: {(total_correct.sum() / total_pixels.sum()) * 100:.2f}%")
        print(f"➡️  Mean Per-Class Accuracy (MPA): {mpa:.2f}%")
    else:
        print("No valid classes found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Pixel Accuracy Evaluation")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory with ground truth masks")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory with predicted masks")
    parser.add_argument("--class-map", type=str, required=True, help="JSON file mapping index to class name")
    parser.add_argument("--ignore-index", type=int, default=0, help="Optional ignore index (e.g., 255)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    main(args)
