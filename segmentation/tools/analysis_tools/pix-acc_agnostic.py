import os
import argparse
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from prettytable import PrettyTable


def load_mask(path):
    return np.array(Image.open(path).resize((1024, 1024), Image.Resampling.NEAREST), dtype=np.int32)


def class_agnostic_pixel_accuracy(gt_mask, pred_mask, background_index=0):
    assert gt_mask.shape == pred_mask.shape, f"Shape mismatch: {gt_mask.shape} vs {pred_mask.shape}"

    gt_fg = gt_mask != background_index
    pred_fg = pred_mask != background_index

    matched = gt_fg & pred_fg

    total_fg = gt_fg.sum()
    matched_fg = matched.sum()

    if total_fg == 0:
        return 1.0, 0, 0  # no foreground, perfect match

    accuracy = matched_fg / total_fg
    return accuracy, total_fg, matched_fg


def process_pair(args):
    gt_path, pred_path, background_index = args
    gt_mask = load_mask(gt_path)
    pred_mask = load_mask(pred_path)
    acc, total_fg, matched_fg = class_agnostic_pixel_accuracy(gt_mask, pred_mask, background_index)
    return acc, total_fg, matched_fg, os.path.basename(gt_path)


def main(args):
    gt_files = sorted([f for f in os.listdir(args.gt_dir) if f.endswith((".png", ".jpg"))])
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith((".png", ".jpg"))])

    assert len(gt_files) == len(pred_files), "Mismatch in number of files."

    task_args = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(args.gt_dir, gt_file)
        pred_path = os.path.join(args.pred_dir, pred_file)
        task_args.append((gt_path, pred_path, args.background_index))

    total_fg_pixels = 0
    total_matched_pixels = 0

    table = PrettyTable()
    table.field_names = ["Image", "Accuracy (%)", "FG Pixels", "Matched"]

    with Pool(args.workers or cpu_count()) as pool:
        for acc, total, matched, filename in tqdm(pool.imap_unordered(process_pair, task_args), total=len(task_args), desc="Evaluating"):
            total_fg_pixels += total
            total_matched_pixels += matched
            if args.per_image:
                if total == 0:
                    acc_str = "N/A"
                else:
                    acc_str = f"{acc * 100:.2f}"
                table.add_row([filename, acc_str, total, matched])

    if args.per_image:
        print("\n📋 Per-Image Class-Agnostic Pixel Accuracy:")
        print(table)

    if total_fg_pixels > 0:
        overall_acc = total_matched_pixels / total_fg_pixels
        print(f"\n✅ Overall Class-Agnostic Pixel Accuracy: {overall_acc * 100:.2f}% ({total_matched_pixels}/{total_fg_pixels})")
    else:
        print("\n⚠️ No foreground pixels found in ground truth.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Class-Agnostic Pixel Accuracy")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory with ground truth masks")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory with predicted masks")
    parser.add_argument("--background-index", type=int, default=0, help="Class index to consider as background")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--class-map", type=str, help="JSON file mapping index to class name")

    parser.add_argument("--per-image", action="store_true", help="Show per-image results")
    args = parser.parse_args()

    main(args)
