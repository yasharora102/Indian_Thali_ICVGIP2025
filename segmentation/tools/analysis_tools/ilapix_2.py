import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
import os
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
target_size = (1024, 1024)  # (width, height) for final visualization
# ──────────────────────────────────────────────────────────────────────────────

def get_random_colormap(num_classes=256, seed=42):
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 255, size=(num_classes, 3)) / 255.0
    colors[0] = 0.0  # background = black
    return colors

def extract_blobs(id_map: np.ndarray, min_area: int = 1):
    blobs = []
    lbl = label(id_map > 0)
    for region in regionprops(lbl):
        if region.area < min_area:
            continue
        r0, c0 = region.coords[0]
        cls = int(id_map[r0, c0])
        if cls == 0:
            continue
        mask = np.zeros_like(id_map, dtype=bool)
        mask[tuple(region.coords.T)] = True
        blobs.append({
            'mask': mask,
            'class': cls,
            'bbox': region.bbox
        })
    return blobs

def calculate_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    uni   = np.logical_or(m1, m2).sum()
    return (inter / uni) if uni else 0.0

def apply_colormap(label_img: np.ndarray, colormap: np.ndarray):
    h, w = label_img.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for lbl in np.unique(label_img):
        if lbl == 0:
            continue
        out[label_img == lbl] = colormap[lbl % len(colormap)]
    return out

def overlay_on_image(rgb: np.ndarray, label_img: np.ndarray, colormap: np.ndarray, alpha: float = 0.45):
    color_mask = apply_colormap(label_img, colormap)
    overlay = (1 - alpha) * (rgb.astype(np.float32) / 255.0) + alpha * color_mask
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

def resize_array(arr, size=target_size, is_label=False):
    img = Image.fromarray(arr.astype(np.uint8))
    resample_mode = Image.NEAREST if is_label else Image.BILINEAR
    img = img.resize(size, resample=resample_mode)
    return np.array(img)

# ──────────────────────────────────────────────────────────────────────────────
def visualize_all_side_by_side(
    gt_path, pred_path, out_path, image_path=None,
    iou_threshold=0.5, min_blob_area=1, title=None
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    gt_ids = np.array(Image.open(gt_path), dtype=np.int32)
    pr_ids = np.array(Image.open(pred_path), dtype=np.int32)

    # Resize masks to 1024x1024
    gt_ids = resize_array(gt_ids, target_size, is_label=True).astype(np.int32)
    pr_ids = resize_array(pr_ids, target_size, is_label=True).astype(np.int32)

    correct = (gt_ids == pr_ids).astype(np.uint8)
    pixel_acc = correct.mean()

    gt_blobs = extract_blobs(gt_ids, min_area=min_blob_area)
    pr_blobs = extract_blobs(pr_ids, min_area=min_blob_area)

    matches = {}
    correct_cnt = 0
    for i, g in enumerate(gt_blobs):
        best_iou, best_j = 0.0, None
        for j, p in enumerate(pr_blobs):
            iou = calculate_iou(g['mask'], p['mask'])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold:
            matches[i] = (best_j, best_iou)
            if pr_blobs[best_j]['class'] == g['class']:
                correct_cnt += 1
    inst_acc = (correct_cnt / len(gt_blobs)) if gt_blobs else 0.0

    ila_mask = np.zeros_like(gt_ids, dtype=np.uint8)
    for i in matches:
        ila_mask[gt_blobs[i]['mask']] = 1

    colormap = get_random_colormap(max(int(gt_ids.max()), int(pr_ids.max())) + 1)

    gt_colored   = apply_colormap(gt_ids, colormap)
    pred_colored = apply_colormap(pr_ids, colormap)

    base_rgb = None
    if image_path and os.path.isfile(image_path):
        base_rgb = np.array(Image.open(image_path).convert('RGB'))
        base_rgb = resize_array(base_rgb, target_size, is_label=False)

    gt_overlaid   = overlay_on_image(base_rgb, gt_ids, colormap) if base_rgb is not None else (gt_colored * 255).astype(np.uint8)
    pred_overlaid = overlay_on_image(base_rgb, pr_ids, colormap) if base_rgb is not None else (pred_colored * 255).astype(np.uint8)

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    axs[0].imshow(gt_overlaid)
    axs[0].set_title("GT overlaid on Image")
    axs[0].axis('off')

    axs[1].imshow(pred_overlaid)
    axs[1].set_title("Pred overlaid on Image")
    axs[1].axis('off')

    axs[2].imshow(gt_colored)
    axs[2].imshow(correct, cmap='Greens', alpha=0.45)
    axs[2].set_title(f"OPA (Correct Pixels)\nPixel Acc: {pixel_acc*100:.2f}%")
    axs[2].axis('off')

    axs[3].imshow(gt_colored)
    axs[3].imshow(ila_mask, cmap='Blues', alpha=0.45)
    axs[3].set_title(f"ILA (Matched Instances)\nInst Acc: {inst_acc*100:.2f}%")
    axs[3].axis('off')

    for i, (j, iou) in matches.items():
        g = gt_blobs[i]
        p = pr_blobs[j]
        minr, minc, maxr, maxc = g['bbox']
        axs[3].add_patch(plt.Rectangle((minc, minr), maxc-minc, maxr-minr,
                                       edgecolor='cyan', facecolor='none', linewidth=1.5))
        axs[3].text(minc, minr, f"GT:{g['class']} PR:{p['class']}\nIoU:{iou:.2f}",
                    color='yellow', fontsize=8, va='top',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    gt_suffix   = '_gtFine_labelIds.png'
    pred_suffix = '_leftImg8bit.png'
    img_suffix  = '_leftImg8bit.jpg'

    gt_dir    = '/scratch/seg_benchmark/splits_flat/test/masks/'
    pred_dir  = '/scratch/seg_benchmark/copied/results_160K/'
    # pred_dir  = '/scratch/seg_benchmark/NEW/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/results_160K'
    image_dir = '/scratch/seg_benchmark/splits_flat/test/images/'
    out_dir   = 'overlap_maskswin_visualization2'
    os.makedirs(out_dir, exist_ok=True)

    for file in tqdm(sorted(os.listdir(gt_dir)), desc="Processing files"):
        if not file.endswith(gt_suffix):
            continue
        stem = file[:-len(gt_suffix)]
        gt_path   = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, stem + pred_suffix)
        img_path  = os.path.join(image_dir, stem + img_suffix)
        if not os.path.isfile(pred_path):
            continue
        if not os.path.isfile(img_path):
            img_path = None
        out_path = os.path.join(out_dir, f'opa_ila_overlay_{stem}.png')
        visualize_all_side_by_side(
            gt_path, pred_path, out_path, image_path=img_path,
            iou_threshold=0.5, title=f"Overlay for {stem}"
        )
