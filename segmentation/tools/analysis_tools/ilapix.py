# # import numpy as np
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # from skimage.measure import label, regionprops
# # import os

# # def compute_instance_matches(gt_mask, pred_mask, iou_threshold=0.5):
# #     gt_labeled = label(gt_mask)
# #     pred_labeled = label(pred_mask)

# #     gt_props = regionprops(gt_labeled)
# #     pred_props = regionprops(pred_labeled)

# #     matched_gt_labels = set()
# #     matched_pred_labels = set()

# #     for gt_region in gt_props:
# #         gt_label_val = gt_region.label
# #         gt_instance = (gt_labeled == gt_label_val)

# #         best_iou = 0
# #         best_pred_label = None

# #         for pred_region in pred_props:
# #             pred_label_val = pred_region.label
# #             pred_instance = (pred_labeled == pred_label_val)

# #             intersection = np.logical_and(gt_instance, pred_instance).sum()
# #             union = np.logical_or(gt_instance, pred_instance).sum()
# #             iou = intersection / union if union > 0 else 0

# #             if iou > best_iou:
# #                 best_iou = iou
# #                 best_pred_label = pred_label_val

# #         if best_iou >= iou_threshold:
# #             matched_gt_labels.add(gt_label_val)
# #             matched_pred_labels.add(best_pred_label)

# #     return gt_labeled, pred_labeled, matched_gt_labels, matched_pred_labels

# # def visualize_all_side_by_side(gt_path, pred_path, out_path, iou_threshold=0.5):
# #     os.makedirs(os.path.dirname(out_path), exist_ok=True)

# #     gt = np.array(Image.open(gt_path), dtype=np.int32)
# #     pred = np.array(Image.open(pred_path), dtype=np.int32)
    
# #     print(f"Unique GT labels: {np.unique(gt)}")
# #     print(f"Unique Pred labels: {np.unique(pred)}")

# #     # Compute OPA
# #     correct = (gt == pred).astype(np.uint8)

# #     # Compute ILA
# #     gt_labeled, pred_labeled, matched_gt_labels, _ = compute_instance_matches(gt, pred, iou_threshold)
# #     ila_mask = np.zeros_like(gt_labeled, dtype=np.uint8)
# #     for label_val in matched_gt_labels:
# #         ila_mask[gt_labeled == label_val] = 1

# #     # Plot
# #     fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# #     cmap = 'tab20'  # consistent categorical colormap

# #     axs[0].imshow(gt_labeled, cmap=cmap)
# #     axs[0].set_title("Ground Truth (Labeled)")
# #     axs[0].axis('off')

# #     axs[1].imshow(pred_labeled, cmap=cmap)
# #     axs[1].set_title("Prediction (Labeled)")
# #     axs[1].axis('off')

# #     axs[2].imshow(gt_labeled, cmap=cmap)
# #     axs[2].imshow(correct, cmap='Greens', alpha=0.5)
# #     axs[2].set_title("OPA (Correct Pixels)")
# #     axs[2].axis('off')

# #     axs[3].imshow(gt_labeled, cmap=cmap)
# #     axs[3].imshow(ila_mask, cmap='Blues', alpha=0.5)
# #     axs[3].set_title("ILA (Matched Instances)")
# #     axs[3].axis('off')

# #     plt.tight_layout()
# #     plt.savefig(out_path, dpi=150)
# #     plt.close()


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from skimage.measure import label, regionprops
# import os
# import random

# def get_random_colormap(num_classes=256, seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     colors = np.random.randint(0, 255, size=(num_classes, 3)) / 255.0
#     return colors

# def compute_instance_matches(gt_mask, pred_mask, iou_threshold=0.5):
#     gt_labeled = label(gt_mask)
#     pred_labeled = label(pred_mask)

#     gt_props = regionprops(gt_labeled)
#     pred_props = regionprops(pred_labeled)

#     matched_gt_labels = set()
#     matched_pred_labels = set()

#     for gt_region in gt_props:
#         gt_label_val = gt_region.label
#         gt_instance = (gt_labeled == gt_label_val)

#         best_iou = 0
#         best_pred_label = None

#         for pred_region in pred_props:
#             pred_label_val = pred_region.label
#             pred_instance = (pred_labeled == pred_label_val)

#             intersection = np.logical_and(gt_instance, pred_instance).sum()
#             union = np.logical_or(gt_instance, pred_instance).sum()
#             iou = intersection / union if union > 0 else 0

#             if iou > best_iou:
#                 best_iou = iou
#                 best_pred_label = pred_label_val

#         if best_iou >= iou_threshold:
#             matched_gt_labels.add(gt_label_val)
#             matched_pred_labels.add(best_pred_label)

#     return gt_labeled, pred_labeled, matched_gt_labels, matched_pred_labels, gt_props

# def visualize_all_side_by_side(gt_path, pred_path, out_path, iou_threshold=0.5):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     gt = np.array(Image.open(gt_path), dtype=np.int32)
#     pred = np.array(Image.open(pred_path), dtype=np.int32)

#     correct = (gt == pred).astype(np.uint8)
#     gt_labeled, pred_labeled, matched_gt_labels, _, gt_props = compute_instance_matches(gt, pred, iou_threshold)
    
#     ila_mask = np.zeros_like(gt_labeled, dtype=np.uint8)
#     for label_val in matched_gt_labels:
#         ila_mask[gt_labeled == label_val] = 1

#     # Generate fixed random colormap
#     colormap = get_random_colormap(51)
    
#     def apply_colormap(label_img):
#         color_img = np.zeros((*label_img.shape, 3), dtype=np.float32)
#         for lbl in np.unique(label_img):
#             if lbl == 0:
#                 continue
#             color_img[label_img == lbl] = colormap[lbl]
#         return color_img

#     # Apply colormap
#     gt_colored = apply_colormap(gt_labeled)
#     pred_colored = apply_colormap(pred_labeled)

#     # Plotting
#     fig, axs = plt.subplots(1, 4, figsize=(20, 5))

#     axs[0].imshow(gt_colored)
#     axs[0].set_title("Ground Truth Instances")
#     axs[0].axis('off')

#     axs[1].imshow(pred_colored)
#     axs[1].set_title("Predicted Instances")
#     axs[1].axis('off')

#     axs[2].imshow(gt_colored)
#     axs[2].imshow(correct, cmap='Greens', alpha=0.5)
#     axs[2].set_title("OPA (Correct Pixels)")
#     axs[2].axis('off')

#     axs[3].imshow(gt_colored)
#     axs[3].imshow(ila_mask, cmap='Blues', alpha=0.5)
#     axs[3].set_title("ILA (Matched Instances)")
#     axs[3].axis('off')

#     # Add bounding boxes to ILA
#     for prop in gt_props:
#         if prop.label in matched_gt_labels:
#             minr, minc, maxr, maxc = prop.bbox
#             axs[3].add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                            edgecolor='cyan', facecolor='none', linewidth=1.5))

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
import os
import random
from tqdm import tqdm
def get_random_colormap(num_classes=256, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    colors = np.random.randint(0, 255, size=(num_classes, 3)) / 255.0
    return colors

def extract_blobs(id_map: np.ndarray, min_area: int = 1):
    """
    Returns a list of blobs, each as dict with keys:
     - 'mask': boolean array of the blob
     - 'class': integer class label
     - 'bbox': (min_row, min_col, max_row, max_col)
    """
    blobs = []
    lbl = label(id_map)
    for region in regionprops(lbl):
        if region.area < min_area:
            continue
        cls = int(id_map[tuple(region.coords[0])])
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

def visualize_all_side_by_side(
    gt_path, pred_path, out_path, iou_threshold=0.5, min_blob_area=1,title=None
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    gt_ids   = np.array(Image.open(gt_path),  dtype=np.int32)
    pr_ids   = np.array(Image.open(pred_path),dtype=np.int32)
    correct  = (gt_ids == pr_ids).astype(np.uint8)

    # extract instance blobs
    gt_blobs = extract_blobs(gt_ids, min_area=min_blob_area)
    pr_blobs = extract_blobs(pr_ids, min_area=min_blob_area)

    # match each GT blob to best PR blob
    matches = {}      # gt_index -> (pred_index, iou)
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

    # build ILA mask (all GT blobs with IoU >= thr)
    ila_mask = np.zeros_like(gt_ids, dtype=np.uint8)
    for i in matches:
        ila_mask[gt_blobs[i]['mask']] = 1

    # colormap
    colormap = get_random_colormap(51)
    def apply_colormap(label_img):
        img = np.zeros((*label_img.shape, 3), dtype=np.float32)
        for lbl in np.unique(label_img):
            if lbl == 0: continue
            img[label_img==lbl] = colormap[lbl]
        return img

    gt_colored   = apply_colormap(gt_ids)
    pred_colored = apply_colormap(pr_ids)
    pixel_acc    = correct.sum() / correct.size

    # plot
    fig, axs = plt.subplots(1, 4, figsize=(20,5))

    axs[0].imshow(gt_colored)
    axs[0].set_title("Ground Truth Instances")
    axs[0].axis('off')

    axs[1].imshow(pred_colored)
    axs[1].set_title("Predicted Instances")
    axs[1].axis('off')

    axs[2].imshow(gt_colored)
    axs[2].imshow(correct, cmap='Greens', alpha=0.5)
    axs[2].set_title(f"OPA (Correct Pixels)\nPixel Acc: {pixel_acc*100:.2f}%")
    axs[2].axis('off')

    axs[3].imshow(gt_colored)
    axs[3].imshow(ila_mask, cmap='Blues', alpha=0.5)
    axs[3].set_title(f"ILA (Matched Instances)\nInst Acc: {inst_acc*100:.2f}%")
    axs[3].axis('off')

    # annotate ILA with GT label, PR label, and IoU
    for i, (j, iou) in matches.items():
        g = gt_blobs[i]
        p = pr_blobs[j]
        minr, minc, maxr, maxc = g['bbox']

        # draw box
        axs[3].add_patch(plt.Rectangle(
            (minc, minr),
            maxc-minc,
            maxr-minr,
            edgecolor='cyan',
            facecolor='none',
            linewidth=1.5
        ))
        # text
        axs[3].text(
            minc,
            minr,
            f"GT:{g['class']} PR:{p['class']}\nIoU:{iou:.2f}",
            color='yellow',
            fontsize=8,
            va='top',
            bbox=dict(facecolor='black', alpha=0.5, pad=1)
        )
    plt.suptitle(title, fontsize=16) if title else None
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

if __name__ == '__main__':
    
    
    
    gt_prefix = '_gtFine_labelIds.png'
    pred_prefix = '_leftImg8bit.png'
    
    gt_path = '/scratch/seg_benchmark/splits_flat/test/masks/'
    pred_path = '/scratch/seg_benchmark/NEW/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/results_160K'
    
    for file in tqdm(os.listdir(gt_path), desc="Processing files"):
        if file.endswith(gt_prefix):
            gt_file = file
            pred_file = file.replace(gt_prefix, pred_prefix)
            gt_full_path = os.path.join(gt_path, gt_file)
            pred_full_path = os.path.join(pred_path, pred_file)
            
            out_path = f'overlap_deeplab/opa_ila_overlay_{pred_file.split("_")[0]}.png'
            
            visualize_all_side_by_side(
                gt_path=gt_full_path,
                pred_path=pred_full_path,
                out_path=out_path,
                iou_threshold=0.5,
                title=f"Overlay for {pred_file.split('_')[0]}"
            )
    # visualize_all_side_by_side(
    #     gt_path='/scratch/seg_benchmark/splits_flat/test/masks/20250630_151349_gtFine_labelIds.png',
    #     pred_path='/scratch/seg_benchmark/NEW/mask2former_swin_T_seed_320K/mask2former_swin_T_seed_320K/results_40K/20250630_151349_leftImg8bit.png',
    #     out_path='overlay/opa_ila_overlay_mskrswin2.png',
    #     iou_threshold=0.5,
    #     title="Mask2Former-SwinT"
    # )
    
#     visualize_all_side_by_side(
#         gt_path='/scratch/seg_benchmark/splits_flat/test/masks/20250630_151349_gtFine_labelIds.png',
#         pred_path='/scratch/seg_benchmark/NEW/mask2former_R50_320K/results_40K/20250630_151349_leftImg8bit.png',
#         out_path='overlay/opa_ila_overlay_mskr50_2.png',
#         iou_threshold=0.5,
#         title="Mask2Former-R50"
#     )
    
#     visualize_all_side_by_side(
#         gt_path='/scratch/seg_benchmark/splits_flat/test/masks/20250630_151349_gtFine_labelIds.png',
#         pred_path='/scratch/seg_benchmark/NEW/FCN_R50-resized_320K/FCN_R50-resized_320K/results_40K/20250630_151349_leftImg8bit.png',
#         out_path='overlay/opa_ila_overlay_fcnr50_2.png',
#         iou_threshold=0.5,
#         title="FCN-R50"
#     )
    
#     visualize_all_side_by_side(
#         gt_path='/scratch/seg_benchmark/splits_flat/test/masks/20250630_151349_gtFine_labelIds.png',
#         pred_path='/scratch/seg_benchmark/NEW/segformer-mitb3-FUll-resized_seed_320K/segformer-mitb3-FUll-resized_seed_320K/results_40K/20250630_151349_leftImg8bit.png',
#         out_path='overlay/opa_ila_overlay_segformer_mitb3_2.png',
#         iou_threshold=0.5,
#         title="SegFormer-MITB3"
#     )
#     visualize_all_side_by_side(
#         gt_path='/scratch/seg_benchmark/splits_flat/test/masks/20250630_151349_gtFine_labelIds.png',
#         pred_path='/scratch/seg_benchmark/NEW/segnext_l-resized_seed_320K/segnext_l-resized_seed_320K/results_40K/20250630_151349_leftImg8bit.png',
#         out_path='overlay/opa_ila_overlay_segnext_l_2.png',
#         iou_threshold=0.5,
#         title="SegNext-L"  # Added title for consistency
#     )
    
    
#     visualize_all_side_by_side(
#         gt_path='/scratch/seg_benchmark/splits_flat/test/masks/20250630_151349_gtFine_labelIds.png',
#         pred_path='/scratch/seg_benchmark/NEW/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/results_40K/20250630_151349_leftImg8bit.png',
#         out_path='overlay/opa_ila_overlay_deeplab_2.png',
#         iou_threshold=0.5,
#         title="DeepLabv3plus-R50"  # Added title for consistency
#     )


# # /scratch/seg_benchmark/NEW/FCN_R50-resized_320K/FCN_R50-resized_320K/results_40K/ \
# # /scratch/seg_benchmark/NEW/segformer-mitb3-FUll-resized_seed_320K/segformer-mitb3-FUll-resized_seed_320K/results_40K \
# # /scratch/seg_benchmark/NEW/mask2former_R50_320K/results_40K/ \
# # /scratch/seg_benchmark/NEW/segnext_l-resized_seed_320K/segnext_l-resized_seed_320K/results_40K/ 
# # /scratch/seg_benchmark/NEW/segnext_l-resized_seed_320K/segnext_l-resized_seed_320K/results_40K/ 





# # import numpy as np
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # import os

# # # Replace with your actual paths
# # image_path = "/scratch/seg_benchmark/splits_flat/test/images/20250630_151349_leftImg8bit.jpg"
# # gt_path = "/scratch/seg_benchmark/splits_flat/test/masks/20250630_151349_gtFine_labelIds.png"
# # pred_path = "/scratch/seg_benchmark/NEW/segnext_l-resized_seed_320K/segnext_l-resized_seed_320K/results_40K/20250630_151349_leftImg8bit.png"

# # # Load the inputs
# # image = Image.open(image_path).convert("RGB")
# # gt = np.array(Image.open(gt_path))
# # pred = np.array(Image.open(pred_path))

# # # Calculate ILA and OPA binary masks
# # ila_mask = (gt == pred).astype(np.uint8) * 255  # pixels where prediction = ground truth
# # opa_mask = ((gt > 0) | (pred > 0)).astype(np.uint8) * 255  # pixels where either is valid

# # # Fixed colormap
# # def get_fixed_colormap(unique_labels, seed=123):
# #     np.random.seed(seed)
# #     return {label: tuple(np.random.randint(0, 256, size=3)) for label in sorted(unique_labels)}

# # def apply_colormap(label_map, colormap):
# #     h, w = label_map.shape
# #     rgb = np.zeros((h, w, 3), dtype=np.uint8)
# #     for label, color in colormap.items():
# #         rgb[label_map == label] = color
# #     return rgb

# # # Get colormap
# # all_labels = np.unique(np.concatenate([gt.flatten(), pred.flatten()]))
# # colormap = get_fixed_colormap(all_labels)

# # # Colorized GT and Pred masks
# # gt_rgb = apply_colormap(gt, colormap)
# # pred_rgb = apply_colormap(pred, colormap)

# # # Plot
# # fig, axs = plt.subplots(2, 2, figsize=(18, 14))

# # axs[0, 0].imshow(gt_rgb)
# # axs[0, 0].set_title("GT Mask (Fixed Colors)")
# # axs[0, 0].axis("off")

# # axs[0, 1].imshow(pred_rgb)
# # axs[0, 1].set_title("Prediction Mask (Fixed Colors)")
# # axs[0, 1].axis("off")

# # axs[1, 0].imshow(image)
# # axs[1, 0].imshow(ila_mask, cmap='Greens', alpha=0.5)
# # axs[1, 0].set_title("ILA Overlay (Correct Predictions)")
# # axs[1, 0].axis("off")

# # axs[1, 1].imshow(image)
# # axs[1, 1].imshow(opa_mask, cmap='Oranges', alpha=0.5)
# # axs[1, 1].set_title("OPA Overlay (Valid Regions)")
# # axs[1, 1].axis("off")

# # plt.tight_layout()
# # plt.savefig("ILA_OPA_Visualization.png")
# # plt.show()
