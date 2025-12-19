#!/usr/bin/env python
import os
import argparse
import json
import random
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your dataset and model from the training script
from network7_ablations import UnifiedDataset_Cond, FusionWeightNet_ROI_Conditional_Heavy, build_rois_and_stats

# --- Set random seeds for reproducibility ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Nutrient lookup table: calories, carbs, protein, fats per portion and portion size ---
nutrient_table = {
    0: (92.2, 7.3, 3.1, 5.6, 100),
1: (93, 13, 2, 3, 100),
2: (102, 10.5, 1.5, 6.5, 119),
3: (133, 20, 2, 6.7, 100),
4: (126, 19, 6.5, 3.3, 125),
5: (122.5, 10, 3.8, 7.5, 100),
6: (80, 14, 2, 3, 100),
7: (126, 21.47, 4.02, 3.51, 100),
8: (584, 39.55, 13.75, 41.17, 100),
9: (81, 11, 2.3, 3.9, 156),
10: (71, 11, 2, 2.4, 100),
11: (155, 21, 8, 4.3, 100),
12: (50, 3, 3, 3.3, 100),
13: (69, 5.3, 3.9, 3.7, 113),
14: (97, 9.51, 3.28, 6.03, 100),
15: (46, 5.74, 1.25, 2.45, 100),
16: (163, 19.2, 5.6, 8.42, 100),
17: (116, 16.84, 6.53, 3.03, 100),
18: (135, 34.28, 0.35, 0.18, 100),
19: (158, 25.45, 8.59, 2.8, 100),
20: (143, 9.89, 6.06, 10.3, 100),
21: (134, 20.61, 6.47, 3.79, 100),
22: (92.2, 7.3, 3.1, 5.6, 100),
23: (187, 29.76, 11.42, 3.08, 100),
24: (68, 4, 2, 5, 100),
25: (371, 59.87, 25.56, 3.25, 100),
26: (151, 23.74, 2.97, 5.34, 100),
27: (258, 54.26, 9.39, 1.67, 100),
28: (101, 6.43, 3.31, 7.21, 100),
29: (165, 19.77, 7.04, 7.08, 100),
30: (19, 2.82, 0.39, 0.88, 100),
31: (19, 4.63, 0.69, 0.1, 100),
32: (273, 38.06, 11.63, 9.8, 240),
33: (129, 28, 2.67, 0.28, 100),
34: (223, 29, 12, 7.7, 268),
35: (130, 23.33, 3.16, 2.53, 100),
36: (125, 20, 2.5, 4, 100),
37: (30, 7.55, 0.61, 0.15, 100),
38: (39, 9.81, 0.61, 0.14, 100),
39: (89, 22.84, 1.09, 0.33, 100),
40: (34, 8.16, 0.84, 0.19, 100)
}
# Build per-gram multipliers
def build_nutrient_per_g(table):
    cal_g, carb_g, prot_g, fat_g = {}, {}, {}, {}
    for cid, (cal, carb, prot, fat, grams) in table.items():
        cal_g[cid]  = cal   / grams
        carb_g[cid] = carb  / grams
        prot_g[cid] = prot  / grams
        fat_g[cid]  = fat   / grams
    return cal_g, carb_g, prot_g, fat_g

calorie_per_g, carb_per_g, protein_per_g, fat_per_g = build_nutrient_per_g(nutrient_table)

# --- Evaluation functions ---
def evaluate_per_class(model, loader, device,geom_type):
    model.eval()
    all_preds, all_gts, all_cls = [], [], []
    sample_records = []
    idx_global = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval'):
            # stack inputs
            rgb   = torch.stack([s['img']   for s in batch]).to(device)
            depth = torch.stack([s['depth'] for s in batch]).to(device)
            # forward through backbone
            fr = model.rgb_encoder(rgb)
            fd = F.interpolate(depth, size=fr.shape[2:], mode='bilinear', align_corners=False)
            # build rois & stats
            rois, stats, weights, cls_ids = build_rois_and_stats(batch, fr, device,geom_type)
            # predict all classes
            preds_all = model(fr, fd, rois, stats)
            # pick true-class preds
            idx = torch.arange(preds_all.size(0), device=device)
            preds = preds_all[idx, cls_ids]
            # to numpy
            p_np, g_np, c_np = preds.cpu().numpy(), weights.cpu().numpy(), cls_ids.cpu().numpy()
            all_preds.append(p_np)
            all_gts.append(g_np)
            all_cls.append(c_np)
            # per-sample nutrient records
            for p, g, c in zip(p_np, g_np, c_np):
                mae  = abs(p - g)
                mse  = (p - g)**2
                perc = 100 * mae / (g if g != 0 else 1e-8)
                # nutrient values
                cal_gt   = g * calorie_per_g[c]
                cal_pred = p * calorie_per_g[c]
                prot_gt   = g * protein_per_g[c]
                prot_pred = p * protein_per_g[c]
                carb_gt   = g * carb_per_g[c]
                carb_pred = p * carb_per_g[c]
                fat_gt    = g * fat_per_g[c]
                fat_pred  = p * fat_per_g[c]
                sample_records.append({
                    'global_idx': idx_global,
                    'class_id':   int(c),
                    'mae': mae,
                    'mse': mse,
                    'perc': perc,
                    'gt': g,
                    'pred': p,
                    'cal_gt': cal_gt,
                    'cal_pred': cal_pred,
                    'cal_loss': abs(cal_pred - cal_gt),
                    'prot_gt': prot_gt,
                    'prot_pred': prot_pred,
                    'prot_loss': abs(prot_pred - prot_gt),
                    'carb_gt': carb_gt,
                    'carb_pred': carb_pred,
                    'carb_loss': abs(carb_pred - carb_gt),
                    'fat_gt': fat_gt,
                    'fat_pred': fat_pred,
                    'fat_loss': abs(fat_pred - fat_gt)
                })
                idx_global += 1
    preds = np.concatenate(all_preds)
    gts   = np.concatenate(all_gts)
    cls   = np.concatenate(all_cls)
    return preds, gts, cls, sample_records


def compute_metrics_per_class(preds, gts, cls_ids, num_classes):
    metrics = {}
    for c in range(num_classes):
        inds = np.where(cls_ids == c)[0]
        if len(inds) == 0: continue
        gt_vals, pred_vals = gts[inds], preds[inds]
        mae  = abs(pred_vals - gt_vals).mean()
        mse  = ((pred_vals - gt_vals)**2).mean()
        perc = (100 * abs(pred_vals - gt_vals) / np.where(gt_vals==0,1e-8,gt_vals)).mean()
        metrics[c] = {'count':len(inds), 'mae':mae, 'mse':mse, 'perc':perc}
    return metrics


def write_csv_records(records, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Saved CSV: {path}")


# --- Main ---
def main():
    p = argparse.ArgumentParser()
    # ablation flags
    p.add_argument('--modality',    choices=['rgb','depth','both'], default='rgb',
                       help="Modality to use: rgb, depth, or both")
    p.add_argument('--attention',   choices=['none','cbam','self','both'], default='none',
                       help="Attention mode: none, cbam, self, or both")
    p.add_argument('--geom_type',        choices=['none','area','depth','both'], default='none',
                       help="Geometric features: none, area, depth, or both")
    p.add_argument('--backbone',    choices=['resnet50','mobilenet_v3_large','efficientnet_b0'], default='resnet50',
                       help="Backbone network")
    p.add_argument('--roi_res',     type=int, default=7,
                       help="ROI alignment resolution")
    p.add_argument('--root_dir',    required=True)
    p.add_argument('--depth_dir',   required=True)
    p.add_argument('--weight_json', required=True)
    p.add_argument('--index_map',   required=True)
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--out_csv_class',  required=True)
    p.add_argument('--out_csv_sample', required=True)
    args = p.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load mappings
    with open(args.weight_json) as f: wdata = json.load(f)
    with open(args.index_map) as f: idx2c = json.load(f)
    # remove background if present
    for k in list(idx2c):
        if idx2c[k].lower()=='background': del idx2c[k]
    num_classes = len(idx2c)

    # dataset + loader
    ds = UnifiedDataset_Cond(args.root_dir, args.depth_dir, wdata, idx2c)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, collate_fn=lambda x: x)

    # model
    # model = FusionWeightNet_ROI_Conditional_Heavy(
    #     backbone_name='resnet50',
    #     pretrained=False,
    #     unfreeze_backbone=False,
    #     attention_mode='none',
    #     modality='rgb',
    #     geom_type='none',
    #     roi_res=7,
    #     resize=(224,224),
    #     num_classes=num_classes
    # ).to(device)
    model = FusionWeightNet_ROI_Conditional_Heavy(
        backbone_name=args.backbone,
        pretrained=False,
        unfreeze_backbone=False,
        attention_mode=args.attention,
        modality=args.modality,
        geom_type=args.geom_type,
        roi_res=args.roi_res,
        resize=(224,224),
        num_classes=num_classes
    ).to(device)
    print(f"Model: {args.backbone}, Modality: {args.modality}, Attention: {args.attention}, Geom: {args.geom_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {args.batch_size}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)

    # evaluate
    preds, gts, cls_ids, samples = evaluate_per_class(model, loader, device, args.geom_type)
    class_metrics = compute_metrics_per_class(preds, gts, cls_ids, num_classes)

    # write per-class CSV
    class_records = []
    for c, m in class_metrics.items():
        rec = {'class_id':c, 'class_name':idx2c.get(str(c), str(c)), **m}
        class_records.append(rec)
    write_csv_records(class_records, args.out_csv_class,
                      ['class_id','class_name','count','mae','mse','perc'])

    # write per-sample CSV
    write_csv_records(samples, args.out_csv_sample, list(samples[0].keys()))

        # overall metrics
    overall_mae = abs(preds-gts).mean()
    overall_mse = ((preds-gts)**2).mean()
    overall_perc = (100 * abs(preds-gts) / np.where(gts==0,1e-8,gts)).mean()
    print('Overall Metrics:')
    print(f"  MAE:  {overall_mae:.3f}")
    print(f"  MSE:  {overall_mse:.3f}")
    print(f"  % Err: {overall_perc:.2f}%")

    # overall nutrient losses
    overall_cal_loss  = np.mean([rec['cal_loss']  for rec in samples])
    overall_prot_loss = np.mean([rec['prot_loss'] for rec in samples])
    overall_carb_loss = np.mean([rec['carb_loss'] for rec in samples])
    overall_fat_loss  = np.mean([rec['fat_loss']  for rec in samples])
    print('Overall Nutrient Losses:')
    print(f"  Calories Loss: {overall_cal_loss:.3f}")
    print(f"  Protein Loss:  {overall_prot_loss:.3f}")
    print(f"  Carbs Loss:    {overall_carb_loss:.3f}")
    print(f"  Fat Loss:      {overall_fat_loss:.3f}")

if __name__ == '__main__':
    main()
