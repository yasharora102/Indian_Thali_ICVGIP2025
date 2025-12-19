#!/usr/bin/env python
# extract_patches_and_embeddings.py

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from config import ExperimentConfig
from segmentation import Segmenter
from utils import remove_overlapping_masks
from embedding import load_pe_model, load_clip_model

def encode_with_fallback(model, img_tensor):
    """Try get_image_features, then encode_image, then forward_features."""
    try:
        feat = model.get_image_features(img_tensor)
    except AttributeError:
        try:
            feat = model.encode_image(img_tensor)
        except AttributeError:
            feat = model.forward_features(img_tensor)
    return feat

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='config/experiment.yaml')
    p.add_argument('--input_dir',  required=True,
                   help='Folder containing your images')
    p.add_argument('--output_dir', required=True,
                   help='Where to write masks.npy, embeddings.npy, metadata.json')
    p.add_argument('--use_clip',   action='store_true',
                   help='Use CLIP instead of PE')
    args = p.parse_args()

    # Load experiment config & segmentation model
    cfg    = ExperimentConfig.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg    = Segmenter(cfg.models.segmentation, device)

    # Load PE or CLIP model
    if args.use_clip:
        pe_model, preprocess, _ = load_clip_model(device, 'openai/clip-vit-base-patch32')
    else:
        pe_model, preprocess, _ = load_pe_model(device, cfg.models.pe['config'])
    pe_model.eval().to(device)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(Path(args.input_dir).glob('*.*')):
        stem    = img_path.stem
        out_dir = out_root / stem
        out_dir.mkdir(exist_ok=True)

        # 1) Segment + remove overlaps
        img       = Image.open(img_path).convert('RGB').resize((1024,1024))
        boxes, masks, names, scores = seg.segment(img)
        masks, boxes, names, _       = remove_overlapping_masks(masks, boxes, names)

        embeddings = []
        metadata   = []

        # 2) For each mask → crop → encode → collect
        for i, (box, mask, label, score) in enumerate(zip(boxes, masks, names, scores)):
            x1, y1, x2, y2 = map(int, box)
            patch = img.crop((x1, y1, x2, y2))
            print(f"[DEBUG] Processing patch {i} with bbox: {[x1, y1, x2, y2]}")
            
            # When using CLIP, call preprocess with images=... and return_tensors='pt'
            if args.use_clip:
                try:
                    inp_dict = preprocess(images=patch, return_tensors="pt")
                    inp = inp_dict["pixel_values"].to(device)
                except Exception as e:
                    print(f"[ERROR] Preprocessing patch {i} failed (CLIP): {e}")
                    continue
            else:
                try:
                    inp = preprocess(patch).unsqueeze(0).to(device)
                except Exception as e:
                    print(f"[ERROR] Preprocessing patch {i} failed (PE): {e}")
                    continue

            with torch.no_grad():
                try:
                    feat_tensor = encode_with_fallback(pe_model, inp)
                except Exception as e:
                    print(f"[ERROR] Encoding patch {i} failed: {e}")
                    continue

            # Cast to float32 before converting to numpy
            feat = feat_tensor.cpu().float().numpy().reshape(-1)
            embeddings.append(feat)
            metadata.append({
                'patch_id': i,
                'bbox': [x1, y1, x2, y2],
                'score': float(score),
                'label': label
            })

        # 3) Save masks, embeddings, metadata
        np.save(out_dir / 'masks.npy',
                np.stack([m.astype(np.uint8) for m in masks]))        # (N_masks, H, W)
        np.save(out_dir / 'embeddings.npy',
                np.stack(embeddings, axis=0))                        # (N_masks, D)
        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[extract] {stem}: saved {len(embeddings)} patch embeddings")

if __name__ == '__main__':
    main()
