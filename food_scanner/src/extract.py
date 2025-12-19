# Script 1: extract_patches_and_embeddings.py
#!/usr/bin/env python

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
    try:
        return model.get_image_features(img_tensor)
    except AttributeError:
        try:
            return model.encode_image(img_tensor)
        except AttributeError:
            return model.forward_features(img_tensor)


def main():
    p = argparse.ArgumentParser(
        description="Extract fixed-size patches per mask and save both crops and embeddings"
    )
    p.add_argument('--config',           default='config/experiment.yaml', help='Path to experiment YAML')
    p.add_argument('--input_dir',        required=True,                 help='Folder containing raw images')
    p.add_argument('--output_dir',       required=True,                 help='Where to write patches & embeddings')
    p.add_argument('--patch_size',       type=int, default=64,          help='Height/width of each square patch')
    p.add_argument('--patches_per_mask', type=int, default=10,           help='Number of patches to sample per mask')
    p.add_argument('--use_clip',         action='store_true',           help='Use CLIP instead of PE model')
    args = p.parse_args()

    # Load models
    print("[extract] Loading models...")
    cfg    = ExperimentConfig.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg    = Segmenter(cfg.models.segmentation, device)

    if args.use_clip:
        pe_model, preprocess, _ = load_clip_model(device, 'openai/clip-vit-base-patch32')
    else:
        pe_model, preprocess, _ = load_pe_model(device, cfg.models.pe['config'])
    pe_model.eval().to(device)

    input_dir = Path(args.input_dir)
    out_root  = Path(args.output_dir)

    for img_path in sorted(input_dir.glob('*.*')):
        stem   = img_path.stem
        img_out = out_root / stem
        # Create directories
        (img_out / 'patches').mkdir(parents=True, exist_ok=True)
        (img_out / 'embeddings').mkdir(exist_ok=True)

        # Load & segment
        img = Image.open(img_path).convert('RGB').resize((1024, 1024))
        boxes, masks, names, scores = seg.segment(img)
        masks, boxes, names, _       = remove_overlapping_masks(masks, boxes, names)

        metadata = []
        # For each mask, sample k patches
        for m_idx, (box, mask, label, score) in enumerate(zip(boxes, masks, names, scores)):
            ys, xs = mask.nonzero()
            if len(ys) == 0:
                continue

            for p_idx in range(args.patches_per_mask):
                i   = np.random.randint(len(ys))
                cy, cx = int(ys[i]), int(xs[i])
                half   = args.patch_size // 2
                # Compute top-left corner
                y0 = max(0, cy - half)
                x0 = max(0, cx - half)
                # Clamp to image bounds
                y0 = min(y0, img.height - args.patch_size)
                x0 = min(x0, img.width  - args.patch_size)
                crop = img.crop((x0, y0, x0 + args.patch_size, y0 + args.patch_size))

                # Save crop image
                patch_fname = f'patch_{m_idx}_{p_idx}.png'
                crop.save(img_out / 'patches' / patch_fname)

                # Preprocess and embed
                if args.use_clip:
                    inp = preprocess(images=crop, return_tensors='pt')['pixel_values'].to(device)
                else:
                    inp = preprocess(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = encode_with_fallback(pe_model, inp)
                emb = feat.cpu().float().numpy().reshape(-1)

                # Save embedding
                emb_fname = f'emb_{m_idx}_{p_idx}.npy'
                np.save(img_out / 'embeddings' / emb_fname, emb)

                # Record metadata
                metadata.append({
                    'mask_id':  m_idx,
                    'patch_id': p_idx,
                    'center':   [cy, cx],
                    'bbox':     [int(b) for b in box],
                    'label':    label,
                    'score':    float(score),
                    'crop':     str(Path('patches')  / patch_fname),
                    'embedding':str(Path('embeddings')/ emb_fname)
                })

        # Save masks and metadata
        np.save(img_out / 'masks.npy', np.stack([m.astype(np.uint8) for m in masks]))
        with open(img_out / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[extract] {stem}: saved {len(metadata)} patches")


if __name__ == '__main__':
    main()
