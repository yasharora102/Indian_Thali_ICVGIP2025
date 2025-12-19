import os
import json
import numpy as np
from PIL import Image
import torch
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms
from tqdm import tqdm

def load_pe_model(device, config="PE-Core-L14-336"):
    if config == 'PE-Spatial-G14-448':
        model = pe.VisionTransformer.from_config(config, pretrained=True)
    else:
        model = pe.CLIP.from_config(config, pretrained=True).to(device).eval()
    preprocess = pe_transforms.get_image_transform(model.image_size)
    if hasattr(model, 'num_parameters'):
        num_params = model.num_parameters()
    else:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded PE model with {num_params} trainable parameters.")
    return model, preprocess, num_params

from transformers import CLIPModel, CLIPProcessor

def load_clip_model(device, config="clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(config).to(device).eval()
    preprocess = CLIPProcessor.from_pretrained(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded CLIP model with {num_params} trainable parameters.")
    return model, preprocess, num_params


def encode_image(model, preprocess, image_path, device):
    img = Image.open(image_path).convert("RGB")
    inp = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        feat = model.encode_image(inp)
    arr = feat.float().cpu().numpy().reshape(-1)
    arr /= (np.linalg.norm(arr) + 1e-8)
    return arr

def precompute_daily_embeddings(prototype_dir: str, menu_json_path: str, embeddings_root: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = load_pe_model(device)
    with open(menu_json_path) as f:
        daily_menus = json.load(f)
    for date, classes in daily_menus.items():
        out_dir = os.path.join(embeddings_root, date)
        os.makedirs(out_dir, exist_ok=True)
        for cls in tqdm(classes, desc=f"[{date}]"):
            cls_folder = os.path.join(prototype_dir, cls)
            if not os.path.isdir(cls_folder):
                continue
            files = [fn for fn in os.listdir(cls_folder) if fn.lower().endswith((".png",".jpg"))]
            feats = []
            for fn in files:
                path = os.path.join(cls_folder, fn)
                try:
                    feats.append(encode_image(model, preprocess, path, device))
                except:
                    continue
            if feats:
                mean = np.stack(feats,0).mean(0)
                mean /= (np.linalg.norm(mean) + 1e-8)
                np.save(os.path.join(out_dir, f"{cls}.npy"), mean)
