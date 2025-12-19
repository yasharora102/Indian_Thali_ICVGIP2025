


#!/usr/bin/env python3
# app.py — FastAPI backend (SSE). Strict encoder/prototype match (no adapt mode).
import os, io, json, csv, re, uuid, time, asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import torch
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

# ── your modules
from .config import ExperimentConfig
from .segmentation import Segmenter
from .utils import remove_overlapping_masks
from .embedding import load_pe_model, load_clip_model
from .weight_model import FusionWeightNet_ROI_Conditional_Heavy




# ── env config
CFG_PATH     = os.getenv("FS_CONFIG", "../config.yaml")
IDXMAP_PATH  = os.getenv("FS_INDEX_MAP", "../index_map.json")
PROTOS_DIR   = os.getenv("FS_PROTOS_DIR", "../prototypes")
WEIGHT_CKPT  = os.getenv("FS_WEIGHT_CKPT", "../checkpoint_baseline.pth")
USE_CLIP     = 0          # 0=PE, 1=CLIP
CLIP_NAME    = "openai/clip-vit-base-patch32"
RUNS_DIR     = "runs"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_SEED   = 42
ALIASES_PATH = os.getenv("FS_ALIASES", "")                  # optional: proto-name -> index_map name



# ── app + static ──────────────────────────────────────────────────────────────
app = FastAPI(title="Thali Nutrition Scanner — Color Masks + Overlay")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)
if Path("static/index.html").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")

# ── tiny helpers ──────────────────────────────────────────────────────────────
def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',',':'))}\n\n"

async def async_sleep(ms: int):  # tiny yields help flush through proxies
    await asyncio.sleep(ms/1000)

def resize_1024(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((1024,1024), Image.BILINEAR)

def img_to_tensor_224(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tfm(img).unsqueeze(0)

def encode_with_fallback(model, x: torch.Tensor) -> torch.Tensor:
    try: return model.get_image_features(x)
    except AttributeError:
        try: return model.encode_image(x)
        except AttributeError: return model.forward_features(x)

def extract_timestamp_date_from_stem(stem: str):
    m_ts = re.search(r"(\d{8}_\d{6})", stem)
    m_d  = re.search(r"(\d{8})", stem)
    return (m_ts.group(1) if m_ts else None, m_d.group(1) if m_d else None)

def _font(sz=18):
    for n in ("DejaVuSans.ttf","arial.ttf"):
        try: return ImageFont.truetype(n, size=sz)
        except Exception: pass
    return ImageFont.load_default()

# ── nutrients (from your script) ──────────────────────────────────────────────
NUTRIENT_TABLE: Dict[int, Tuple[float,float,float,float,float]] = {
    0:(92.2,7.3,3.1,5.6,100), 1:(93,13,2,3,100), 2:(102,10.5,1.5,6.5,119),
    3:(133,20,2,6.7,100), 4:(126,19,6.5,3.3,125), 5:(122.5,10,3.8,7.5,100),
    6:(80,14,2,3,100), 7:(126,21.47,4.02,3.51,100), 8:(584,39.55,13.75,41.17,100),
    9:(81,11,2.3,3.9,156), 10:(71,11,2,2.4,100), 11:(155,21,8,4.3,100),
    12:(50,3,3,3.3,100), 13:(69,5.3,3.9,3.7,113), 14:(97,9.51,3.28,6.03,100),
    15:(46,5.74,1.25,2.45,100), 16:(163,19.2,5.6,8.42,100), 17:(116,16.84,6.53,3.03,100),
    18:(135,34.28,0.35,0.18,100), 19:(158,25.45,8.59,2.8,100), 20:(143,9.89,6.06,10.3,100),
    21:(134,20.61,6.47,3.79,100), 22:(92.2,7.3,3.1,5.6,100), 23:(187,29.76,11.42,3.08,100),
    24:(68,4,2,5,100), 25:(371,59.87,25.56,3.25,100), 26:(151,23.74,2.97,5.34,100),
    27:(258,54.26,9.39,1.67,100), 28:(101,6.43,3.31,7.21,100), 29:(165,19.77,7.04,7.08,100),
    30:(19,2.82,0.39,0.88,100), 31:(19,4.63,0.69,0.1,100), 32:(273,38.06,11.63,9.8,240),
    33:(129,28,2.67,0.28,100), 34:(223,29,12,7.7,268), 35:(130,23.33,3.16,2.53,100),
    36:(125,20,2.5,4,100), 37:(30,7.55,0.61,0.15,100), 38:(39,9.81,0.61,0.14,100),
    39:(89,22.84,1.09,0.33,100), 40:(34,8.16,0.84,0.19,100)
}
def nutrients_per_gram(cid0: int):
    tup = NUTRIENT_TABLE.get(cid0)
    if not tup: return None
    kcal, carb, prot, fat, grams = tup
    if grams <= 0: return None
    return (kcal/grams, carb/grams, prot/grams, fat/grams)
def compute_nutrients(weight_g: float, cid0: int) -> Dict[str,float]:
    per = nutrients_per_gram(cid0)
    if per is None: return dict(calories=0.0, carbs=0.0, protein=0.0, fat=0.0)
    kcal_g, carb_g, prot_g, fat_g = per
    return {"calories": weight_g*kcal_g, "carbs": weight_g*carb_g, "protein": weight_g*prot_g, "fat": weight_g*fat_g}

# ── config + models ───────────────────────────────────────────────────────────
print("[boot] Loading config:", CFG_PATH)
CFG = ExperimentConfig.load(CFG_PATH)

print("[boot] Loading index_map:", IDXMAP_PATH)
IDX2CLASS: Dict[str,str] = json.load(open(IDXMAP_PATH, "r"))
CLASS2IDX: Dict[str,int] = {v:int(k) for k,v in IDX2CLASS.items()}
MAX_ID = max(CLASS2IDX.values()) if CLASS2IDX else 255

# optional alias map for prototype label -> index_map label fixes
ALIASES = {}
if ALIASES_PATH and Path(ALIASES_PATH).exists():
    try:
        ALIASES = json.load(open(ALIASES_PATH, "r"))
    except Exception:
        ALIASES = {}

def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

CLASS2IDX_CANON = {_canon(k): v for k, v in CLASS2IDX.items()}
ALIASES_CANON   = {_canon(k): v for k, v in ALIASES.items()}

def id_from_name(name: str) -> int:
    if name in CLASS2IDX:  # exact
        return int(CLASS2IDX[name])
    c = _canon(name)
    if c in ALIASES_CANON:  # proto alias -> canonical index_map label
        target = ALIASES_CANON[c]
        return int(CLASS2IDX.get(target, 0))
    return int(CLASS2IDX_CANON.get(c, 0))

# palette (deterministic)
def build_palette(max_id: int, seed: int = 42) -> np.ndarray:
    pal = np.zeros((max_id + 1, 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)
    phi = 0.61803398875
    rng = np.random.default_rng(seed)
    for cid in range(1, max_id + 1):
        h = (cid * phi) % 1.0
        s, v = 0.68, 0.96
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        base = np.array([int(r*255), int(g*255), int(b*255)], dtype=np.int16)
        base += rng.integers(-10, 11, size=3, dtype=np.int16)
        pal[cid] = np.clip(base, 0, 255).astype(np.uint8)
    return pal
PALETTE = build_palette(MAX_ID, COLOR_SEED)

def colorize_index(idx_uint8: np.ndarray) -> Image.Image:
    return Image.fromarray(PALETTE[idx_uint8], mode="RGB")

print("[boot] Init GSAM2 Segmenter…")
SEG_DEV = torch.device(DEVICE)
SEGMENTER = Segmenter(CFG.models.segmentation, SEG_DEV)

print("[boot] Load embedder…")
if USE_CLIP:
    PE_MODEL, PE_PREPROC, _ = load_clip_model(SEG_DEV, CLIP_NAME)
    print(f"[boot] Using CLIP: {CLIP_NAME}")
else:
    PE_MODEL, PE_PREPROC, _ = load_pe_model(SEG_DEV, CFG.models.pe['config'])
    print(f"[boot] Using PE model: {CFG.models.pe['config']}")
PE_MODEL.eval().to(SEG_DEV)
with torch.no_grad():
    dummy = Image.new("RGB", (64,64), (120,120,120))
    dinp = (PE_PREPROC(images=dummy, return_tensors='pt')['pixel_values'].to(SEG_DEV)
            if USE_CLIP else PE_PREPROC(dummy).unsqueeze(0).to(SEG_DEV))
    LIVE_DIM = int(encode_with_fallback(PE_MODEL, dinp).view(1,-1).shape[1])
print(f"[boot] Live embed dim: {LIVE_DIM}")

print("[boot] Init WeightNet…")
NUM_CLASSES_FG = max(int(k) for k in IDX2CLASS.keys() if k != "0")
WEIGHT_NET = FusionWeightNet_ROI_Conditional_Heavy(
    backbone_name='resnet50', pretrained=False, unfreeze_backbone=False,
    attention_mode='none', modality='rgb', geom_type='none',
    roi_res=7, resize=(224,224), num_classes=NUM_CLASSES_FG
).to(SEG_DEV)
def _strict_load(model, path:str):
    state = torch.load(path, map_location="cpu")
    if "state_dict" in state: state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True); print("[boot] weight ckpt strict=True")
    except Exception:
        stripped = {(k[7:] if k.startswith("module.") else k): v for k,v in state.items()}
        model.load_state_dict(stripped, strict=True); print("[boot] weight ckpt (prefix stripped)")
_strict_load(WEIGHT_NET, WEIGHT_CKPT)
WEIGHT_NET.eval()
with torch.no_grad():
    _fr = WEIGHT_NET.rgb_attention(WEIGHT_NET.rgb_encoder(torch.zeros((1,3,224,224), device=SEG_DEV)))
    FEAT_H, FEAT_W = _fr.shape[2], _fr.shape[3]

# ── segmentation helpers ──────────────────────────────────────────────────────
def run_gsam2(img1024: Image.Image):
    try:
        boxes, masks, names, scores = SEGMENTER.segment(img1024, prompt="food.")
    except TypeError:
        boxes, masks, names, scores = SEGMENTER.segment(img1024)
    masks, boxes, names, _ = remove_overlapping_masks(masks, boxes, names)

    # normalize
    if isinstance(masks, np.ndarray):
        if masks.ndim != 3: raise ValueError(f"Unexpected masks shape: {masks.shape}")
        masks_list = [m.astype(bool) for m in np.asarray(masks)]
    else:
        masks_list = [np.asarray(m, dtype=bool) for m in masks]

    boxes_arr = np.asarray(boxes) if boxes is not None else np.zeros((0,4), dtype=np.float32)
    if boxes_arr.ndim == 1 and boxes_arr.size == 0:
        boxes_list = []
    elif boxes_arr.ndim == 2 and boxes_arr.shape[1] == 4:
        boxes_list = boxes_arr.tolist()
    else:
        boxes_list = list(boxes) if boxes is not None else []

    if names is None or (hasattr(names, "__len__") and len(names) == 0) or all(n is None for n in (names.tolist() if hasattr(names, "tolist") else names)):
        names_list = [f"roi_{i}" for i in range(len(masks_list))]
    else:
        names_list = names.tolist() if hasattr(names, "tolist") else list(names)

    scores_list = scores.tolist() if (scores is not None and hasattr(scores, "tolist")) else (list(scores) if scores is not None else [])

    return masks_list, boxes_list, names_list, scores_list

def drop_largest_box(masks, boxes, names, scores):
    """Remove single largest bbox by area. Works with lists/ndarrays."""
    if boxes is None: return masks, boxes, names, scores, None, 0
    boxes_arr = np.asarray(boxes)
    if boxes_arr.ndim != 2 or boxes_arr.shape[0] == 0: return masks, boxes, names, scores, None, 0
    w = np.maximum(0, boxes_arr[:, 2] - boxes_arr[:, 0])
    h = np.maximum(0, boxes_arr[:, 3] - boxes_arr[:, 1])
    areas = (w * h).astype(np.int64)
    j = int(np.argmax(areas)); dropped_area = int(areas[j])

    def _drop(seq):
        if seq is None: return None
        if isinstance(seq, np.ndarray):
            if seq.ndim == 0: return seq
            return np.delete(seq, j, axis=0)
        return [x for i, x in enumerate(seq) if i != j]

    return _drop(masks), _drop(boxes), _drop(names), _drop(scores), j, dropped_area

# ── classification ────────────────────────────────────────────────────────────
def load_prototypes_for_date(root: Path, date: str, use_mean: bool):
    p = root / date
    if not p.is_dir():
        raise RuntimeError(f"No prototypes for {date} under {root}")
    by_cls: Dict[str, List[np.ndarray]] = {}
    for f in p.rglob("*.npy"):
        v = np.load(f).reshape(-1)
        cls = f.stem.split("_",1)[0]
        by_cls.setdefault(cls, []).append(v)
    mats, labels = [], []
    for cls, arrs in by_cls.items():
        A = np.stack(arrs, 0)
        if use_mean:
            mats.append(A.mean(0)); labels.append(cls)
        else:
            for x in A: mats.append(x); labels.append(cls)
    if not mats: raise RuntimeError(f"Empty prototypes for {date}")
    M = np.stack(mats, 0)
    return M, labels, M.shape[1]

async def classify_masks(
    img1024: Image.Image, masks: List[np.ndarray], date: str, proto_root: Path,
    use_mean: bool, knn_k: int, patches_per_mask: int, pooling: str, conf_thresh: float,
    on_progress: Optional[Callable[[int,int,dict], None]] = None
):
    proto_mat, proto_labels, proto_dim = load_prototypes_for_date(proto_root, date, use_mean)
    if LIVE_DIM != proto_dim:
        raise ValueError(
            f"Embed dim mismatch: encoder outputs {LIVE_DIM}, prototypes are {proto_dim}. "
            f"Use the SAME encoder as prototypes (FS_USE_CLIP/FS_CLIP_NAME or CFG.models.pe['config'])."
        )
    knn = NearestNeighbors(n_neighbors=knn_k, metric="cosine").fit(proto_mat)

    maj_labels, maj_confs, pol_labels, pol_confs = [], [], [], []
    N = len(masks)
    for idx, m in enumerate(masks):
        ys, xs = np.where(m)
        if ys.size == 0:
            maj_labels.append("__empty__"); maj_confs.append(0.0)
            pol_labels.append("__empty__"); pol_confs.append(0.0)
            if on_progress: on_progress(idx+1, N, {"mask_id": idx, "class_name":"__empty__", "confidence":0.0})
            await async_sleep(1); continue

        pick = np.random.choice(np.arange(ys.size), size=min(patches_per_mask, ys.size), replace=ys.size<patches_per_mask)
        votes, pooled = [], {}
        for j in pick:
            cy, cx = int(ys[j]), int(xs[j])
            half = 32
            y0 = max(0, min(cy-half, img1024.height-64)); x0 = max(0, min(cx-half, img1024.width-64))
            crop = img1024.crop((x0, y0, x0+64, y0+64))
            if USE_CLIP:
                inp = PE_PREPROC(images=crop, return_tensors='pt')['pixel_values'].to(SEG_DEV)
            else:
                inp = PE_PREPROC(crop).unsqueeze(0).to(SEG_DEV)
            with torch.no_grad():
                feat = encode_with_fallback(PE_MODEL, inp).detach().float().cpu().numpy().reshape(1,-1)
            dists, idxs = knn.kneighbors(feat, return_distance=True)
            sims = 1 - dists[0]; labels = [proto_labels[i] for i in idxs[0]]
            votes.append(max(set(labels), key=labels.count))
            for lbl, s in zip(labels, sims): pooled[lbl] = pooled.get(lbl, 0.0) + float(s)

        # majority
        if votes:
            maj = max(set(votes), key=votes.count); maj_c = votes.count(maj)/max(1,len(pick))
        else:
            maj, maj_c = "__unknown_from_mask__", 0.0
        # pooled
        if pooled:
            pol = max(pooled, key=pooled.get); pol_c = pooled[pol] / (knn_k * max(1, len(pick)))
        else:
            pol, pol_c = "__unknown_from_mask__", 0.0

        if maj_c < conf_thresh: maj, maj_c = "__unknown_from_mask__", 0.0
        if pol_c < conf_thresh: pol, pol_c = "__unknown_from_mask__", 0.0

        maj_labels.append(maj); maj_confs.append(float(maj_c))
        pol_labels.append(pol); pol_confs.append(float(pol_c))

        show_label, show_conf = (pol, pol_c) if pooling == "pooled" else (maj, maj_c)
        if on_progress: on_progress(idx+1, N, {"mask_id": idx, "class_name": show_label, "confidence": float(show_conf)})
        await async_sleep(1)

    return maj_labels, maj_confs, pol_labels, pol_confs

# ── index + overlay helpers ───────────────────────────────────────────────────
def build_index_array(labels: List[str], masks: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    if not masks:
        return np.zeros((1024,1024), dtype=np.uint8), []
    H, W = masks[0].shape
    idx = np.zeros((H, W), dtype=np.uint8)
    unknown = []
    for m, cname in zip(masks, labels):
        rid = id_from_name(cname)
        if rid == 0 and cname not in ("__empty__", "__unknown_from_mask__"):
            unknown.append(cname)
        if rid > 0:
            idx[m] = rid
    return idx, sorted(set(unknown))

def save_index_mask_files(idx: np.ndarray, tag: str, out_dir: Path, run_id: str) -> dict:
    raw_p  = out_dir / f"pred_labelIds_{tag}.png"
    color_p= out_dir / f"pred_labelIds_{tag}_color.png"
    Image.fromarray(idx).save(raw_p)
    colorize_index(idx).save(color_p)
    return {"index_url": f"/runs/{run_id}/{raw_p.name}", "mask_url": f"/runs/{run_id}/{color_p.name}", "tag": tag}

def overlay_from_color_index(base_rgb: Image.Image, idx_uint8: np.ndarray, alpha: int = 96) -> Image.Image:
    color_img = colorize_index(idx_uint8).convert("RGBA")
    A = (idx_uint8 > 0).astype(np.uint8) * alpha
    alpha_img = Image.fromarray(A, mode="L")
    color_img.putalpha(alpha_img)
    return Image.alpha_composite(base_rgb.convert("RGBA"), color_img).convert("RGB")

# ── ROI builder for weights ───────────────────────────────────────────────────
def masks_to_rois(raw_ids: List[int], masks: List[np.ndarray], H: int, W: int):
    rois, cls0, areas, kept_idx = [], [], [], []
    for idx, (rid, m) in enumerate(zip(raw_ids, masks)):
        ys, xs = np.where(m)
        if ys.size == 0 or rid <= 0:
            continue
        x1,x2,y1,y2 = xs.min(), xs.max(), ys.min(), ys.max()
        rx1 = x1 / W * FEAT_W; rx2 = x2 / W * FEAT_W
        ry1 = y1 / H * FEAT_H; ry2 = y2 / H * FEAT_H
        rois.append([0, rx1, ry1, rx2, ry2])
        cls0.append(max(0, rid-1))
        areas.append(int(xs.size))
        kept_idx.append(idx)
    if not rois:
        return (torch.zeros((0,5),dtype=torch.float32,device=SEG_DEV),
                torch.zeros((0,),dtype=torch.long,device=SEG_DEV),
                [], [])
    return (torch.tensor(rois, dtype=torch.float32, device=SEG_DEV),
            torch.tensor(cls0, dtype=torch.long, device=SEG_DEV),
            areas, kept_idx)

# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    if Path("static/index.html").exists():
        return FileResponse(Path("static/index.html"))
    return JSONResponse({"ok": True, "msg": "Backend running. Put your UI at /static/index.html"})

@app.post("/api/start")
async def start_upload(file: UploadFile = File(...)):
    data = await file.read()
    img = resize_1024(Image.open(io.BytesIO(data)).convert("RGB"))
    stem = Path(file.filename).stem
    run_id = f"{stem}_{uuid.uuid4().hex[:8]}"
    out_dir = Path(RUNS_DIR) / run_id; out_dir.mkdir(parents=True, exist_ok=True)
    orig_name = f"{stem}.png"; img.save(out_dir / orig_name)
    return {"run_id": run_id, "original_url": f"/runs/{run_id}/{orig_name}"}

@app.get("/api/stream/{run_id}")
async def stream_pipeline(
    run_id: str,
    knn_k: int = 3,
    patches_per_mask: int = 10,
    pooling: str = "both",      # "majority" | "pooled" | "both"
    conf_thresh: float = 0.0,
    use_mean_protos: int = 1
):
    async def gen():
        try:
            out_dir = Path(RUNS_DIR) / run_id
            imgs = sorted(out_dir.glob("*.png"))
            if not imgs:
                yield sse_event("error", {"message":"original image missing"}); return
            img1024 = Image.open(imgs[0]).convert("RGB")

            # SEGMENTATION
            yield sse_event("stage", {"name":"segmentation","message":"Running segmentation…"})
            masks, boxes, names, scores = run_gsam2(img1024)

            # drop largest bbox (likely plate)
            masks, boxes, names, scores, dropped_idx, dropped_area = drop_largest_box(masks, boxes, names, scores)
            if dropped_idx is not None:
                yield sse_event("warn", {"message": "Dropped largest bbox (likely plate).", "index": dropped_idx, "area": dropped_area})

            if len(masks) == 0:
                yield sse_event("error", {"message":"No masks after dropping plate."}); return
            yield sse_event("segmentation", {"n_masks": len(masks)})

            # CLASSIFICATION
            yield sse_event("stage", {"name":"classification","message":"Classifying masks…"})
            cls_partial: List[dict] = []
            def on_cls(i, N, extra): cls_partial.append({**extra, "done": i, "total": N})

            ts_key, date_key = extract_timestamp_date_from_stem(imgs[0].stem)
            if not date_key: date_key = time.strftime("%Y%m%d")

            try:
                maj_labels, maj_confs, pol_labels, pol_confs = await classify_masks(
                    img1024, masks, date_key, Path(PROTOS_DIR),
                    use_mean=bool(use_mean_protos),
                    knn_k=knn_k, patches_per_mask=patches_per_mask,
                    pooling=pooling, conf_thresh=conf_thresh,
                    on_progress=on_cls
                )
            except ValueError as e:
                yield sse_event("error", {"message": str(e)}); return

            # stream per-mask classification progress
            for row in cls_partial:
                yield sse_event("classification", row); await async_sleep(1)

            # choose labels by pooling mode (for mask + overlay)
            chosen_labels = pol_labels if pooling == "pooled" else maj_labels

            # DEBUG dump
            debug = []
            for i, (maj, mc, pol, pc) in enumerate(zip(maj_labels, maj_confs, pol_labels, pol_confs)):
                debug.append({
                    "mask_id": i,
                    "majority": {"label": maj, "conf": float(mc), "raw_id": id_from_name(maj)},
                    "pooled":   {"label": pol, "conf": float(pc), "raw_id": id_from_name(pol)}
                })
            with open(out_dir / "classification_debug.json", "w") as f:
                json.dump(debug, f, indent=2)

            # INDEX MASK (raw + color) + OVERLAY from COLOR INDEX
            idx, unknown = build_index_array(chosen_labels, masks)
            if unknown:
                yield sse_event("warn", {"message":"Some predicted labels not in index_map.json; shown as background.", "labels": unknown})

            tag = ("pooled" if pooling == "pooled" else "majority")
            mask_urls = save_index_mask_files(idx, tag, out_dir, run_id)
            yield sse_event("index_mask", {"mask_url": mask_urls["mask_url"], "index_url": mask_urls["index_url"], "tag": tag})

            overlay = overlay_from_color_index(img1024, idx, alpha=96)
            overlay.save(out_dir / "overlay.png")
            yield sse_event("overlay", {"overlay_url": f"/runs/{run_id}/overlay.png"})

            # WEIGHTS
            yield sse_event("stage", {"name":"weight","message":"Estimating weights…"})
            chosen_labels = pol_labels if pooling == "pooled" else maj_labels
            raw_ids_all   = [id_from_name(c) for c in chosen_labels]

            table_rows: List[dict] = []
            totals = dict(weight_g=0.0, calories=0.0, carbs=0.0, protein=0.0, fat=0.0)

            with torch.no_grad():
                img_t = img_to_tensor_224(img1024).to(SEG_DEV)
                fr = WEIGHT_NET.rgb_attention(WEIGHT_NET.rgb_encoder(img_t))

            rois, cls_ids0, areas, kept_idx = masks_to_rois(raw_ids_all, masks, img1024.height, img1024.width)
            if rois.size(0) == 0:
                yield sse_event("warn", {"message":"No foreground ROIs retained for weight estimation."})
            else:
                with torch.no_grad():
                    B = 8; N = rois.size(0)
                    for start in range(0, N, B):
                        end = min(start+B, N)
                        part = rois[start:end]; sel = cls_ids0[start:end]
                        fd = torch.zeros((1,1,FEAT_H,FEAT_W), device=SEG_DEV, dtype=fr.dtype)
                        stats = torch.empty((0,0), device=SEG_DEV, dtype=fr.dtype)

                        preds_all = WEIGHT_NET(fr, fd, part, stats)
                        idxr = torch.arange(preds_all.size(0), device=SEG_DEV)
                        preds = preds_all[idxr, sel[:preds_all.size(0)]].float().cpu().numpy().tolist()

                        for i_local, w in enumerate(preds):
                            roi_pos = start + i_local
                            mask_id = kept_idx[roi_pos]
                            rid     = raw_ids_all[mask_id]
                            cid0    = max(0, rid-1) if rid >= 1 else 0
                            nut     = compute_nutrients(float(w), cid0)
                            conf    = (pol_confs if pooling == "pooled" else maj_confs)[mask_id]
                            cls_name= chosen_labels[mask_id]

                            row = dict(
                                mask_id=mask_id, class_name=cls_name, raw_id=rid, confidence=float(conf),
                                weight_g=float(w), calories=float(nut["calories"]), carbs=float(nut["carbs"]),
                                protein=float(nut["protein"]), fat=float(nut["fat"]),
                                area_px=int(np.count_nonzero(masks[mask_id])),
                            )
                            table_rows.append(row)
                            for k in totals.keys():
                                totals[k] += float(row[k]) if k in row else 0.0
                            yield sse_event("weight", {"row": row, "totals": totals})
                            await async_sleep(1)

            # CSV
            csv_p = out_dir / "results.csv"
            with open(csv_p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["mask_id","class_name","raw_id","weight_g","calories","carbs","protein","fat","area_px","confidence"])
                for r in table_rows:
                    w.writerow([r["mask_id"], r["class_name"], r["raw_id"],
                                f'{r["weight_g"]:.6f}', f'{r["calories"]:.6f}',
                                f'{r["carbs"]:.6f}', f'{r["protein"]:.6f}', f'{r["fat"]:.6f}',
                                r["area_px"], f'{r["confidence"]:.4f}'])
                w.writerow([])
                w.writerow(["TOTALS","", "", f'{totals["weight_g"]:.6f}', f'{totals["calories"]:.6f}',
                            f'{totals["carbs"]:.6f}', f'{totals["protein"]:.6f}', f'{totals["fat"]:.6f}', "", ""])

            # DONE
            yield sse_event("done", {
                "run_id": run_id,
                "original_url": f"/runs/{run_id}/{imgs[0].name}",
                "index_url": mask_urls["index_url"],
                "mask_url": mask_urls["mask_url"],
                "overlay_url": f"/runs/{run_id}/overlay.png",
                "csv_url": f"/runs/{run_id}/results.csv",
                "totals": totals
            })

        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")



# ── dev main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True, proxy_headers=True)