#!/usr/bin/env python
import os
import argparse
import json
import random
import csv
# from torchinfo import summary
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision import transforms, models
from torchvision.ops import roi_align
from torchvision.transforms import functional as TF
from PIL import Image

import wandb
from tqdm import tqdm

from model import CBAM, SelfAttentionBlock


def parse_args():
    parser = argparse.ArgumentParser()
    # data / train args
    parser.add_argument('--root_dir',      required=True)
    parser.add_argument('--depth_dir',     required=True)
    parser.add_argument('--val_dir',       default=None)
    parser.add_argument('--val_depth_dir', default=None)
    parser.add_argument('--weight_json',   required=True)
    parser.add_argument('--index_map',     required=True)
    parser.add_argument('--out_dir',       required=True)
    parser.add_argument('--batch_size',    type=int, default=4)
    parser.add_argument('--epochs',        type=int, default=100)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--unfreeze',      action='store_true')
    parser.add_argument('--seed',          type=int, default=42)

    # ablation flags
    parser.add_argument('--modality', choices=['rgb','depth','both'], default='rgb',
                        help="Use only RGB, only depth, or both")
    parser.add_argument('--attention', choices=['none','cbam','self','both'], default='none',
                        help="Attention mode")
    parser.add_argument('--geom', choices=['none','area','depth','both'], default='none',
                        help="Include geometric features: none, area, depth, or both")
    parser.add_argument('--backbone', choices=['resnet50','mobilenet_v3_large','efficientnet_b0'],
                        default='resnet50', help="Backbone model")
    parser.add_argument('--roi_res', type=int, choices=[5,7,14], default=7,
                        help="ROI alignment output resolution")
    parser.add_argument(
        '--depth_stats_only', action='store_true',
        help='Compute mean-depth for each ROI on the val/test set and exit'
    )
    return parser.parse_args()


class UnifiedDataset_Cond(Dataset):
    def __init__(self, root_dir, depth_dir, weight_data, index_to_class,
                 resize=(224,224), train=False):
        self.root_dir, self.depth_dir = root_dir, depth_dir
        self.weight_data, self.index_to_class = weight_data, index_to_class
        self.train = train
        self.resize = resize
        self.rgb_tf = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize(resize, Image.NEAREST),
            transforms.PILToTensor()
        ])

        # build sample list (year from first key)
        month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,
                     'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,
                     'nov':11,'dec':12}
        sample_year = int(next(iter(self.weight_data))[:4])
        self.samples = []
        for date_folder in sorted(os.listdir(root_dir)):
            parts = date_folder.split('-')
            if len(parts)!=2: continue
            day, mon = parts
            try:
                d = int(day); m = month_map[mon.lower()]
            except: continue
            key = f"{sample_year:04d}{m:02d}{d:02d}"
            if key not in self.weight_data: continue

            gt_base = os.path.join(root_dir, date_folder, 'gtFine','default')
            img_base = os.path.join(root_dir, date_folder,'imgsFine','leftImg8bit','default')
            dp_base  = os.path.join(depth_dir, date_folder,'imgsFine','leftImg8bit','default')
            if not os.path.isdir(gt_base): continue

            for set_folder in sorted(os.listdir(gt_base)):
                entries = self.weight_data[key].get(set_folder.capitalize(), [])
                weight_map = {e['food_item']: e['weight'] for e in entries}
                gt_dir  = os.path.join(gt_base, set_folder)
                img_dir = os.path.join(img_base, set_folder)
                dp_dir  = os.path.join(dp_base, set_folder)
                if not os.path.isdir(gt_dir): continue

                for fname in sorted(os.listdir(gt_dir)):
                    if not fname.lower().endswith(('.png','.jpg')): continue
                    mask_p = os.path.join(gt_dir, fname)
                    img_p  = os.path.join(img_dir, fname.replace('_gtFine_labelIds.png','_leftImg8bit.jpg'))
                    depth_npy = os.path.join(dp_dir,
                        fname.replace("_gtFine_labelIds.png","_leftImg8bit") + '_depth.npy')
                    if not (os.path.exists(img_p) and os.path.exists(depth_npy)): continue

                    mask = Image.open(mask_p).convert('L').resize(resize, Image.NEAREST)
                    m_arr = np.array(mask)
                    weights, bboxes, cls_ids = [], [], []
                    for cid in np.unique(m_arr):
                        if cid==0: continue
                        cname = self.index_to_class.get(str(cid))
                        if cname not in weight_map: continue
                        wt = float(weight_map[cname])
                        ys, xs = np.where(m_arr==cid)
                        x1, x2 = xs.min(), xs.max()
                        y1, y2 = ys.min(), ys.max()
                        new_cid = cid - 1
                        weights.append(wt)
                        bboxes.append({'class_id':cid, 'bbox':[x1,y1,x2,y2]})
                        cls_ids.append(new_cid)

                    if weights:
                        self.samples.append({
                            'img': img_p,
                            'mask': mask_p,
                            'depth_npy': depth_npy,
                            'weights': weights,
                            'bboxes': bboxes,
                            'cls_ids': cls_ids
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples[idx]
        img = Image.open(r['img']).convert('RGB')
        mask = Image.open(r['mask']).convert('L')
        depth_arr = np.load(r['depth_npy']).astype(np.float32)
        # normalize depth
        dmin, dmax = depth_arr.min(), depth_arr.max()
        depth_norm = (depth_arr - dmin)/(dmax - dmin + 1e-6)
        depth_pil = Image.fromarray((depth_norm*255).astype(np.uint8))

        # random flip
        if self.train and random.random() < 0.5:
            img       = TF.hflip(img)
            mask      = TF.hflip(mask)
            depth_pil = TF.hflip(depth_pil)

        img_t  = self.rgb_tf(img)
        mask_t = self.mask_tf(mask).squeeze(0).long()
        depth_pil = depth_pil.resize(self.resize, Image.BILINEAR)
        depth_t = transforms.ToTensor()(depth_pil).float()

        return {
            'img': img_t,
            'mask': mask_t,
            'depth': depth_t,
            'weights': torch.tensor(r['weights'], dtype=torch.float32),
            'bboxes': r['bboxes'],
            'cls_ids': torch.tensor(r['cls_ids'], dtype=torch.long)
        }


class FusionWeightNet_ROI_Conditional_Heavy(nn.Module):
    def __init__(self,
                 backbone_name='resnet50',
                 pretrained=True,
                 unfreeze_backbone=False,
                 attention_mode='none',
                 modality='rgb',
                 geom_type='none',
                 roi_res=7,
                 resize=(224,224),
                 num_classes=32):
        super().__init__()
        self.modality       = modality
        self.attention_mode = attention_mode
        self.geom_type      = geom_type
        # determine geometric feature count
        if geom_type in ('area','depth'):
            self.num_geom_feats = 1
        elif geom_type == 'both':
            self.num_geom_feats = 2
        else:
            self.num_geom_feats = 0
        self.roi_res = roi_res
        self.resize  = resize

        # 1) backbone
        if backbone_name.startswith('resnet'):
            backbone = getattr(models, backbone_name)(weights="IMAGENET1K_V2" if pretrained else None)
        elif backbone_name.startswith('mobilenet'):
            backbone = getattr(models, backbone_name)(weights="IMAGENET1K_V2" if pretrained else None)
        else:  # efficientnet
            backbone = getattr(models, backbone_name)(weights="IMAGENET1K_V1" if pretrained else None)

        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])
        if not unfreeze_backbone:
            for p in self.rgb_encoder.parameters(): p.requires_grad = False

        # infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat  = self.rgb_encoder(dummy)
        feat_c = feat.size(1)

        # 2) attention blocks
        self.rgb_attention  = CBAM(feat_c) if attention_mode in ('cbam','both') else nn.Identity()
        self.self_attention = SelfAttentionBlock(feat_c) if attention_mode in ('self','both') else nn.Identity()

            
        # 3) pooling + MLP head
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        if modality == 'rgb':
            mlp_in = feat_c
        elif modality == 'depth':
            mlp_in = 1
        else:  # both
            mlp_in = feat_c + 1
            
        # 4) MLP    
        head_in = mlp_in + self.num_geom_feats
        self.head_mlp = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, fr, fd, rois, stats):
        # ROI feature extraction
        roi_feats   = roi_align(fr, rois, output_size=(self.roi_res, self.roi_res))
        depth_feats = roi_align(fd, rois, output_size=(self.roi_res, self.roi_res))

        # modality selection
        if self.modality == 'rgb':
            inp = roi_feats
        elif self.modality == 'depth':
            inp = depth_feats
        else:
            inp = torch.cat([roi_feats, depth_feats], dim=1)

        x = self.self_attention(inp)

        x = self.gap(x).view(x.size(0), -1)  # [N, feat_c]
        if self.num_geom_feats > 0:
            x = torch.cat([x, stats], dim=1)
        preds = self.head_mlp(x)  # [N, num_classes]
        return preds


def build_rois_and_stats(batch, fr, device, geom_type):
    rois, stats, weights, cls_ids = [], [], [], []
    B, _, Hf, Wf = fr.shape
    Himg, Wimg = batch[0]['img'].shape[1:]
    for i, s in enumerate(batch):
        depth = s['depth'].to(device)
        mask  = s['mask'].to(device)
        for b in s['bboxes']:
            x1,y1,x2,y2 = b['bbox']
            rois.append([
                i,
                x1/Wimg*Wf, y1/Himg*Hf,
                x2/Wimg*Wf, y2/Himg*Hf
            ])
            m      = (mask == b['class_id']).float()
            area   = m.sum()/(Himg*Wimg)
            mean_d = (depth * m).sum()/(m.sum() + 1e-6)
            if geom_type == 'area':
                stats.append([area])
            elif geom_type == 'depth':
                stats.append([mean_d])
            elif geom_type == 'both':
                stats.append([area, mean_d])
            else:
                stats.append([])
        weights.append(s['weights'])
        cls_ids.append(s['cls_ids'])

    rois    = torch.tensor(rois, device=device, dtype=torch.float32)
    stats   = torch.tensor(stats, device=device, dtype=torch.float32)
    weights = torch.cat(weights).to(device)
    cls_ids = torch.cat(cls_ids).to(device)
    return rois, stats, weights, cls_ids


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = nn.SmoothL1Loss()
    running, total = 0., 0
    for batch in tqdm(loader, desc='Train'):
        optimizer.zero_grad()
        rgb   = torch.stack([s['img']   for s in batch]).to(device)
        depth = torch.stack([s['depth'] for s in batch]).to(device)

        fr = model.rgb_encoder(rgb)
        fr = model.rgb_attention(fr)
        fd = F.interpolate(depth, size=fr.shape[2:], mode='bilinear', align_corners=False)

        rois, stats, weights, cls_ids = build_rois_and_stats(
            batch, fr, device, model.geom_type
        )
        preds_all = model(fr, fd, rois, stats)
        idx   = torch.arange(preds_all.size(0), device=device)
        preds = preds_all[idx, cls_ids]

        loss = criterion(preds, weights)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()

        running += loss.item() * weights.size(0)
        total   += weights.size(0)

    return running / total


def evaluate(model, loader, device):
    model.eval()
    all_p, all_g = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval'):
            rgb   = torch.stack([s['img']   for s in batch]).to(device)
            depth = torch.stack([s['depth'] for s in batch]).to(device)

            fr = model.rgb_encoder(rgb)
            fr = model.rgb_attention(fr)
            fd = F.interpolate(depth, size=fr.shape[2:], mode='bilinear', align_corners=False)

            rois, stats, weights, cls_ids = build_rois_and_stats(
                batch, fr, device, model.geom_type
            )
            preds_all = model(fr, fd, rois, stats)
            idx   = torch.arange(preds_all.size(0), device=device)
            preds = preds_all[idx, cls_ids]

            all_p.append(preds.cpu().numpy())
            all_g.append(weights.cpu().numpy())

    p = np.concatenate(all_p)
    g = np.concatenate(all_g)
    mae = np.abs(p-g).mean()
    mse = ((p-g)**2).mean()
    perc = (100*np.abs(p-g)/(np.where(g==0,1e-8,g))).mean()
    return mae, mse, perc


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # wandb init
    run = wandb.init(
        project='weight-estimation',
        config=vars(args),
        dir=args.out_dir,
        name=f"abl_{args.modality}_{args.attention}_geom-{args.geom}_{args.backbone}_roi{args.roi_res}"
    )

    # load metadata
    with open(args.weight_json) as f: wdata = json.load(f)
    with open(args.index_map)    as f: idx2c = json.load(f)
    for k in [k for k,v in idx2c.items() if v.lower()=="background"]:
        idx2c.pop(k, None)
    num_classes = len(idx2c)

    # data loaders
    train_ds = UnifiedDataset_Cond(args.root_dir, args.depth_dir, wdata, idx2c, train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=8,
                              collate_fn=lambda x: x)
    val_loader = None
    if args.val_dir:
        val_ds = UnifiedDataset_Cond(args.val_dir, args.val_depth_dir, wdata, idx2c, train=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=8,
                                collate_fn=lambda x: x)

    # model / optimizer / scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionWeightNet_ROI_Conditional_Heavy(
        backbone_name    = args.backbone,
        unfreeze_backbone= args.unfreeze,
        attention_mode   = args.attention,
        modality         = args.modality,
        geom_type        = args.geom,
        roi_res          = args.roi_res,
        resize           = (224,224),
        num_classes      = num_classes
    ).to(device)

    print(model)
    print(summary(model))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.lr,
                           epochs=args.epochs,
                           steps_per_epoch=len(train_loader),
                           pct_start=0.1,
                           anneal_strategy='cos')

    best_mae = float('inf')
    best_model_path = os.path.join(args.out_dir, 'best_model.pth')

    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        wandb.log({'train_loss': tr_loss, 'epoch': ep})
        if val_loader:
            mae, mse, perc = evaluate(model, val_loader, device)
            wandb.log({'val_mae': mae, 'val_mse': mse, 'val_perc': perc, 'epoch': ep})
            print(f"[{ep}/{args.epochs}] train_loss={tr_loss:.4f} val_mae={mae:.4f}")
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), best_model_path)
                artifact = wandb.Artifact(name="best_model", type="model")
                artifact.add_file(best_model_path)
                run.log_artifact(artifact, aliases=["best", f"epoch_{ep}"])
                print(f"Logged best model at epoch {ep} with MAE: {best_mae:.4f}")
        else:
            print(f"[{ep}/{args.epochs}] train_loss={tr_loss:.4f}")

        if ep % 10 == 0:
            epoch_model_path = os.path.join(args.out_dir, f'model_ep_{ep}.pth')
            torch.save(model.state_dict(), epoch_model_path)
            artifact = wandb.Artifact(name=f"model_checkpoint_epoch_{ep}", type="model")
            artifact.add_file(epoch_model_path)
            run.log_artifact(artifact)
            print(f"Logged checkpoint epoch {ep}.")

    # final save
    final_model_path = os.path.join(args.out_dir,'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    artifact = wandb.Artifact(name="final_model", type="model")
    artifact.add_file(final_model_path)
    run.log_artifact(artifact, aliases=["latest"])
    print("Logged final model.")

    # results
    with open(os.path.join(args.out_dir,'results.csv'),'w') as f:
        writer = csv.writer(f)
        writer.writerow(['best_val_mae'])
        writer.writerow([best_mae])

    wandb.finish()


if __name__ == '__main__':
    main()
