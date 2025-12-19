import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import roi_align

from pdb import set_trace as stx

import torch.nn.functional as F
from torchvision import models

class FusionWeightNet(nn.Module):
    def __init__(
        self,
        backbone_name='efficientnet_b0',
        pretrained=True,
        fusion_type='concat',
        fc_hidden=256,
        resize=(224,224),
        use_classes=False
    ):
        super().__init__()
        self.resize = resize
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        for param in backbone.parameters():
            param.requires_grad = False
            
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])

        # infer feature channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat = self.rgb_encoder(dummy)
        feat_c = feat.size(1)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, feat_c//4, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feat_c//4, feat_c, 1)
        )
        
        if use_classes:
            self.num_classes = 1
            self.class_emb_dim = 64
            self.class_embedding = nn.Embedding(self.num_classes, self.class_emb_dim)


        in_c = feat_c*2 if fusion_type=='concat' else feat_c
        self.fusion_conv = nn.Conv2d(in_c, feat_c, 1)
        self.fusion_type = fusion_type

        self.gap = nn.AdaptiveAvgPool2d(1)
        if use_classes:
            self.fc_gap = nn.Linear(feat_c + self.class_emb_dim, 1)
        else:
            self.fc_gap = nn.Linear(feat_c, 1)

        if use_classes:
            self.fc_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten(),
                nn.Linear(feat_c*4*4 + self.class_emb_dim, fc_hidden),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(fc_hidden, 1)
            )
            
        else:
            self.fc_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten(),
                nn.Linear(feat_c*4*4, fc_hidden),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(fc_hidden, 1)
            )

    def forward(self, rgb, depth,head='gap'):
        fr = self.rgb_encoder(rgb)
        fd = self.depth_encoder(depth)
        # downsample depth features to match rgb spatial dims
        if fd.shape[2:] != fr.shape[2:]:
            fd = F.interpolate(fd, size=fr.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([fr, fd],1) if self.fusion_type=='concat' else fr+fd
        fused = self.fusion_conv(fused)
        if head=='gap':
            x = self.gap(fused).view(fused.size(0),-1)
            return self.fc_gap(x).squeeze(1)
        return self.fc_head(fused).squeeze(1)
    
    
    
class FusionWeightNet_ROI(nn.Module):
    def __init__(
        self,
        backbone_name='efficientnet_b0',
        pretrained=True,
        fusion_type='concat',
        fc_hidden=256,
        resize=(224,224)
    ):
        super().__init__()
        self.resize = resize
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])

        # infer feature channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat = self.rgb_encoder(dummy)
        feat_c = feat.size(1)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, feat_c//4, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feat_c//4, feat_c, 1)
        )

        in_c = feat_c*2 if fusion_type=='concat' else feat_c
        self.fusion_conv = nn.Conv2d(in_c, feat_c, 1)
        self.fusion_type = fusion_type

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_gap = nn.Linear(feat_c, 1)

        self.fc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(feat_c*4*4, fc_hidden),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden, 1)
        )

    
    def extract_fused(self, rgb, depth):
        """
        rgb: [B,3,H,W], depth: [B,1,H,W]
        returns fused_feats: [B, C, Hf, Wf]
        """
        fr = self.rgb_encoder(rgb)
        fd = self.depth_encoder(depth)
        if fd.shape[2:] != fr.shape[2:]:
            fd = F.interpolate(fd, size=fr.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fusion_conv(
            torch.cat([fr, fd], dim=1)
            if self.fusion_type=='concat'
            else fr + fd
        )
        return fused
    
    def forward(self, rgb, depth, head='gap'):
        fused = self.extract_fused(rgb, depth)
        if head=='gap':
            x = self.gap(fused).view(fused.size(0), -1)
            return self.fc_gap(x).squeeze(1)
        else:
            return self.fc_head(fused).squeeze(1)
        
        
        
    

class FusionWeightNet_MaskPool(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True,
                 fusion_type='concat', fc_hidden=256, resize=(224,224)):
        super().__init__()
        self.resize = resize
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.rgb_encoder   = nn.Sequential(*list(backbone.children())[:-2])
        # infer feature channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat = self.rgb_encoder(dummy)
        feat_c = feat.size(1)
        # a tiny “mask head” just to get per-instance features if you want (optional)
        self.mask_head = nn.Sequential(
            nn.Conv2d(feat_c, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)   # predicts a *mask* logit per pixel
        )

        # same depth‐encoder + fusion as before…
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat = self.rgb_encoder(dummy)
        feat_c = feat.size(1)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, feat_c//4, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(feat_c//4, feat_c, 1)
        )
        in_c = feat_c*2 if fusion_type=='concat' else feat_c
        self.fusion_conv = nn.Conv2d(in_c, feat_c, 1)
        self.fusion_type = fusion_type

        # your existing heads
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.fc_gap  = nn.Linear(feat_c, 1)
        self.fc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(feat_c*4*4, fc_hidden), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(fc_hidden, 1)
        )

    def extract_fused(self, rgb, depth):
        fr = self.rgb_encoder(rgb)
        fd = self.depth_encoder(depth)
        if fd.shape[2:] != fr.shape[2:]:
            fd = F.interpolate(fd, size=fr.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fusion_conv(
            torch.cat([fr, fd],1) if self.fusion_type=='concat' else fr+fd
        )
        return fused

    def roi_mask_pool(self, fused: torch.Tensor,
                      mask: torch.Tensor,
                      bboxes: list,
                      output_size=(7,7),
                      eps=1e-6):
        """
        fused:     [1, C, Hf, Wf]  — your fused feature map
        mask:      [H, W]          — integer mask (0=bg, >0=instances)
        bboxes:    list of dicts, each with 'bbox':[x1,y1,x2,y2]
        output_size: spatial size for ROIAlign
        → returns pooled_feats: [K, C]
        """
        # 1) build ROIs for features (scaled) and mask (in original coords)
        H, W    = mask.shape
        Hf, Wf  = fused.shape[2:]
        scale   = Hf / H
        
        rois_feat = []
        rois_mask = []
        for b in bboxes:
            x1,y1,x2,y2 = b['bbox']
            rois_feat.append([0, x1*scale, y1*scale, x2*scale, y2*scale])
            rois_mask.append([0, x1,       y1,       x2,       y2     ])
        
        device = fused.device
        rois_feat = torch.tensor(rois_feat, dtype=torch.float32, device=device)
        rois_mask = torch.tensor(rois_mask, dtype=torch.float32, device=device)

        # 2) ROIAlign on the feature map → [K, C, Ho, Wo]
        feat_rois = roi_align(
            fused, rois_feat, 
            output_size=output_size,
            spatial_scale=1.0,    # coords already scaled
            aligned=True
        )

        # 3) ROIAlign on the mask → [K, 1, Ho, Wo]
        mask_tensor = mask.unsqueeze(0).unsqueeze(0).float().to(device)
        mask_rois = roi_align(
            mask_tensor, rois_mask,
            output_size=output_size,
            spatial_scale=1.0,     # mask in original pixel coords
            aligned=True
        )
        mask_bin = (mask_rois > 0.5).float()  # [K,1,Ho,Wo]

        # 4) masked average‐pool inside each ROI
        K, C, Ho, Wo = feat_rois.shape
        feat_flat = feat_rois.view(K, C, -1)                # [K, C, Ho*Wo]
        mask_flat = mask_bin.view(K,     -1)                # [K, Ho*Wo]
        sums      = (feat_flat * mask_flat.unsqueeze(1)).sum(-1)  # [K, C]
        areas     = mask_flat.sum(-1, keepdim=True).clamp(min=eps) # [K, 1]
        pooled    = sums / areas                             # [K, C]

        return pooled

    def forward(self, rgb, depth, mask=None, bboxes=None, head='gap'):
        fused = self.extract_fused(rgb, depth)
        if head == 'roi_mask':
            assert mask is not None and bboxes is not None
            # pool per-instance features
            pooled = self.roi_mask_pool(fused, mask[0], bboxes, output_size=(7,7))
            preds  = self.fc_gap(pooled)  # reuse your global‐pool MLP
            return preds.squeeze(1)       # [K]
        elif head=='gap':
            x = self.gap(fused).view(fused.size(0), -1)
            return self.fc_gap(x).squeeze(1)
        else:
            return self.fc_head(fused).squeeze(1)
        
        

class FusionWeightNet_Class(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True,
                 fusion_type='concat', fc_hidden=256, resize=(224,224),
                 num_classes=32, class_emb_dim=64):
        super().__init__()
        self.resize=resize
        backbone=getattr(models,backbone_name)(pretrained=pretrained)
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        
        self.rgb_encoder=nn.Sequential(*list(backbone.children())[:-2])
        
        

        # infer feature channels
        with torch.no_grad():
            dummy=torch.zeros(1,3,resize[0],resize[1])
            feat=self.rgb_encoder(dummy)
        feat_c=feat.size(1)

        self.depth_encoder=nn.Sequential(
            nn.Conv2d(1,feat_c//4,3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(feat_c//4,feat_c,1)
        )

        in_c=feat_c*2 if fusion_type=='concat' else feat_c
        self.fusion_conv=nn.Conv2d(in_c,feat_c,1)
        self.fusion_type=fusion_type

        # class embedding
        self.class_emb=nn.Embedding(num_classes,class_emb_dim)
        # adjust head dims
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.fc_gap=nn.Linear(feat_c+class_emb_dim,1)

        self.fc_head=nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(feat_c*4*4+class_emb_dim,fc_hidden), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(fc_hidden,1)
        )

    def forward(self,rgb,depth,cls_ids,head='gap'):
        fr=self.rgb_encoder(rgb)
        fd=self.depth_encoder(depth)
        if fd.shape[2:]!=fr.shape[2:]:
            fd=F.interpolate(fd,size=fr.shape[2:],mode='bilinear',align_corners=False)
        fused=torch.cat([fr,fd],1) if self.fusion_type=='concat' else fr+fd
        fused=self.fusion_conv(fused)

        if head=='gap':
            x=self.gap(fused).view(fused.size(0),-1)  # [N,feat_c]
        else:
            x=self.fc_head[0:2](fused)                # Apply pooling + flatten
            x=x.view(fused.size(0),-1)

        # concat class embedding
        emb=self.class_emb(cls_ids.to(rgb.device))      # [N,emb_dim]
        x=torch.cat([x,emb],dim=1)

        return self.fc_gap(x).squeeze(1) if head=='gap' else self.fc_head[2:](x).squeeze(1)



class FusionWeightNet_Conditional(nn.Module):
    def __init__(self,
                 backbone_name='efficientnet_b0',
                 pretrained=True,
                 fusion_type='concat',
                 resize=(224,224),
                 num_classes=32,
                 unfreeze_backbone=False):
        super().__init__()
        self.resize = resize
        self.fusion_type = fusion_type

        # RGB backbone
        print(f'Unfreezing backbone: {unfreeze_backbone}')
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])
        # freeze
        if not unfreeze_backbone:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False

        # infer channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat = self.rgb_encoder(dummy)
        feat_c = feat.size(1)

        # fusion block
        in_c = feat_c + 1 if fusion_type=='concat' else feat_c
        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_c, feat_c, 3, padding=1),
            nn.BatchNorm2d(feat_c), nn.ReLU(True)
        )

        # pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # per-class heads params
        self.heads_weight = nn.Parameter(torch.randn(num_classes, feat_c) * 0.01)
        self.heads_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, rgb, depth, cls_ids):
        # stx()
        fr = self.rgb_encoder(rgb)
        fd = F.interpolate(depth, size=fr.shape[2:], mode='bilinear', align_corners=False)
        fused = fr + fd if self.fusion_type!='concat' else torch.cat([fr, fd],1)
        fused = self.fusion_block(fused)
        x = self.gap(fused).view(fused.size(0), -1)
        

        preds_all = torch.matmul(x, self.heads_weight.t()) + self.heads_bias #[42, num_classes]

        # Select each sample's class-specific output with gather (avoiding torch.cond)
        # out = preds_all.gather(1, cls_ids.unsqueeze(1)).squeeze(1)
        # return out  # [N]
        return preds_all  # [N]





class FusionWeightNet_Cond_Area(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True,
                 fusion_type='concat', resize=(224,224), num_classes=32,
                 use_geom=False, phys_emb_dim=32, unfreeze_backbone=False):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_geom = use_geom

        # backbone
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])
        if not unfreeze_backbone:
            for p in self.rgb_encoder.parameters(): p.requires_grad=False

        # infer feature channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat = self.rgb_encoder(dummy)
        feat_c = feat.size(1)

        # fusion conv
        in_c = feat_c + 1 if fusion_type=='concat' else feat_c
        self.fusion = nn.Sequential(
            nn.Conv2d(in_c, feat_c, 3, padding=1),
            nn.BatchNorm2d(feat_c), nn.ReLU(True), nn.Dropout2d(0.2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # optional geometric embedding
        if use_geom:
            self.phys_proj = nn.Sequential(
                nn.Linear(2, phys_emb_dim), nn.ReLU(True)
            )
            final_dim = feat_c + phys_emb_dim
        else:
            final_dim = feat_c

        # per-class head parameters
        self.head_w = nn.Parameter(torch.randn(num_classes, final_dim)*0.01)
        self.head_b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, img_cuts, depth_cuts, cls_ids, areas=None, volumes=None):
        # img_cuts: [N,3,H,W], depth_cuts: [N,1,H,W]
        fr = self.rgb_encoder(img_cuts)
        fd = F.interpolate(depth_cuts, size=fr.shape[2:], mode='bilinear', align_corners=False)
        fused = fr + fd if self.fusion_type!='concat' else torch.cat([fr,fd],1)
        fused = self.fusion(fused)
        x = self.gap(fused).view(fused.size(0),-1)

        if self.use_geom:
            phys = torch.stack([areas, volumes],1).to(x.device)
            pe = self.phys_proj(phys)
            x = torch.cat([x, pe],1)

        preds_all = x @ self.head_w.t() + self.head_b
        out = preds_all.gather(1, cls_ids.unsqueeze(1)).squeeze(1)
        return out
    
    
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = in_channels // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x (B, C, H, W) → y (B, C)
        avg = self.avg_pool(x).view(x.size(0), -1)
        max = self.max_pool(x).view(x.size(0), -1)

        # Feed each through the same MLP
        avg_out = self.fc(avg)
        max_out = self.fc(max)

        out = avg_out + max_out      # (B, C)
        attn = self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, query_key_channels=None, value_channels=None):
        super().__init__()
        # Simplified self-attention for image features
        # Assuming input is (B, C, H, W)
        # We can flatten spatial dimensions or use Conv for q, k, v
        
        if query_key_channels is None:
            query_key_channels = in_channels // 8 # Typical reduction
        if value_channels is None:
            value_channels = in_channels

        self.query_conv = nn.Conv2d(in_channels, query_key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, query_key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, value_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling parameter for attention output

    def forward(self, x):
        B, C, H, W = x.size()

        # Reshape to (B, C_new, H*W) for matrix multiplication
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1) # (B, H*W, C_qk)
        proj_key = self.key_conv(x).view(B, -1, H * W) # (B, C_qk, H*W)

        energy = torch.bmm(proj_query, proj_key) # (B, H*W, H*W)
        attention = F.softmax(energy, dim=-1) # Attention map

        proj_value = self.value_conv(x).view(B, -1, H * W) # (B, C_v, H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # (B, C_v, H*W)
        out = out.view(B, C, H, W) # Reshape back

        out = self.gamma * out + x # Residual connection with learnable gamma
        return out

class FusionWeightNet_Conditional_Heavy(nn.Module):
    def __init__(self,
                 backbone_name='efficientnet_b0',
                 pretrained=True,
                 fusion_type='concat',
                 resize=(224,224),
                 num_classes=32,
                 unfreeze_backbone=False):
        super().__init__()
        self.resize = resize
        self.fusion_type = fusion_type

        # RGB backbone
        print(f'Unfreezing backbone: {unfreeze_backbone}')
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])
        # freeze
        if not unfreeze_backbone:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False

        # infer channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat = self.rgb_encoder(dummy)
        feat_c = feat.size(1)

        # Pre-fusion attention for RGB features
        self.rgb_attention = CBAM(feat_c)

        # fusion block
        in_c = feat_c + 1 if fusion_type=='concat' else feat_c
        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_c, feat_c, 3, padding=1),
            nn.BatchNorm2d(feat_c),
            nn.ReLU(True),
            # Add another attention block after initial fusion convolution
            CBAM(feat_c)
        )

        # Self-attention after fusion block
        self.self_attention = SelfAttentionBlock(feat_c)

        # pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head_mlp = nn.Sequential(
            nn.Linear(feat_c, feat_c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(feat_c // 2, feat_c),
            nn.ReLU(inplace=True),
        )
        
        # Per-class heads params
        # Consider a small MLP head with attention if needed, but for now
        # the global attention before GAP should be sufficient.
        self.heads_weight = nn.Parameter(torch.randn(num_classes, feat_c) * 0.01)
        self.heads_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, rgb, depth, cls_ids):
        # 1. RGB Encoding
        fr = self.rgb_encoder(rgb)

        # 2. RGB Feature Attention (e.g., CBAM on RGB features)
        fr = self.rgb_attention(fr)

        # 3. Depth Interpolation
        fd = F.interpolate(depth, size=fr.shape[2:], mode='bilinear', align_corners=False)

        # 4. Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([fr, fd], 1)
        else: # e.g., 'add' or other element-wise operations
            fused = fr + fd

        # 5. Fusion Block (with internal attention)
        fused = self.fusion_block(fused)

        # 6. Self-Attention on Fused Features
        fused = self.self_attention(fused)

        # 7. Global Average Pooling
        x = self.gap(fused).view(fused.size(0), -1)
        
        x = self.head_mlp(x) 
        

        # 8. Classification Heads
        preds_all = torch.matmul(x, self.heads_weight.t()) + self.heads_bias #[N, num_classes]

        return preds_all
    
    
    
class FusionWeightNet_Conditional_Heavy_Film(nn.Module):
    def __init__(self,
                 backbone_name='efficientnet_b0',
                 pretrained=True,
                 fusion_type='concat',
                 resize=(224,224),
                 num_classes=32,
                 film_dim: int = 64   ,
                 unfreeze_backbone=False):
        super().__init__()
        self.resize = resize
        self.fusion_type = fusion_type

        # RGB backbone
        print(f'Unfreezing backbone: {unfreeze_backbone}')
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])
        # freeze
        if not unfreeze_backbone:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False

        # infer channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,resize[0],resize[1])
            feat = self.rgb_encoder(dummy)
        feat_c = feat.size(1)

        # Pre-fusion attention for RGB features
        self.rgb_attention = CBAM(feat_c)

        # fusion block
        in_c = feat_c + 1 if fusion_type=='concat' else feat_c
        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_c, feat_c, 3, padding=1),
            nn.BatchNorm2d(feat_c),
            nn.ReLU(True),
            # Add another attention block after initial fusion convolution
            CBAM(feat_c)
        )

        # Self-attention after fusion block
        self.self_attention = SelfAttentionBlock(feat_c)

        # pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.class_embed = nn.Embedding(num_classes, film_dim)

        # project embedding → gamma & beta of size [feat_c]
        self.film_gamma = nn.Linear(film_dim, feat_c)
        self.film_beta  = nn.Linear(film_dim, feat_c)
        
        
        self.head_mlp = nn.Sequential(
            nn.Linear(feat_c, feat_c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(feat_c // 2, feat_c),
            nn.ReLU(inplace=True),
        )
        
        # Per-class heads params
        # Consider a small MLP head with attention if needed, but for now
        # the global attention before GAP should be sufficient.
        self.heads_weight = nn.Parameter(torch.randn(num_classes, feat_c) * 0.01)
        self.heads_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, rgb, depth, cls_ids):
        # 1. RGB Encoding
        fr = self.rgb_encoder(rgb)

        # 2. RGB Feature Attention (e.g., CBAM on RGB features)
        fr = self.rgb_attention(fr)

        # 3. Depth Interpolation
        fd = F.interpolate(depth, size=fr.shape[2:], mode='bilinear', align_corners=False)

        # 4. Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([fr, fd], 1)
        else: # e.g., 'add' or other element-wise operations
            fused = fr + fd

        # 5. Fusion Block (with internal attention)
        fused = self.fusion_block(fused)

        # 6. Self-Attention on Fused Features
        fused = self.self_attention(fused)

        # 7. Global Average Pooling
        x = self.gap(fused).view(fused.size(0), -1)
        # —— FiLM modulation per sample ——  
        # cls_ids: LongTensor of shape [N]
        emb = self.class_embed(cls_ids)              # (N, film_dim)
        gamma = self.film_gamma(emb)                 # (N, feat_c)
        beta  = self.film_beta(emb)                  # (N, feat_c)

        # scale & shift
        x = gamma * x + beta     
        
        x = self.head_mlp(x) 
        

        # 8. Classification Heads
        preds_all = torch.matmul(x, self.heads_weight.t()) + self.heads_bias #[N, num_classes]

        return preds_all
    
    
class FusionWeightNet_Conditional_NEW(nn.Module):
    def __init__(self,
                 num_classes: int,
                 class_means: torch.Tensor,
                 class_stds:  torch.Tensor,
                 backbone_name='efficientnet_b0',
                 pretrained=True,
                 fusion_type='concat',
                 resize=(224,224),
                 head_hidden_dim=128,
                 unfreeze_backbone=False):
        super().__init__()
        self.resize      = resize
        self.num_classes = num_classes

        # per-class normalization stats
        self.register_buffer('class_means', class_means)  # [C]
        self.register_buffer('class_stds',  class_stds)   # [C]

        # 1) RGB backbone encoder
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])
        if not unfreeze_backbone:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False

        # infer feature channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,*resize)
            feat  = self.rgb_encoder(dummy)
        feat_c = feat.size(1)

        # 2) fusion conv (simple add)
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_c, feat_c, 3, padding=1),
            nn.BatchNorm2d(feat_c),
            nn.ReLU(True),
        )

        # 3) global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 4) per-class 2-layer MLP heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_c, head_hidden_dim),
                nn.ReLU(True),
                nn.Linear(head_hidden_dim, 1)
            )
            for _ in range(num_classes)
        ])

        # 5) initialize each head's final bias so that
        #    initial prediction (y_norm=0) → μ_c
        for head in self.heads:
            head[-1].bias.data.fill_(0.0)

    def forward(self, rgb, depth, cls_ids):
        # encode
        fr = self.rgb_encoder(rgb)
        fd = F.interpolate(depth,
                           size=fr.shape[2:],
                           mode='bilinear',
                           align_corners=False)
        fused = self.fusion(fr + fd)

        # pool
        x = self.gap(fused).view(fused.size(0), -1)  # [N, feat_c]

        # run each sample through its class-head & un-normalize
        outs = []
        for i, cid in enumerate(cls_ids):
            y_norm = self.heads[cid](x[i:i+1]).squeeze(0)  # scalar
            μ      = self.class_means[cid]
            σ      = self.class_stds[cid]
            outs.append(y_norm * σ + μ)
        return torch.stack(outs, dim=0)  # [N]
    
    
    
    
class FusionWeightNet_Conditional_ROI(nn.Module):
    def __init__(self,
                 backbone_name='efficientnet_b0',
                 pretrained=True,
                 fusion_type='concat',
                 resize=(224,224),
                 num_classes=32,
                 unfreeze_backbone=False):
        super().__init__()
        self.resize = resize
        self.fusion_type = fusion_type

        # RGB backbone
        print(f'Unfreezing backbone: {unfreeze_backbone}')
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])
        if not unfreeze_backbone:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False

        # Infer feature channel count
        with torch.no_grad():
            dummy = torch.zeros(1, 3, resize[0], resize[1])
            feat = self.rgb_encoder(dummy)
        self.feat_c = feat.size(1)

        # Fusion block
        in_c = self.feat_c + 1 if fusion_type == 'concat' else self.feat_c
        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_c, self.feat_c, 3, padding=1),
            nn.BatchNorm2d(self.feat_c),
            nn.ReLU(True)
        )

        # Global pooling + head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_mlp = nn.Sequential(
            nn.Linear(self.feat_c, self.feat_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.feat_c, self.feat_c),
            nn.ReLU(inplace=True),
        )

        self.heads_weight = nn.Parameter(torch.randn(num_classes, self.feat_c) * 0.01)
        self.heads_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, img, depth, bboxes, cls_ids=None):
        B, _, H, W = img.shape
        feats = self.rgb_encoder(img)  # [B, C, h, w]
        _, _, h, w = feats.shape

        # Resize depth for ROIAlign
        depth_feat = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)

        # Prepare ROIs in format: [batch_idx, x1, y1, x2, y2]
        rois = []
        for i, box_list in enumerate(bboxes):
            for box in box_list:
                x1, y1, x2, y2 = box['bbox']
                rois.append([i, x1 / W * w, y1 / H * h, x2 / W * w, y2 / H * h])
        rois = torch.tensor(rois, dtype=torch.float32, device=img.device)

        # ROIAlign on RGB and Depth
        rgb_rois = roi_align(feats, rois, output_size=7, aligned=True)        # [N, C, 7, 7]
        depth_rois = roi_align(depth_feat, rois, output_size=7, aligned=True) # [N, 1, 7, 7]

        # Fuse RGB + depth
        if self.fusion_type == 'concat':
            fused = torch.cat([rgb_rois, depth_rois], dim=1)  # [N, C+1, 7, 7]
        else:
            fused = rgb_rois + depth_rois                    # broadcast add

        fused = self.fusion_block(fused)  # [N, C, 7, 7]
        pooled = self.gap(fused).view(fused.size(0), -1)  # [N, C]

        # Feed through MLP head
        x = self.head_mlp(pooled)  # [N, C]

        # Compute per-class score
        preds_all = torch.matmul(x, self.heads_weight.t()) + self.heads_bias  # [N, num_classes]
        return preds_all