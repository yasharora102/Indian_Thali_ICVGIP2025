import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_align

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
