import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import roi_align

from pdb import set_trace as stx

import torch.nn.functional as F
from torchvision import models

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
