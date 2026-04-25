"""
Adaptive Spatio-Temporal Refinement Module (ASTRM).

From: "Precise Event Spotting in Sports Videos: Solving Long-Range Dependency
and Class Imbalance" (Santra et al., CVPR 2025).

The module refines per-frame 2D feature maps (B*T, C, H, W) using three
sequential blocks, each followed by a residual-weighted gating:

  1) Local Spatial Block:    7x7 spatial attention over [max-pool, avg-pool]
                             across the channel dimension (CBAM-style).
  2) Local Temporal Block:   3D conv (3x1x1) -> BN -> ReLU -> 1x1x1 conv that
                             models short-range temporal context per channel.
  3) Global Temporal Block:  per-(B, C) dynamic 1D temporal kernel produced by
                             FCs over the globally-pooled signal, applied along
                             T as a depthwise conv.

The module expects to be inserted INSIDE each bottleneck of a 2D backbone
(after the bottleneck's first 1x1 conv) and reshapes to/from a 5D temporal
tensor internally.

Memory note:
The global-temporal step is implemented as a single F.conv2d call with
groups = B*C (kernel_size = (K, 1)), avoiding the materialization of K
time-shifted copies of the feature map. This keeps it usable even on the
larger early-stage feature maps.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ASTRM(nn.Module):
    def __init__(
        self,
        channels,
        n_segment,
        reduction=4,
        kernel_size=3,
    ):
        super().__init__()
        if channels < reduction:
            reduction = max(1, channels)
        hidden = max(1, channels // reduction)

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"ASTRM kernel_size must be a positive odd int, got {kernel_size}"
            )

        self.channels = channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.hidden = hidden

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.t_reduce = nn.Conv3d(
            channels, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.t_bn = nn.BatchNorm3d(hidden)
        self.t_expand = nn.Conv3d(
            hidden, channels, kernel_size=(1, 1, 1), bias=False
        )

        self.g_fc1 = nn.Linear(channels, hidden)
        self.g_fc2 = nn.Linear(hidden, channels * kernel_size)

        # Identity-friendly init: zero out the gating producers so the module
        # behaves as ~identity at initialization (good when added on top of a
        # pretrained backbone).
        nn.init.zeros_(self.t_expand.weight)
        nn.init.zeros_(self.g_fc2.weight)
        nn.init.zeros_(self.g_fc2.bias)

    def forward(self, x):
        """x: (B*T, C, H, W) -- output of conv1 inside a bottleneck."""
        BT, C, H, W = x.shape
        T = self.n_segment
        if BT % T != 0:
            raise RuntimeError(
                f"ASTRM expected leading dim divisible by n_segment={T}, got BT={BT}"
            )
        B = BT // T

        # ---- 1) Local Spatial Block ----
        max_c = x.amax(dim=1, keepdim=True)              # (BT, 1, H, W)
        avg_c = x.mean(dim=1, keepdim=True)              # (BT, 1, H, W)
        s_attn = torch.sigmoid(
            self.spatial_conv(torch.cat([max_c, avg_c], dim=1))
        )                                                # (BT, 1, H, W)
        x_s = x * (1.0 + s_attn)

        # Reshape to 5D for temporal ops: (B, C, T, H, W)
        x_5d = x_s.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # ---- 2) Local Temporal Block ----
        t = self.t_reduce(x_5d)                          # (B, C/r, T, H, W)
        t = self.t_bn(t)
        t = F.relu(t, inplace=True)
        t_attn = torch.sigmoid(self.t_expand(t))         # (B, C, T, H, W)
        x_t = x_5d * (1.0 + t_attn)

        # ---- 3) Global Temporal Block ----
        # GAP over space, then over time -> (B, C)
        g_pool = x_t.mean(dim=(-2, -1)).mean(dim=-1)
        h = F.relu(self.g_fc1(g_pool), inplace=True)
        kernel = self.g_fc2(h).view(B, C, self.kernel_size)
        kernel = F.softmax(kernel, dim=-1)               # (B, C, K)

        K = self.kernel_size
        pad = K // 2
        # Memory-efficient depthwise temporal conv using grouped conv2d:
        # treat (T, H*W) as a 2D "image", apply (K, 1) kernel along T only,
        # with one independent kernel per (b, c) (groups = B*C).
        x_in = x_t.contiguous().view(1, B * C, T, H * W)
        w = kernel.reshape(B * C, 1, K, 1)
        out = F.conv2d(x_in, w, padding=(pad, 0), groups=B * C)
        out_5d = out.view(B, C, T, H, W)

        # Back to (BT, C, H, W) for the rest of the bottleneck.
        out_2d = out_5d.permute(0, 2, 1, 3, 4).contiguous().view(BT, C, H, W)
        return out_2d
