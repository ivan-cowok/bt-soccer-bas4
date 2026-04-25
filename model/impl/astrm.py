"""
Adaptive Spatio-Temporal Refinement Module (ASTRM).

From: "Precise Event Spotting in Sports Videos: Solving Long-Range Dependency
and Class Imbalance" (Santra et al., CVPR 2025).
The temporal aspect is adapted from TAdaConv (Huang et al., ICLR 2022).

The module refines per-frame 2D feature maps (B*T, C, H, W) using three
sequential blocks:

  1) Local Spatial Block:    7x7 spatial attention over [max-pool, avg-pool]
                             across the channel dimension (CBAM-style),
                             applied as a residual-weighted gating.
  2) Local Temporal Block:   3D conv (3x1x1) -> BN -> ReLU -> 1x1x1 conv that
                             models short-range temporal context per channel,
                             applied as a residual-weighted gating.
  3) Global Temporal Block:  per-(B, C) dynamic 1D temporal kernel produced by
                             two FCs operating on the *time axis* of the
                             spatially-pooled signal, applied along T as a
                             depthwise conv.

Per the paper (eq. 4):    G_t(x) = softmax( f_FC( f_FC( f_GAP(x) ) ) )
Crucially, f_GAP pools the SPATIAL dimensions only -- the temporal axis is
preserved -- so the FC layers see the full T-frame channel descriptor and
therefore have a "global view of the data" when generating the kernel.

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
        temporal_expand=2,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"ASTRM kernel_size must be a positive odd int, got {kernel_size}"
            )

        T = int(n_segment)
        K = int(kernel_size)
        # Channel-reduction for the local-temporal 3D bottleneck.
        c_hidden = max(1, channels // max(1, reduction))
        # Temporal-EXPANSION factor for the global-temporal FC bottleneck.
        # Per the paper's Figure 3, FC1 maps T -> 2T (NOT a reduction); the
        # second FC then projects 2T -> K. So the FC pattern is:
        #   (B, C, T)  -> Linear(T, 2T)  -> (B, C, 2T)
        #              -> Linear(2T, K)  -> (B, C, K)
        # NB: this is a *constructor-arg local* (not the same thing as the
        # local-temporal 3D conv attribute self.t_expand below).
        t_hidden = max(1, T * max(1, int(temporal_expand)))

        self.channels = channels
        self.n_segment = T
        self.kernel_size = K
        self.c_hidden = c_hidden
        self.t_hidden = t_hidden

        # ---- Local Spatial Block (CBAM-style spatial attention) ----
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # ---- Local Temporal Block (channel-bottlenecked 3D conv) ----
        self.t_reduce = nn.Conv3d(
            channels, c_hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0),
            bias=False,
        )
        self.t_bn = nn.BatchNorm3d(c_hidden)
        self.t_expand = nn.Conv3d(
            c_hidden, channels, kernel_size=(1, 1, 1), bias=False
        )

        # ---- Global Temporal Block (eq. 4 of the paper, Figure 3) ----
        # f_GAP pools only the SPATIAL dims, so the input to the FCs has shape
        # (B, C, T). The FCs operate on the T axis (channel-shared parameters,
        # but per-(b,c) outputs because each channel has a different temporal
        # signature). Per Figure 3 the first FC expands T -> 2T (i.e. captures
        # the "global view") and the second projects 2T -> K kernel taps.
        self.g_fc1 = nn.Linear(T, t_hidden)
        self.g_fc2 = nn.Linear(t_hidden, K)

        # ---- Identity-friendly init ----
        # Local temporal: zero-init the expander so the gating is 0 at start
        # and the residual connection (1 + sigmoid(0)) is just 1.5x scaling
        # which is benign. (Same as before.)
        nn.init.zeros_(self.t_expand.weight)
        # Global temporal: zero-init FC2 weight so f_FC2(h) = bias for every
        # input. Then bias-init the center tap so softmax(bias) places almost
        # all of its mass at index K//2, i.e., the kernel is ~delta-at-center
        # at init -- the temporal conv is a near-identity, NOT a uniform
        # smoothing filter (which is what zero-bias would give us). This is
        # important: a uniform smoothing kernel at init would attenuate
        # high-frequency event signals before training even starts.
        nn.init.zeros_(self.g_fc2.weight)
        with torch.no_grad():
            self.g_fc2.bias.zero_()
            self.g_fc2.bias[K // 2] = 5.0  # softmax([0,..,5,..,0]) ≈ delta

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
        # f_GAP: spatial-only GAP -> (B, C, T). The temporal axis is preserved
        # so the FCs below have a "global view" across all T frames.
        v = x_t.mean(dim=(-2, -1))                       # (B, C, T)
        # FCs operate on the T axis (channel-shared parameters):
        #   (B, C, T) -> (B, C, t_hidden) -> (B, C, K)
        h = F.relu(self.g_fc1(v), inplace=True)
        kernel = self.g_fc2(h)                           # (B, C, K)
        # Softmax over the K kernel taps -> per-(b,c) probability distribution.
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
