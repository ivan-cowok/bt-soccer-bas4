"""
Soft Instance Contrastive (Soft-IC) loss with projection head + memory bank.

Faithful re-implementation of the formulation in:
    "Precise Event Spotting in Sports Videos: Solving Long-Range Dependency
     and Class Imbalance" (Santra et al., CVPR 2025), eqs. (5)-(11).

Design summary
--------------
Two pieces:

  (a) ProjectionHead -- a small MLP that maps the post-GRU per-frame
      embedding (B, T, d_GRU*2) to a low-dimensional feature space (default
      128) and L2-normalizes the result. Trained jointly with the rest of
      the network. Lives inside the model so its weights are part of the
      checkpoint.

  (b) SoftICLoss -- an nn.Module owning a FIFO memory bank of size M
      (paper default: 256) that stores tuples
          (z_l, w_l) in (R^{feat_dim}, R^C)
      from past micro-batches. Implements eqs. (9)-(10):

        per-pair, for anchor i and class j with w_{ij} > 0:
            L_{ij} = - 1/(w_{ij} * |M(j)|) *
                       sum_{l in M(j)} w_{lj} * log(
                           exp(z_i . z_l / tau) /
                           sum_{m in A(j)} exp(z_i . z_m / tau)
                       )
          where M(j) = {l in bank : w_{lj} > 0}
                A(j) = M \ M(j)        (negatives ONLY, paper convention)

        L_SIC = mean over (i, j) with w_{ij} > 0 of L_{ij}

      A short warm-up period (until the bank holds at least `warmup_size`
      entries) yields zero loss so the contrastive term doesn't fire on a
      mostly-empty bank. The bank lives in float32 for log-sum-exp numerics.

Total training objective:
        L_final(.) = L_classification + lambda_SIC * L_SIC
"""

import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """SimCLR/MoCo-v2-style 2-layer MLP projection followed by L2-norm.

    Output dim defaults to 128 (the value used in the paper).
    """

    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else \
            max(int(out_dim), int(in_dim) // 2)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, int(out_dim)),
        )
        self.in_dim = int(in_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = int(out_dim)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


class SoftICLoss(nn.Module):
    """Soft-IC contrastive loss with FIFO memory bank.

    The projection head is NOT owned by this module; it lives in the parent
    model so its parameters are tracked by the optimizer. This module owns
    only the memory-bank state and the loss math.

    Args:
        num_classes:   C, dimensionality of the soft-label space.
        feat_dim:      d, dimensionality of the projected feature (default
                       128, matches the paper).
        bank_size:     M, number of past entries kept (default 256, paper).
        temperature:   tau in the InfoNCE-style ratio (default 0.1, paper).
        warmup_size:   minimum number of bank entries before the loss starts
                       contributing. Smaller -> the term turns on earlier
                       but with noisier statistics.
        eps:           numerical-stability constant.
    """

    def __init__(self, num_classes, feat_dim=128, bank_size=256,
                 temperature=0.1, warmup_size=32, eps=1e-8):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.bank_size = int(bank_size)
        self.temperature = float(temperature)
        self.warmup_size = int(warmup_size)
        self.eps = float(eps)

        # Bank lives in float32 -- the contrastive log-sum-exp is sensitive
        # to mantissa precision, and bf16's ~7 mantissa bits hurt under
        # autocast. persistent=False keeps the bank out of the checkpoint
        # so resumed training rebuilds it (cheap: ~1 step) and inference
        # checkpoints stay small.
        self.register_buffer(
            'bank_feats',
            torch.zeros(self.bank_size, self.feat_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'bank_labels',
            torch.zeros(self.bank_size, self.num_classes, dtype=torch.float32),
            persistent=False,
        )
        # Scalar buffers (0-dim tensors) for ring-buffer state.
        self.register_buffer(
            'bank_ptr', torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer(
            'bank_filled', torch.zeros((), dtype=torch.long), persistent=False)

        # Optional "pending" stash for SAM 2-pass training: if we enqueue
        # the first-pass features RIGHT AWAY, they're in the bank when the
        # second pass computes its loss -- so anchors would see (near) self
        # as a positive and the contrastive term collapses. Instead, we
        # stash on the first pass and flush AFTER the second pass via
        # ``flush_pending()``. Plain (not-buffer) attributes -- they hold
        # detached fp32 tensors that must NEVER be saved.
        self._pending_feats = None
        self._pending_labels = None

    # ------------------------------------------------------------------ #
    # Bank management
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _enqueue(self, feats, labels):
        """Append a batch into the FIFO ring buffer (always no-grad/fp32)."""
        feats = feats.detach().to(self.bank_feats.dtype)
        labels = labels.detach().to(self.bank_labels.dtype)
        n = feats.shape[0]
        if n == 0:
            return

        # If the incoming chunk is bigger than the bank, keep only the most
        # recent `bank_size` entries (the rest would be immediately overwritten
        # anyway).
        if n > self.bank_size:
            feats = feats[-self.bank_size:]
            labels = labels[-self.bank_size:]
            n = self.bank_size

        ptr = int(self.bank_ptr.item())
        end = ptr + n
        if end <= self.bank_size:
            self.bank_feats[ptr:end].copy_(feats)
            self.bank_labels[ptr:end].copy_(labels)
        else:
            tail = self.bank_size - ptr
            self.bank_feats[ptr:].copy_(feats[:tail])
            self.bank_labels[ptr:].copy_(labels[:tail])
            head = n - tail
            self.bank_feats[:head].copy_(feats[tail:])
            self.bank_labels[:head].copy_(labels[tail:])

        self.bank_ptr.fill_(end % self.bank_size)
        self.bank_filled.fill_(min(self.bank_size,
                                   int(self.bank_filled.item()) + n))

    def reset_bank(self):
        """Zero the bank and rewind the pointer (use across epoch boundaries
        only if you want to discard staleness; not called by default)."""
        with torch.no_grad():
            self.bank_feats.zero_()
            self.bank_labels.zero_()
            self.bank_ptr.zero_()
            self.bank_filled.zero_()
            self._pending_feats = None
            self._pending_labels = None

    @torch.no_grad()
    def flush_pending(self):
        """Enqueue the most recent stashed (feats, labels) into the bank and
        clear the stash. Called by AdaSpot.epoch() after the SAM second
        pass so the second pass never sees the current micro-batch in the
        bank. No-op when nothing is stashed."""
        if self._pending_feats is None:
            return
        self._enqueue(self._pending_feats, self._pending_labels)
        self._pending_feats = None
        self._pending_labels = None

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, features, soft_labels, enqueue='now'):
        """Compute Soft-IC loss for the current micro-batch.

        Args:
            features:    (N, feat_dim) projected, L2-normalized features.
            soft_labels: (N, num_classes) non-negative soft labels.
            enqueue:     one of
                'now'    -- enqueue this batch into the bank AFTER the loss
                            is computed (default; correct for vanilla SGD).
                'pending'-- stash this batch for a deferred enqueue via
                            ``flush_pending()``. Used between the first and
                            second SAM passes so the second pass doesn't see
                            its own (near-identical) features in the bank.
                False    -- skip the enqueue entirely. Used on the second
                            SAM pass and during validation so val features
                            don't pollute the training bank.

        Returns:
            Scalar tensor (float32). Returns 0.0 (no gradient) when the
            bank holds fewer than `warmup_size` entries.
        """
        if features.dim() != 2 or soft_labels.dim() != 2:
            raise ValueError(
                f"Soft-IC expects 2D inputs; got features={tuple(features.shape)}, "
                f"soft_labels={tuple(soft_labels.shape)}"
            )
        if features.shape[0] != soft_labels.shape[0]:
            raise ValueError(
                f"Soft-IC: N mismatch ({features.shape[0]} vs "
                f"{soft_labels.shape[0]})"
            )
        if features.shape[1] != self.feat_dim:
            raise ValueError(
                f"Soft-IC: feature dim {features.shape[1]} "
                f"!= configured feat_dim {self.feat_dim}"
            )
        if soft_labels.shape[1] != self.num_classes:
            raise ValueError(
                f"Soft-IC: label dim {soft_labels.shape[1]} "
                f"!= configured num_classes {self.num_classes}"
            )

        # Up-cast for stability under bf16 autocast.
        z = features.float()
        y = soft_labels.float()

        filled = int(self.bank_filled.item())
        # Warm-up phase: the bank is too small to give a meaningful
        # contrastive signal. We still enqueue so it fills up.
        if filled < self.warmup_size:
            self._maybe_enqueue(z, y, enqueue)
            return z.new_zeros(())

        bank_feats = self.bank_feats[:filled]    # (M, d)  -- detached
        bank_labels = self.bank_labels[:filled]  # (M, C)

        tau = self.temperature
        # Pairwise sim in fp32: (N, M).
        sim = (z @ bank_feats.t()) / tau

        total_loss = z.new_zeros(())
        total_count = 0

        # Loop over classes. C is small (~13 for SoccerNet-Ball) so this
        # is cheap and avoids a large 4-D mask tensor.
        for c in range(self.num_classes):
            pos_mask_b = (bank_labels[:, c] > 0)        # (M,)
            n_pos = int(pos_mask_b.sum().item())
            n_neg = filled - n_pos
            # InfoNCE needs at least one positive AND one negative.
            if n_pos == 0 or n_neg == 0:
                continue

            anchor_mask = (y[:, c] > 0)                 # (N,)
            n_a = int(anchor_mask.sum().item())
            if n_a == 0:
                continue

            sim_a = sim[anchor_mask]                    # (n_a, M)

            # A(c) = M \ M(c) -- denominator over negatives only (paper).
            neg_logits = sim_a[:, ~pos_mask_b]          # (n_a, n_neg)
            denom = torch.logsumexp(neg_logits, dim=-1)  # (n_a,)

            pos_logits = sim_a[:, pos_mask_b]            # (n_a, n_pos)
            log_ratio = pos_logits - denom.unsqueeze(-1)  # (n_a, n_pos)

            # Each positive contributes weighted by its own class-c soft mass.
            pos_weights = bank_labels[pos_mask_b, c]    # (n_pos,)
            weighted_sum = (pos_weights.unsqueeze(0) * log_ratio).sum(dim=-1)
            #   (n_a,)

            # Per-pair prefactor 1 / (omega * |M(c)|) from eq. (10), with
            # omega = w_{ic} (anchor's class-c soft mass).
            omega = y[anchor_mask, c]                   # (n_a,)
            per_pair = -weighted_sum / (omega * float(n_pos) + self.eps)

            total_loss = total_loss + per_pair.sum()
            total_count = total_count + n_a

        # Enqueue current batch AFTER the loss computation so anchors are
        # never contrasted against themselves.
        self._maybe_enqueue(z, y, enqueue)

        if total_count == 0:
            return z.new_zeros(())

        return total_loss / float(total_count)

    @torch.no_grad()
    def _maybe_enqueue(self, z, y, mode):
        """Dispatch enqueue/stash based on ``mode`` (see forward docstring)."""
        if mode == 'now':
            self._enqueue(z, y)
        elif mode == 'pending':
            # Stash a fp32 copy. If a previous pending batch was never
            # flushed (shouldn't happen in normal flow), we overwrite it
            # rather than silently double-stashing.
            self._pending_feats = z.detach().to(self.bank_feats.dtype).clone()
            self._pending_labels = y.detach().to(self.bank_labels.dtype).clone()
        elif mode is False or mode is None:
            return
        else:
            raise ValueError(
                f"SoftICLoss enqueue must be 'now', 'pending' or False; "
                f"got {mode!r}"
            )
