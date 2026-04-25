"""
Soft Instance Contrastive (Soft-IC) loss.

From: "Precise Event Spotting in Sports Videos: Solving Long-Range Dependency
and Class Imbalance" (Santra et al., CVPR 2025).

Motivation:
    Standard contrastive losses (InfoNCE, SupCon) treat each pair as either
    "positive" or "negative" based on hard class membership. This fails on
    severely imbalanced action-spotting datasets where rare classes get few
    positives per batch and the background class drowns the signal.

    Soft-IC uses *soft* labels. Each instance i has a probability distribution
    y_i over C classes (one-hot in the standard case, or a true mixture under
    mixup). The loss encourages each instance's similarity-weighted neighborhood
    to recover its own soft label distribution.

Formulation:
    Let z_i in R^d be an L2-normalized instance feature, y_i in R^C its
    (non-negative) soft label vector, and tau the temperature.

        s_ij = (z_i . z_j) / tau                          (pairwise similarity)
        a_ij = softmax_j(s_ij), with diagonal masked to -inf
        q_i  = sum_j a_ij * y_j                           (similarity-mixed labels)
        L_i  = - sum_c y_i,c * log(q_i,c + eps)
        L_SIC = (sum_i w_i * L_i) / (sum_i w_i + eps)

    where w_i = sum_c y_i,c is the "label mass" of instance i. This makes
    instances with all-zero soft labels (e.g. the implicit-background frames
    under BCE-YOLO) contribute exactly zero to the numerator AND zero to the
    denominator, so they don't dilute the signal but still appear as candidate
    "neighbors" in the similarity matrix.

Total training objective when enabled:
        L_final(.) = L_classification + lambda_SIC * L_SIC
"""

import torch
import torch.nn.functional as F


def soft_ic_loss(features, soft_labels, temperature=0.1, eps=1e-8):
    """
    Args:
        features:    (N, d) instance features. Will be L2-normalized.
        soft_labels: (N, C) non-negative soft labels per instance. Need not
                     sum to 1 (we re-normalize internally for the per-instance
                     target distribution; the raw mass is used for weighting).
        temperature: softmax temperature for the similarity matrix.
        eps:         numerical-stability constant.

    Returns:
        A scalar tensor: the Soft-IC loss for this batch.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be 2D (N, d); got {features.shape}")
    if soft_labels.dim() != 2:
        raise ValueError(
            f"soft_labels must be 2D (N, C); got {soft_labels.shape}"
        )
    if features.shape[0] != soft_labels.shape[0]:
        raise ValueError(
            f"features and soft_labels must agree on N; "
            f"got {features.shape[0]} vs {soft_labels.shape[0]}"
        )

    N = features.shape[0]
    if N < 2:
        # Degenerate: only one instance, no contrastive signal possible.
        return features.new_zeros(())

    # Use float32 for the contrastive sim matrix even under bfloat16 autocast.
    # bf16 has enough range but its precision (~7 mantissa bits) can hurt the
    # softmax-of-similarities numerics on a few hundred frames.
    z = F.normalize(features.float(), dim=-1)
    y = soft_labels.float()

    sim = z @ z.t() / float(temperature)                # (N, N)

    # Mask the diagonal so each instance attends only to other instances.
    diag = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, float('-inf'))

    a = F.softmax(sim, dim=-1)                          # (N, N), rows sum to 1
    q = a @ y                                           # (N, C)

    # Per-instance soft cross-entropy. We want each row of y to act as the
    # *target distribution*, so we normalize y per row to a proper distribution
    # for the log-likelihood; the original mass is reused as the per-row weight.
    row_mass = y.sum(dim=-1, keepdim=True)              # (N, 1)
    target = y / (row_mass + eps)                       # (N, C); zero rows -> zero rows
    per_row = -(target * torch.log(q + eps)).sum(dim=-1)  # (N,)

    w = row_mass.squeeze(-1)                            # (N,)
    denom = w.sum() + eps
    return (w * per_row).sum() / denom
