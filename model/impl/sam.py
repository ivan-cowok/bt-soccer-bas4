"""
Sharpness-Aware Minimization (SAM) and Adaptive SAM (ASAM).

References:
    Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
        Generalization," ICLR 2021.
    Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant
        Learning of Deep Neural Networks," ICML 2021.

This implementation is intentionally a thin wrapper around an *already
constructed* base optimizer (e.g. AdamW) so it slots into the existing
`BaseRGBModel.get_optimizer(...)` flow without changing its signature.

Per training step, expected usage:

    loss1 = compute_loss(batch); loss1.backward()
    sam.first_step(zero_grad=True)        # w <- w + epsilon*

    disable_bn_running_stats(model)
    loss2 = compute_loss(batch); loss2.backward()
    sam.second_step(zero_grad=True)       # w <- w - epsilon*; base_optimizer.step()
    enable_bn_running_stats(model)

The BN-momentum freeze around the second forward pass is the standard trick
to prevent the perturbed weights from polluting the running statistics.
"""

import torch
from torch import nn


class SAM:
    """Non-Optimizer wrapper around an instantiated base optimizer.

    Args:
        params: iterable of parameters to perturb (typically the same list
            already given to `base_optimizer`).
        base_optimizer: an *instance* of a torch optimizer (e.g. AdamW(...)).
        rho: perturbation radius (in parameter space).
        adaptive: if True, perturbation is element-wise scaled by |w_i| (ASAM).

    Notes:
        * This wrapper relies on `base_optimizer.param_groups` and forwards
          `step()` / `zero_grad()` to it, so any lr scheduling done on the
          base optimizer keeps working.
        * It stores the perturbation `e_w` per-parameter (in `_state`) so that
          `second_step` can subtract it before the actual base-optimizer step.
        * The first backward must produce real gradients (no loss-scaler
          scaling left over). For bfloat16 autocast (this repo's default) no
          unscaling is needed; for float16 you would call
          `scaler.unscale_(base_optimizer)` before `first_step()`.
    """

    def __init__(self, params, base_optimizer, rho=0.5, adaptive=True):
        if rho <= 0:
            raise ValueError(f"SAM rho must be > 0, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        # Share param_groups with the base optimizer so lr scheduler updates
        # propagate without surprises.
        self.param_groups = self.base_optimizer.param_groups
        self._state = {}
        # Keep `params` reachable for debugging / scripts that might want it.
        self._params = list(params) if params is not None else None

    # --- proxy API on top of the base optimizer ---------------------------------
    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    # --- SAM core ---------------------------------------------------------------
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute epsilon* and add it to the parameters."""
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.adaptive:
                    e_w = (torch.pow(p, 2)) * p.grad * scale
                else:
                    e_w = p.grad * scale
                p.add_(e_w)
                self._state[p] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, do_step=True):
        """Restore the parameters and (optionally) run the base optimizer step.

        Args:
            zero_grad: zero gradients after the step.
            do_step: if False, the perturbation is undone but `base_optimizer`
                is NOT stepped. This is the path used by gradient accumulation,
                where the caller wants to keep accumulating sharp gradients
                across more micro-batches before the actual optimizer step.
        """
        for group in self.param_groups:
            for p in group['params']:
                e_w = self._state.pop(p, None)
                if e_w is not None:
                    p.sub_(e_w)
        self._state.clear()
        if do_step:
            self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        device = self.param_groups[0]['params'][0].device
        terms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.adaptive:
                    terms.append((torch.abs(p) * p.grad).norm(p=2).to(device))
                else:
                    terms.append(p.grad.norm(p=2).to(device))
        if not terms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(terms), p=2)


# ---------------------------------------------------------------------------
# BatchNorm running-stats freeze around the SAM second pass.
#
# Without this, the second forward pass updates running mean/var using the
# perturbed weights, which both drifts the stats and slightly biases SAM.
# ---------------------------------------------------------------------------
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def disable_bn_running_stats(model):
    def fn(m):
        if isinstance(m, _BN_TYPES):
            m._sam_backup_momentum = m.momentum
            m.momentum = 0
    model.apply(fn)


def enable_bn_running_stats(model):
    def fn(m):
        if isinstance(m, _BN_TYPES) and hasattr(m, '_sam_backup_momentum'):
            m.momentum = m._sam_backup_momentum
            del m._sam_backup_momentum
    model.apply(fn)


def build_sam(params, base_optimizer, mode='none', rho=0.5):
    """Convenience factory.

    `mode` is one of: 'none', 'sam', 'asam'.
    Returns a SAM wrapper or None when disabled.
    """
    mode = (mode or 'none').lower()
    if mode == 'none':
        return None
    if mode == 'sam':
        return SAM(params, base_optimizer, rho=rho, adaptive=False)
    if mode == 'asam':
        return SAM(params, base_optimizer, rho=rho, adaptive=True)
    raise ValueError(
        f"training.sam must be one of 'none', 'sam', 'asam'; got {mode!r}"
    )
