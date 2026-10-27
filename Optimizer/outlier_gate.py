"""Chi-square gate on velocity observations, shared by both PGO backends.

Cap rejection growth: an outlier can corrupt the fit and make clean observations
violate the threshold. Re-score all observations each pass to allow re-admission.
Trajectory graphs use reject_all to bypass the per-pass cap; the fraction cap
and min_kept still protect the velocity gauge.
"""

from __future__ import annotations

import torch


def rejection_mask(
    d2: torch.Tensor,
    threshold: float,
    *,
    n_rejected_prev: torch.Tensor,
    max_reject_frac: float = 0.25,
    max_reject_per_pass: int = 1,
    min_kept: int = 2,
    reject_all: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (keep, n_rejected) after re-ranking all observations by current d2.

    d2 is (B, N) squared Mahalanobis distance under each observation's covariance,
    including rejected observations. threshold is a chi-square(3) quantile.
    Rejection growth is capped relative to n_rejected_prev unless reject_all;
    max_reject_frac and min_kept bound the total in either mode.

    keep is a (B, N) 0/1 tensor in d2.dtype; n_rejected is (B,) for the next pass.
    """
    B, N = d2.shape
    eligible = (d2 > threshold).sum(dim=1)                                  # (B,)

    hard  = min(max(1, int(N * max_reject_frac)), max(0, N - min_kept))
    if reject_all:
        max_reject_per_pass = N
    limit = torch.clamp(n_rejected_prev + max_reject_per_pass, max=hard)
    n_rejected = torch.minimum(eligible, limit)                             # (B,)

    # Rank worst-first over ALL observations, then reject each window's own count.
    order = d2.argsort(dim=1, descending=True)
    rank  = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(N, device=d2.device).expand_as(order))
    keep = (rank >= n_rejected[:, None]).to(d2.dtype)                       # (B, N)

    return keep, n_rejected


def mahalanobis_d2(residual: torch.Tensor, information: torch.Tensor) -> torch.Tensor:
    """(B, N) squared Mahalanobis distance ``r^T W r`` per observation."""
    return (
        residual.unsqueeze(-2) @ information @ residual.unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)
