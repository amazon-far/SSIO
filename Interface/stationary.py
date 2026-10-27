"""IMU-only stationary detection, shared by ISequence's loaders.

Kept in Interface to avoid a dependency back to Sequence.
"""

import torch
import torch.nn.functional as F

def median_dt_s(time_ns: torch.Tensor) -> float:
    """Median sample period in seconds from int64 nanosecond stamps.

    Difference int64 stamps before scaling to avoid collapsing adjacent samples.
    Reject non-monotonic clocks.
    """
    t = time_ns.reshape(-1)
    if t.numel() < 2:
        raise ValueError(f"median_dt_s: need at least 2 stamps, got {t.numel()}")
    if not bool(torch.all(t[1:] > t[:-1])):
        raise ValueError("median_dt_s: time_ns is not strictly increasing")
    return float((t[1:] - t[:-1]).to(torch.float64).median().item() * 1e-9)


def _rolling_mean(x: torch.Tensor, win: int) -> torch.Tensor:
    """Centered rolling mean over the last axis."""
    pad_l = win // 2
    pad_r = win - 1 - pad_l
    xp = F.pad(x.unsqueeze(1), (pad_l, pad_r), mode="replicate")   # (B, 1, N+win-1)
    k  = x.new_ones(1, 1, win) / win
    return F.conv1d(xp, k).squeeze(1)                              # (B, N)


def detect_stationary(
    acc          : torch.Tensor,   # (B, N, 3) RAW specific force (retains gravity)
    gyro         : torch.Tensor,   # (B, N, 3) RAW angular rate
    dt           : float,          # seconds between consecutive samples
    *,
    a_std_max    : float = 0.20,   # rolling std of ||acc|| (m/s^2)
    gyro_mean_max: float = 0.10,   # rolling mean of ||gyro|| (rad/s)
    gyro_std_max : float = 0.05,   # rolling std  of ||gyro|| (rad/s)
    grav_dev_max : float = 0.40,   # |rolling mean ||acc|| - g| (m/s^2)
    gravity      : float = 9.81007,
    win_s        : float = 1.0,
) -> torch.Tensor:
    """Detect stationary frames from raw main-body IMU statistics."""
    win = round(win_s / dt)
    assert win >= 4

    a_norm   = acc.norm(dim=-1)                            # (B, N)
    w_norm   = gyro.norm(dim=-1)
    a_mean   = _rolling_mean(a_norm, win)
    a_std    = (_rolling_mean(a_norm.pow(2), win) - a_mean.pow(2)).clamp(min=0).sqrt()
    w_mean   = _rolling_mean(w_norm, win)
    # Mean rejects steady rotation; standard deviation rejects zero-mean oscillation.
    w_std    = (_rolling_mean(w_norm.pow(2), win) - w_mean.pow(2)).clamp(min=0).sqrt()
    grav_dev = (a_mean - gravity).abs()
    return ((a_std  < a_std_max)     & (w_mean   < gyro_mean_max) &
            (w_std  < gyro_std_max)  & (grav_dev < grav_dev_max))
