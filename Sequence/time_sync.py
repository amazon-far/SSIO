"""Time sanitization, synchronization, and interpolation."""

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pypose as pp
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def sanitize_time(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop non-increasing timestamps; keep first occurrence of strictly increasing times."""
    if t.ndim != 1:
        raise ValueError(f"Time array must be 1D, got shape {t.shape}")
    if len(t) == 0:
        return t, np.asarray([], dtype=int), 0
    keep_idx = [0]
    last = t[0]
    for i in range(1, len(t)):
        if t[i] > last:
            keep_idx.append(i)
            last = t[i]
    keep_idx_arr = np.asarray(keep_idx, dtype=int)
    dropped = len(t) - len(keep_idx_arr)
    if dropped > 0:
        warnings.warn(f"Sanitizing time dropped {dropped} non-increasing timestamps", stacklevel=2)
    return t[keep_idx_arr], keep_idx_arr, dropped


def synchronize_timeline(
    times: Iterable[np.ndarray],
    target_rate: float | None = None,
    target_times: np.ndarray | None = None,
    min_span: float | None = None,
) -> np.ndarray:
    """Build a common synchronized timeline for multiple time streams."""
    if target_times is None and (target_rate is None or target_rate <= 0):
        raise ValueError("Either target_times must be provided or target_rate must be non-None float > 0")

    starts: list[float] = []
    ends: list[float] = []
    for idx, t in enumerate(times):
        if t is None or len(t) == 0:
            raise ValueError(f"Empty time array for time stream {idx}")
        t_sanitized, _keep, _dropped = sanitize_time(np.asarray(t))
        if len(t_sanitized) < 2:
            raise ValueError(f"Less than 2 timestamps in time stream {idx} after sanitization")
        starts.append(float(t_sanitized[0]))
        ends.append(float(t_sanitized[-1]))
    t_start = float(np.max(starts))
    t_end = float(np.min(ends))
    span = t_end - t_start
    if not (t_end > t_start):
        raise ValueError("No overlapping time window across streams")
    if min_span is not None and span < min_span:
        raise ValueError(f"Overlapping span {span:.3f}s smaller than minimum required {min_span:.3f}s")

    if target_times is not None:
        sync_times = target_times
    else:
        assert target_rate is not None
        dt = 1.0 / target_rate
        sync_times = np.arange(t_start, t_end, dt, dtype=float)
    sync_times = sync_times[(sync_times >= t_start) & (sync_times <= t_end)]
    if len(sync_times) < 2:
        raise ValueError(f"Less than 2 synchronized time samples: {len(sync_times)}")

    return sync_times


def qinterp(qs: torch.Tensor, t: torch.Tensor, t_int: torch.Tensor) -> torch.Tensor:
    qs_scipy = R.from_quat(qs.numpy())
    slerp = Slerp(t.numpy(), qs_scipy)
    interp_rot = slerp(t_int.numpy()).as_quat()
    return torch.tensor(interp_rot, dtype=qs.dtype, device=qs.device)


def interp_rot(time: np.ndarray, opt_time: np.ndarray, quat: np.ndarray) -> pp.LieTensor:
    """Quaternion SLERP interpolation from opt_time to time."""
    if quat.shape[-1] != 4 or quat.ndim < 2:
        raise ValueError(f"Expected last dim=4 for quaternions, got shape {quat.shape}")

    t_keep, keep_idx, dropped = sanitize_time(opt_time)
    if len(t_keep) < 2:
        raise ValueError("At least 2 strictly increasing timestamps required for quaternion interpolation")
    quat_sanitized = np.take(quat, keep_idx, axis=-2) if dropped > 0 else quat

    imu_dt = torch.tensor(time - t_keep[0])
    gt_dt = torch.tensor(t_keep - t_keep[0])

    quat_t = torch.tensor(quat_sanitized)
    if quat_t.ndim == 2:
        out = qinterp(quat_t, gt_dt, imu_dt)
    else:
        leading = quat_t.shape[:-2]
        f = quat_t.shape[-2]
        q_flat = quat_t.reshape(-1, f, 4)
        outs = [qinterp(q_flat[i], gt_dt, imu_dt) for i in range(q_flat.shape[0])]
        out = torch.stack(outs, dim=0)
        out = out.reshape(*leading, out.shape[-2], 4)
    return pp.SO3(out)


def linear_interp_any(
    data_time: np.ndarray,
    data: np.ndarray,
    time_tgt: np.ndarray,
    time_axis: int | None = None,
) -> np.ndarray:
    """Linear interpolation along a specified time axis."""
    if time_axis is None:
        time_axis = 0 if data.ndim == 1 else -2
    if data.shape[time_axis] != len(data_time):
        raise ValueError(
            f"data_time length {len(data_time)} does not match data.shape[{time_axis}]={data.shape[time_axis]}"
        )

    t_keep, keep_idx, dropped = sanitize_time(data_time)
    if len(t_keep) < 2:
        raise ValueError("At least two valid timestamps are required for interp1d")
    data_sanitized = np.take(data, keep_idx, axis=time_axis) if dropped > 0 else data
    f = interp1d(
        t_keep,
        data_sanitized,
        axis=time_axis,
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore[arg-type]
        assume_sorted=True,
    )
    return f(time_tgt)


def resample_list(
    streams: dict[str, dict[str, Any]],
    sync_times: np.ndarray,
) -> dict[str, Any]:
    """Resample time-series streams onto a common timeline."""
    results: dict[str, Any] = {}
    for name, stream in streams.items():
        data_time = stream["time"]
        data = stream["data"]
        quat = stream.get("quat", False)
        time_axis = stream.get("time_axis", None)
        if quat:
            results[name] = interp_rot(sync_times, data_time, data)
        else:
            results[name] = torch.tensor(linear_interp_any(data_time, data, sync_times, time_axis=time_axis))
    return results
