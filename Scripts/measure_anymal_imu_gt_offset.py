#!/usr/bin/env python
"""Measure the true IMU<->GT clock offset for the ANYmal GrandTour GNSS missions.

Drives the real ``ANYmal_Sequence`` loader (``build_config`` omits
``imu_gt_offsets_json``, so no offset is pre-applied and we see the raw
residual) and, per mission, aligns the main-IMU gyro magnitude to the GT-attitude
angular-rate magnitude by cross-correlation on the shared uniform grid. Both are
frame-invariant magnitudes, so the box_base-vs-base frame difference is irrelevant.

Sign convention (matches the loader constant): the reported ``offset_ms`` is the
value to ADD to the ANYmal-domain timestamps (anymal_imu + proprioception) to align
them onto the cpt7 GT / PTP clock, i.e. it is exactly the number to write as
``offset_to_add_s`` in the per-IMU offsets JSON (seconds). Derivation: on the shared
grid ``gyro[t] = m(t + offset)`` where ``m`` is the true motion sampled by GT, so
``gyro[t] ~= gt_rate[t + offset/dt]``; the cross-correlation finds that lag.

Outputs a per-sequence table and, after dropping low-correlation and MAD outliers,
the robust mean offset (the constant candidate). Run from the repo root:

    python Scripts/measure_anymal_imu_gt_offset.py

The per-IMU offsets file the ANYmal dataset configs read is produced by
``measure_anymal_imu_gt_offsets_per_imu.py``.
"""
import argparse
import os
import sys

# Make the repo root importable regardless of where this is launched from
# (sys.path[0] is Scripts/ when run as a script).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pypose as pp

import Interface  # noqa: F401  registers host_env / as_device / as_dtype resolvers
from omegaconf import OmegaConf
from Sequence.ANYmal import ANYmal_Sequence

# (seq_name, codename, split) — the 39 GNSS (cpt7_ie) missions, from the
# ANYmal_{Sup29,Val,Test} dataset configs. Indoor/state_estimator missions are
# excluded (they share the ANYmal clock, so there is no cross-domain offset).
GNSS_MISSIONS = [
    ("2024-11-02-17-10-25", "SPX-1",   "sup"),
    ("2024-11-02-17-18-32", "SPX-2",   "sup"),
    ("2024-11-02-17-43-10", "SPX-3",   "sup"),
    ("2024-11-03-07-52-45", "SNOW-1",  "sup"),
    ("2024-11-03-07-57-34", "SNOW-2",  "sup"),
    ("2024-11-03-08-17-23", "SNOW-3",  "sup"),
    ("2024-11-02-21-12-51", "ICE-1",   "sup"),
    ("2024-11-03-13-51-43", "EIG-1",   "sup"),
    ("2024-11-03-13-59-54", "EIG-2",   "sup"),
    ("2024-11-04-12-55-59", "CYN-1",   "sup"),
    ("2024-11-04-13-07-13", "CYN-2",   "sup"),
    ("2024-11-11-12-07-40", "PIL-1",   "sup"),
    ("2024-11-11-12-42-47", "PIL-2",   "sup"),
    ("2024-11-14-11-17-02", "HOEB-1",  "sup"),
    ("2024-11-14-14-36-02", "KAEB-1",  "sup"),
    ("2024-11-14-15-22-43", "KAEB-2",  "sup"),
    ("2024-11-14-16-04-09", "KAEB-3",  "sup"),
    ("2024-11-15-11-18-14", "ALB-1",   "sup"),
    ("2024-11-15-11-37-15", "ALB-2",   "sup"),
    ("2024-11-15-12-06-03", "ALB-3",   "sup"),
    ("2024-11-15-14-14-12", "LMB-1",   "sup"),
    ("2024-11-15-14-43-52", "LMB-2",   "sup"),
    ("2024-11-18-12-05-01", "ARC-1",   "sup"),
    ("2024-11-18-13-22-14", "ARC-2",   "sup"),
    ("2024-11-18-13-48-19", "ARC-3",   "sup"),
    ("2024-11-18-15-46-05", "ARC-4",   "sup"),
    ("2024-11-18-16-59-23", "ARC-5",   "sup"),
    ("2024-12-03-13-15-38", "SBB-1",   "sup"),
    ("2024-12-03-13-26-40", "SBB-2",   "sup"),
    ("2024-11-14-13-45-37", "HEAP-1",  "val"),
    ("2024-11-15-10-16-35", "TRIM-1",  "val"),
    ("2024-11-15-16-41-14", "LEE-1",   "val"),
    ("2024-10-01-11-29-55", "ETH-1",   "test"),
    ("2024-10-01-12-00-49", "ETH-3",   "test"),
    ("2024-11-25-14-57-08", "LEICA-1", "test"),
    ("2024-11-25-16-36-19", "LEICA-2", "test"),
    ("2024-11-04-16-05-00", "HIL-1",   "test"),
    ("2024-11-04-10-57-34", "GRI-1",   "test"),
    ("2024-11-11-14-29-44", "ROOT-1",  "test"),
]


def angular_rate_from_quat(q_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular rate (rad/s) from a quaternion sequence via zero-phase
    central difference. q_xyzw: (N,4) xyzw."""
    q = torch.as_tensor(q_xyzw, dtype=torch.float64)
    w = np.zeros((len(q), 3))
    if len(q) >= 3:
        w[1:-1] = (pp.SO3(q[:-2]).Inv() @ pp.SO3(q[2:])).Log().numpy() / (2.0 * dt)
        w[0] = w[1]
        w[-1] = w[-2]
    return w


def best_lag(a: np.ndarray, b: np.ndarray, max_lag_samp: int):
    """Cross-correlation lag L (float samples) such that a[t] ~= b[t + L], with a
    parabolic sub-sample refine, plus the peak correlation. ADD convention: L>0 means
    a leads b (a's stamps read too small vs b), so +L*dt is added to a's timestamps."""
    a = a - a.mean()
    b = b - b.mean()
    N = len(a)
    corrs = {}
    for L in range(-max_lag_samp, max_lag_samp + 1):
        if L >= 0:
            x, y = a[: N - L], b[L:]
        else:
            x, y = a[-L:], b[: N + L]
        n = min(len(x), len(y))
        if n >= 200:
            c = np.corrcoef(x[:n], y[:n])[0, 1]
            if np.isfinite(c):
                corrs[L] = float(c)
    if not corrs:
        return 0.0, 0.0
    L0 = max(corrs, key=corrs.get)
    frac = 0.0
    if (L0 - 1) in corrs and (L0 + 1) in corrs:
        cm, c0, cp = corrs[L0 - 1], corrs[L0], corrs[L0 + 1]
        den = cm - 2 * c0 + cp
        if abs(den) > 1e-12:
            frac = float(np.clip(0.5 * (cm - cp) / den, -1.0, 1.0))
    return L0 + frac, corrs[L0]


def _self_test():
    """Lock the sign convention: build a[t] = b[t + K] and confirm best_lag = +K."""
    rng = np.random.default_rng(0)
    full = np.cumsum(rng.standard_normal(5000))   # smooth-ish random walk
    K = 4
    b = full[:4000]
    a = full[K:4000 + K]                           # a[t] = full[t+K] = b[t+K]
    L, c = best_lag(a, b, 20)
    assert abs(L - K) < 0.5, f"sign self-test FAILED: recovered L={L:.3f}, expected {K}"
    return L, c


def build_config(seq_name: str) -> OmegaConf:
    return OmegaConf.create({
        "data_root": "${host_env:DATA_ROOT}/grand_tour",
        "seq_name": seq_name,
        "device": "${as_device:'cpu'}",
        "dtype": "${as_dtype:'fp64'}",
        "acc_sigma": 0.1,
        "gyro_sigma": 0.0316227766,
        "num_imus": 1,
        "num_joints": 12,
        "target_rate": 200.0,
        "gt_source": "cpt7_ie",
    })


def measure_mission(seq_name: str, max_lag_s: float):
    seq = ANYmal_Sequence(build_config(seq_name))
    gyro = seq._gyro[:, 0, :].cpu().numpy().astype(np.float64)          # (N,3) base frame
    quat = seq._gt_poses_data[:, 3:7].cpu().numpy().astype(np.float64)  # (N,4) xyzw
    t_s = seq._time_ns[:, 0, 0].cpu().numpy().astype(np.float64) * 1e-9
    dt = float(np.median(np.diff(t_s)))
    w_gt = angular_rate_from_quat(quat, dt)
    a = np.linalg.norm(gyro, axis=1)
    b = np.linalg.norm(w_gt, axis=1)
    L, corr = best_lag(a, b, int(round(max_lag_s / dt)))
    return {
        "n": len(gyro),
        "dur_s": t_s[-1] - t_s[0],
        "dt": dt,
        "offset_ms": L * dt * 1e3,   # ADD to ANYmal-domain stamps
        "corr": corr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/grand_tour_dataset/imu_gt_offset_report.txt")
    ap.add_argument("--max-lag-s", type=float, default=0.05,
                    help="search bound (s); well within one gait period")
    ap.add_argument("--min-corr", type=float, default=0.5,
                    help="drop missions whose peak corr is below this")
    ap.add_argument("--mad-k", type=float, default=3.0,
                    help="drop offsets more than k robust-sigma from the median")
    args = ap.parse_args()

    st_L, st_c = _self_test()
    lines = []
    lines.append("ANYmal GrandTour IMU<->GT clock-offset measurement")
    lines.append("loader offset: none pre-applied (build_config omits imu_gt_offsets_json)")
    lines.append(f"sign self-test: recovered L={st_L:.3f} samples (expected 4.0), corr={st_c:.3f}")
    lines.append(f"offset_ms = value to ADD to ANYmal-domain timestamps to align onto cpt7 GT clock")
    lines.append("")
    header = f"{'codename':9s} {'split':4s} {'seq_name':21s} {'N':>7s} {'dur_s':>7s} {'offset_ms':>10s} {'corr':>6s}  note"
    lines.append(header)
    lines.append("-" * len(header))

    rows = []
    for seq_name, code, split in GNSS_MISSIONS:
        try:
            r = measure_mission(seq_name, args.max_lag_s)
            note = "" if r["corr"] >= args.min_corr else "LOW-CORR (excluded)"
            lines.append(f"{code:9s} {split:4s} {seq_name:21s} {r['n']:7d} {r['dur_s']:7.1f} "
                         f"{r['offset_ms']:+10.3f} {r['corr']:6.3f}  {note}")
            rows.append((code, split, seq_name, r, note == ""))
        except Exception as e:  # noqa: BLE001
            lines.append(f"{code:9s} {split:4s} {seq_name:21s} {'ERR':>7s} {'-':>7s} {'-':>10s} {'-':>6s}  {type(e).__name__}: {e}")
            rows.append((code, split, seq_name, None, False))

    # --- aggregate: corr gate, then MAD outlier rejection ---
    corr_ok = [r["offset_ms"] for _, _, _, r, ok in rows if ok]
    lines.append("")
    if not corr_ok:
        lines.append("No missions passed the correlation gate — cannot form a constant.")
    else:
        off = np.array(corr_ok)
        med = float(np.median(off))
        mad = 1.4826 * float(np.median(np.abs(off - med)))
        if mad > 0:
            keep = np.abs(off - med) <= args.mad_k * mad
        else:
            keep = np.ones_like(off, dtype=bool)
        kept = off[keep]
        dropped = off[~keep]
        lines.append(f"corr-gate passed : {len(off)} / {len(rows)} missions (min_corr={args.min_corr})")
        lines.append(f"MAD outliers     : {len(dropped)} dropped "
                     f"({', '.join(f'{d:+.2f}' for d in dropped) if len(dropped) else 'none'})")
        lines.append(f"kept for average : {len(kept)} missions")
        lines.append("")
        lines.append(f"  mean   offset = {kept.mean():+.4f} ms   ({kept.mean()*1e-3:+.6f} s)  <-- constant candidate")
        lines.append(f"  median offset = {np.median(kept):+.4f} ms")
        lines.append(f"  std    offset = {kept.std(ddof=1) if len(kept) > 1 else 0.0:.4f} ms")
        lines.append(f"  min/max       = {kept.min():+.4f} / {kept.max():+.4f} ms")
        lines.append("")
        lines.append(f"To apply: set offset_to_add_s = {kept.mean()*1e-3:+.6f} for this IMU in the\n  offsets JSON that the configs point `imu_gt_offsets_json` at (see\n  measure_anymal_imu_gt_offsets_per_imu.py for the per-IMU version).")

    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
