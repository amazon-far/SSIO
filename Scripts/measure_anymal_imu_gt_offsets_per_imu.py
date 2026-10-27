#!/usr/bin/env python
"""Per-IMU IMU<->GT time-offset measurement for the ANYmal GrandTour GNSS missions.

Companion to ``measure_anymal_imu_gt_offset.py`` (which measures only the main IMU
through the loader). This one reads EVERY IMU topic directly from the zarr, so it is
independent of the loader and of any offset the loader would pre-apply from
``imu_gt_offsets_json``. For each IMU it aligns gyro magnitude to the
cpt7 GT attitude-rate magnitude by cross-correlation on a shared uniform grid, over all
39 GNSS missions, then reports per-IMU mean / std / median / range and the number of
missions dropped (low correlation + MAD outliers).

Sign convention (same as the companion): ``offset_ms`` is the value to ADD to that IMU's
timestamps to align its signal onto the cpt7 GT signal. These offsets conflate a true
clock skew with per-sensor gyro group delay and the Inertial-Explorer GT solution latency
(cpt7_imu, co-located with the GT unit, is itself nonzero), so read them as "align this
IMU's signal to the GT signal", not as pure clock offsets.

Outputs JSON + a text summary under outputs/grand_tour_dataset/. Run from
the repo root: 
    python Scripts/measure_anymal_imu_gt_offsets_per_imu.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import zarr
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

# Reuse the mission list, the angular-rate helper, the cross-correlation, and the
# sign self-test from the main script (single source of truth).
from measure_anymal_imu_gt_offset import (
    GNSS_MISSIONS, angular_rate_from_quat, best_lag, _self_test,
)

# All IMU topics in the GrandTour zarr (main first, then the sensor-box IMUs).
IMU_TOPICS = [
    "anymal_imu", "stim320_imu", "adis_imu", "alphasense_imu",
    "livox_imu", "ap20_imu", "cpt7_imu",
]
_GT = "cpt7_ie_tc_odometry"


def measure_one(imu_grp, gt_time, gt_quat, probe_rate, max_lag_s):
    """Offset (ms, ADD convention) + peak corr aligning one IMU's gyro to GT attitude rate."""
    it = np.asarray(imu_grp["timestamp"][:], dtype=float)
    gy = np.asarray(imu_grp["ang_vel"][:], dtype=float)
    dt = 1.0 / probe_rate
    t0 = max(it[0], gt_time[0])
    t1 = min(it[-1], gt_time[-1])
    if t1 - t0 < 5.0:
        raise RuntimeError("overlap < 5 s")
    grid = np.arange(0.0, t1 - t0, dt)
    grid = grid[grid <= min(it[-1] - t0, gt_time[-1] - t0)]
    g = interp1d(it - t0, gy, axis=0, bounds_error=False, fill_value="extrapolate")(grid)
    q = Slerp(gt_time - t0, R.from_quat(gt_quat))(grid).as_quat()   # SLERP GT quat onto grid
    a = np.linalg.norm(g, axis=1)
    b = np.linalg.norm(angular_rate_from_quat(q, dt), axis=1)
    L, corr = best_lag(a, b, int(round(max_lag_s / dt)))
    return L * dt * 1e3, corr


def robust_stats(offsets, mad_k):
    """Return (kept, n_mad_dropped, stats-dict) after MAD outlier rejection."""
    off = np.asarray(offsets, dtype=float)
    med = float(np.median(off))
    mad = 1.4826 * float(np.median(np.abs(off - med)))
    keep = np.abs(off - med) <= mad_k * mad if mad > 0 else np.ones_like(off, dtype=bool)
    kept = off[keep]
    stats = {
        "mean_ms":   float(kept.mean()),
        "std_ms":    float(kept.std(ddof=1)) if len(kept) > 1 else 0.0,
        "median_ms": float(np.median(kept)),
        "min_ms":    float(kept.min()),
        "max_ms":    float(kept.max()),
        "offset_to_add_s": float(kept.mean() * 1e-3),
    }
    return kept, int((~keep).sum()), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="GrandTour root, e.g. $DATA_ROOT/grand_tour")
    ap.add_argument("--out-json", default="Config/dataset/anymal_imu_gt_offsets_per_imu.json")
    ap.add_argument("--out-txt",  default="outputs/grand_tour_dataset/imu_gt_offsets_per_imu.txt")
    ap.add_argument("--probe-rate", type=float, default=200.0)
    ap.add_argument("--max-lag-s",  type=float, default=0.05)
    ap.add_argument("--min-corr",   type=float, default=0.5)
    ap.add_argument("--mad-k",      type=float, default=3.0)
    args = ap.parse_args()

    st_L, st_c = _self_test()

    # per_mission[imu] = list of {seq, code, offset_ms, corr, used}
    per_mission = {t: [] for t in IMU_TOPICS}
    for seq_name, code, split in GNSS_MISSIONS:
        data_dir = os.path.join(args.data_root, seq_name, "data")
        try:
            root = zarr.open_group(data_dir, mode="r")
            o = root[_GT]
            gt_time = np.asarray(o["timestamp"][:], dtype=float)
            gt_quat = np.asarray(o["pose_orien"][:], dtype=float)
        except Exception as e:  # noqa: BLE001
            for t in IMU_TOPICS:
                per_mission[t].append({"seq": seq_name, "code": code, "error": f"GT: {type(e).__name__}: {e}"})
            continue
        for t in IMU_TOPICS:
            rec = {"seq": seq_name, "code": code}
            try:
                if t not in root:
                    raise RuntimeError("topic absent")
                off_ms, corr = measure_one(root[t], gt_time, gt_quat, args.probe_rate, args.max_lag_s)
                rec.update({"offset_ms": round(off_ms, 3), "corr": round(corr, 3),
                            "used": bool(corr >= args.min_corr)})
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
            per_mission[t].append(rec)

    # aggregate per IMU
    per_imu = {}
    for t in IMU_TOPICS:
        recs = per_mission[t]
        n_total = len(recs)
        n_err = sum("error" in r for r in recs)
        corr_ok = [r["offset_ms"] for r in recs if r.get("used")]
        n_low_corr = sum(("error" not in r) and (not r.get("used", False)) for r in recs)
        if not corr_ok:
            per_imu[t] = {"n_missions": n_total, "n_error": n_err, "n_low_corr": n_low_corr,
                          "n_mad_outliers": 0, "n_used": 0, "note": "no corr-gated missions"}
            continue
        kept, n_mad, stats = robust_stats(corr_ok, args.mad_k)
        per_imu[t] = {
            "n_missions": n_total, "n_error": n_err, "n_low_corr": n_low_corr,
            "n_mad_outliers": n_mad, "n_outliers_total": n_low_corr + n_err + n_mad,
            "n_used": int(len(kept)), **stats,
        }

    payload = {
        "method": "gyro-magnitude vs cpt7 GT attitude-rate cross-correlation on a uniform grid",
        "sign": "offset_ms = value to ADD to that IMU's timestamps to align onto the cpt7 GT signal",
        "note": "conflates clock skew + per-sensor gyro group delay + Inertial-Explorer GT latency; "
                "cpt7_imu (co-located with GT) is itself nonzero, so these are signal-alignment values",
        "params": {"probe_rate_hz": args.probe_rate, "max_lag_s": args.max_lag_s,
                   "min_corr": args.min_corr, "mad_k": args.mad_k, "n_gnss_missions": len(GNSS_MISSIONS)},
        "sign_self_test": {"recovered_L_samples": round(st_L, 3), "expected": 4.0, "corr": round(st_c, 3)},
        "per_imu": per_imu,
        "per_mission": per_mission,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    # text summary
    lines = ["ANYmal GrandTour per-IMU IMU<->GT offset (ADD ms to align onto cpt7 GT signal)",
             f"sign self-test: L={st_L:.3f} (want 4.0), corr={st_c:.3f}   missions={len(GNSS_MISSIONS)}", ""]
    hdr = f"{'imu':16s} {'mean_ms':>8s} {'std_ms':>7s} {'median':>7s} {'min':>7s} {'max':>7s} {'used':>5s} {'lowC':>5s} {'MAD':>4s} {'err':>4s}"
    lines += [hdr, "-" * len(hdr)]
    for t in IMU_TOPICS:
        s = per_imu[t]
        if "mean_ms" not in s:
            lines.append(f"{t:16s}  (no usable missions: {s})")
            continue
        lines.append(f"{t:16s} {s['mean_ms']:+8.3f} {s['std_ms']:7.3f} {s['median_ms']:+7.3f} "
                     f"{s['min_ms']:+7.3f} {s['max_ms']:+7.3f} {s['n_used']:5d} {s['n_low_corr']:5d} "
                     f"{s['n_mad_outliers']:4d} {s['n_error']:4d}")
    report = "\n".join(lines)
    with open(args.out_txt, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[written] {args.out_json}\n[written] {args.out_txt}")


if __name__ == "__main__":
    main()
