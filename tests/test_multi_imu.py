"""Shape + numerical sanity for the multi-IMU K axis.

Two properties must hold after the multi-IMU wiring:

1. **K=1 transparency** — every per-IMU codepath emits the same shapes /
   tensors a single-IMU pipeline would have — the K=1 path must stay
   byte-compatible.

2. **Main-IMU teacher** — SSIO and PGO preintegration use only IMU index 0.
   Adding or perturbing auxiliary IMUs must not change ``Dv`` or its covariance.

All tests run on CPU against a synthetic constant-velocity IMUData fixture;
no real dataset is loaded.
"""
from __future__ import annotations

import os
import sys

import torch
import pypose as pp

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Interface.data       import IMUData_w_GT
from train_ssio     import get_ref_delta_v


def _make_synth_imu(B: int = 2, N: int = 16, K: int = 1, dtype=torch.float32) -> IMUData_w_GT:
    """Synthetic IMUData_w_GT with constant velocity in the body frame.

    Body trajectory: zero rotation throughout (R_wb = identity), constant
    body-frame velocity of (1, 0, 0). The IMU sits at the body origin so
    acc reads ``-R_wb.Inv() @ g + 0 = (0, 0, +g)`` in the body frame under
    pypose's z-up convention; gyro reads zero.

    All K IMUs initially receive identical readings.
    """
    dt_s = 0.005  # 200 Hz
    time_ns = (
        (torch.arange(N, dtype=torch.float64) * dt_s * 1e9)
        .round().to(torch.int64)
    )                                                         # (N,)
    time_ns = time_ns.reshape(1, N, 1, 1).expand(B, N, K, 1).contiguous()

    g = 9.81
    acc_b  = torch.tensor([0.0, 0.0, g], dtype=dtype)
    gyro_b = torch.zeros(3, dtype=dtype)
    acc  = acc_b .reshape(1, 1, 1, 3).expand(B, N, K, 3).contiguous()
    gyro = gyro_b.reshape(1, 1, 1, 3).expand(B, N, K, 3).contiguous()

    eye3 = torch.eye(3, dtype=dtype)
    acc_cov  = (eye3 * 1.0e-4).expand(B, N, K, 3, 3).contiguous()
    gyro_cov = (eye3 * 1.0e-5).expand(B, N, K, 3, 3).contiguous()

    # Body trajectory: identity rotation, position growing along +x at 1 m/s.
    R_id = pp.identity_SO3(B, N).to(dtype=dtype).tensor()      # (B, N, 4)
    pos  = torch.zeros(B, N, 3, dtype=dtype)
    pos[..., 0] = torch.arange(N, dtype=dtype) * dt_s
    gt_poses = pp.SE3(torch.cat([pos, R_id], dim=-1))          # (B, N, 7)
    gt_vel_w = torch.zeros(B, N, 3, dtype=dtype)
    gt_vel_w[..., 0] = 1.0                                      # constant +x

    return IMUData_w_GT(
        time_ns        = time_ns,
        acc            = acc,
        gyro           = gyro,
        acc_cov        = acc_cov,
        gyro_cov       = gyro_cov,
        gt_poses       = gt_poses,
        gt_vel_w       = gt_vel_w,
        # No correction stage in the synthetic fixture — corrected streams ARE the raw IMU.
        corrected_acc  = acc,
        corrected_gyro = gyro,
    )


def test_get_ref_delta_v_k1_shape():
    """K=1 pipeline emits (B, N, 3) / (B, N, 3, 3) — the single-IMU shapes."""
    data = _make_synth_imu(B=2, N=16, K=1)
    integrator = pp.module.IMUPreintegrator(prop_cov=False, reset=True).to(dtype=torch.float32)
    Dv, Dv_cov = get_ref_delta_v(integrator, data, start_index=13)
    assert Dv.shape     == (2, 16, 3),    f"Dv shape mismatch: {tuple(Dv.shape)}"
    assert Dv_cov.shape == (2, 16, 3, 3), f"Dv_cov shape mismatch: {tuple(Dv_cov.shape)}"
    assert torch.isfinite(Dv).all(),     "Dv has non-finite entries"
    assert torch.isfinite(Dv_cov).all(), "Dv_cov has non-finite entries"


def test_get_ref_delta_v_ignores_auxiliary_imus():
    """Adding or perturbing auxiliary IMUs leaves the main-IMU teacher unchanged."""
    K_DUP = 4
    data_k1 = _make_synth_imu(B=2, N=16, K=1)
    data_kn = _make_synth_imu(B=2, N=16, K=K_DUP)
    data_kn.corrected_acc[:, :, 1:] += 100.0
    data_kn.corrected_gyro[:, :, 1:] += 10.0
    data_kn.acc_cov[:, :, 1:] *= 1000.0

    integrator = pp.module.IMUPreintegrator(prop_cov=False, reset=True).to(dtype=torch.float32)
    Dv1, Cov1 = get_ref_delta_v(integrator, data_k1, start_index=13)
    Dvk, Covk = get_ref_delta_v(integrator, data_kn, start_index=13)

    dv_diff = (Dv1 - Dvk).abs().max().item()
    assert dv_diff < 1e-5, f"auxiliary IMUs changed Dv: max |.| = {dv_diff:.3e}"

    cov_diff = (Cov1 - Covk).abs().max().item()
    assert cov_diff < 1e-5, f"auxiliary IMUs changed Dv_cov: max |.| = {cov_diff:.3e}"


def test_run_preintegrator_k1_shape():
    """train_pgd.run_preintegrator preserves (B, M, ...) shapes at K=1."""
    # run_preintegrator needs an INetwork with .start_index and .stride. We
    # don't have a concrete network here, so we stub one.
    class _StubNetwork:
        start_index = 0
        stride = 4

    from train_pgd import run_preintegrator
    integrator = pp.module.IMUPreintegrator(prop_cov=True, reset=True).to(dtype=torch.float32)
    data = _make_synth_imu(B=2, N=16, K=1)
    out = run_preintegrator(_StubNetwork(), integrator, data)
    # M = (N - 1 - s_idx) // stride + 1 = (15 - 0) // 4 + 1 = 3 + 1 = 4
    assert out.dv     .shape == (2, 4, 3),    f"dv shape: {tuple(out.dv.shape)}"
    assert out.dr     .shape == (2, 4, 3),    f"dr shape: {tuple(out.dr.shape)}"
    assert out.dt     .shape == (2, 4, 1),    f"dt shape: {tuple(out.dt.shape)}"
    assert out.dvr_cov.shape == (2, 4, 6, 6), f"dvr_cov shape: {tuple(out.dvr_cov.shape)}"
    for field in (out.dv, out.dr, out.dt, out.dvr_cov):
        assert torch.isfinite(field).all(), "non-finite entries in run_preintegrator output"


def test_run_preintegrator_ignores_auxiliary_imus():
    """Stage-two preintegration reads only IMU index 0."""
    K_DUP = 4
    class _StubNetwork:
        start_index = 0
        stride = 4

    from train_pgd import run_preintegrator
    integrator = pp.module.IMUPreintegrator(prop_cov=True, reset=True).to(dtype=torch.float32)
    data_k1 = _make_synth_imu(B=2, N=16, K=1)
    data_kn = _make_synth_imu(B=2, N=16, K=K_DUP)
    data_kn.corrected_acc[:, :, 1:] += 100.0
    data_kn.corrected_gyro[:, :, 1:] += 10.0
    out1 = run_preintegrator(_StubNetwork(), integrator, data_k1)
    outk = run_preintegrator(_StubNetwork(), integrator, data_kn)
    # Active slots are [0, M-1) — the trailing M-1 slot is hard-padded with
    # zeros (dv/dr/dt) or I_6 (cov), independent of K. Compare only active.
    active = slice(0, out1.dv.shape[1] - 1)
    dv_diff  = (out1.dv [:, active] - outk.dv [:, active]).abs().max().item()
    dr_diff  = (out1.dr [:, active] - outk.dr [:, active]).abs().max().item()
    dt_diff  = (out1.dt [:, active] - outk.dt [:, active]).abs().max().item()
    cov_diff = (out1.dvr_cov[:, active] - outk.dvr_cov[:, active]).abs().max().item()
    assert dv_diff  < 1e-5, f"auxiliary IMUs changed dv: {dv_diff:.3e}"
    assert dr_diff  < 1e-5, f"auxiliary IMUs changed dr: {dr_diff:.3e}"
    assert dt_diff  < 1e-5, f"auxiliary IMUs changed dt: {dt_diff:.3e}"
    assert cov_diff < 1e-5, f"auxiliary IMUs changed covariance: {cov_diff:.3e}"
