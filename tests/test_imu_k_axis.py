"""How the two preintegrators handle ``IMUData``'s K axis: main-body IMU only.

``IMUData`` carries an explicit K axis (K=1 for most loaders, K=5 for MIPO), and both
preintegrators read K index 0, the main-body IMU — the same choice ``evaluate.py``'s
preint baseline and ``Sequence.WindowIntegratedOrientation`` make. K>1 is indexed rather
than fused, because the realistic ways it arrives are lever-arm cases (a multi-IMU rig,
or a multi-part ``HumanoidOdom`` — ``imus.npy`` is already ``(P, T, 9)``), and IMUs
at different mount points do not measure the same specific force:

    a_k = a_ref + alpha x r_k + omega x (omega x r_k)

An unweighted mean over K therefore estimates no physical point's Δv, and dividing its
covariance by K claims independence co-located sensors (shared clock, mount, vibration,
temperature) do not have. Both failures return finite, right-shaped numbers, so
reading only the main IMU is what keeps them out of the training target.

Runs on CPU against a synthetic constant-velocity fixture; no real dataset is loaded.
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
from train_pgd    import run_preintegrator


class _StubNetwork:
    start_index = 0
    stride = 4


def _make_synth_imu(B: int = 2, N: int = 16, K: int = 1, dtype=torch.float32) -> IMUData_w_GT:
    """Synthetic IMUData_w_GT with constant velocity in the body frame.

    Body trajectory: zero rotation throughout (R_wb = identity), constant body-frame
    velocity of (1, 0, 0). The IMU sits at the body origin so acc reads
    ``-R_wb.Inv() @ g + 0 = (0, 0, +g)`` in the body frame under pypose's z-up
    convention; gyro reads zero. Constant velocity means the gravity-free Δv is
    **zero at every anchor**, which is what makes this fixture a physics check and not
    just a shape check.

    Every one of the K slots gets identical readings, so a K>1 build is the friendliest
    possible input to a fusion path. K>1 is accepted and slot 0 is INDEXED, never fused;
    that aux slots cannot influence the result is proven in
    ``tests/test_multi_imu.py::test_get_ref_delta_v_ignores_auxiliary_imus``, which
    perturbs them and asserts the Δv is unchanged.
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


def test_get_ref_delta_v_reads_the_k_axis_and_is_gravity_free():
    """K=1: the (B, N, 1, 3) layout is sliced correctly and gravity is removed.

    Constant body velocity ⇒ zero Δv at every anchor. A K-axis slice that grabbed the
    wrong axis, or a gravity sign error, both show up here as a non-zero Δv while
    staying finite and right-shaped.
    """
    data = _make_synth_imu(B=2, N=16, K=1)
    integrator = pp.module.IMUPreintegrator(prop_cov=False, reset=True).to(dtype=torch.float32)

    Dv, Dv_cov = get_ref_delta_v(integrator, data, start_index=13)

    assert Dv.shape == (2, 16, 3) and Dv_cov.shape == (2, 16, 3, 3)
    assert Dv.abs().max().item() < 1e-4, f"constant velocity must give zero Dv, got {Dv.abs().max():.3e}"
    assert torch.isfinite(Dv_cov).all()
    # Cov accumulates from anchor 0 (zero-padded) and is monotone in time.
    assert torch.equal(Dv_cov[:, 0], torch.zeros_like(Dv_cov[:, 0]))
    diag = torch.diagonal(Dv_cov, dim1=-2, dim2=-1)
    assert (diag[:, 1:] - diag[:, :-1] >= -1e-9).all()


def test_run_preintegrator_reads_the_k_axis():
    """K=1 through PGD: zero dv on constant velocity, exact dr, positive dt."""
    data = _make_synth_imu(B=2, N=16, K=1)
    integrator = pp.module.IMUPreintegrator(prop_cov=True, reset=True).to(dtype=torch.float32)

    out = run_preintegrator(_StubNetwork(), integrator, data)

    # Active slots are [0, M-1); the trailing slot is hard-padded (zeros / I_6).
    active = slice(0, out.dv.shape[1] - 1)
    assert out.dv[:, active].abs().max().item() < 1e-4
    assert out.dr[:, active].abs().max().item() < 1e-5   # identity GT rotation throughout
    assert (out.dt[:, active] > 0).all()
    assert torch.isfinite(out.dvr_cov).all()

