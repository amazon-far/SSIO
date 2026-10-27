"""Side-by-side correctness tests for Sparse_IMUPGOGraph vs Dense_IMUPGOGraph.

The load-bearing test is "matches dense solution on noise-free fixture": both
optimizers receive identical inputs and we assert their final body-frame
velocity and rotation states agree to ~1e-3.
"""

from __future__ import annotations

import sys
import os
import pytest
import torch
import pypose as pp

# Make the repo root importable regardless of pytest's rootdir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Optimizer.direct_pgo import Dense_IMUPGOGraph
from Optimizer.sparse_pgo import Sparse_IMUPGOGraph


# The sparse backend (and every dense-vs-sparse comparison) needs CUDA.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sparse PGO requires CUDA")


def _make_synthetic(B=2, N=6, seed=42, device="cuda:0", dtype=torch.float32):
    """Build a noise-free fixture: pick a true (R, v) trajectory, derive
    consistent IMU dr/dv from it.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = 0.05  # 20 Hz keyframes

    # True relative rotation per segment.
    true_dr = torch.randn(B, N, 3, generator=g, dtype=dtype) * 0.05
    R_chain = [pp.identity_SO3(B, 1).to(dtype=dtype)]
    for k in range(N - 1):
        Rk = R_chain[-1] @ pp.so3(true_dr[:, k:k+1]).Exp()
        R_chain.append(Rk)
    R_true = pp.SO3(torch.cat([Rk.tensor() for Rk in R_chain], dim=1))   # (B, N)

    # True body-frame velocity.
    v_true = torch.randn(B, N, 3, generator=g, dtype=dtype) * 0.5

    # Derive IMU dv such that PGO IMU residual is exactly zero on (R_true, v_true).
    Ri = R_true[:, :-1]
    Rj = R_true[:, 1:]
    vi = v_true[:, :-1]
    vj = v_true[:, 1:]
    dv_true = (Ri.Inv() @ Rj) @ vj - vi
    dv_pad = torch.cat([dv_true, torch.zeros(B, 1, 3, dtype=dtype)], dim=1)
    dr_pad = torch.cat([true_dr[:, :-1], torch.zeros(B, 1, 3, dtype=dtype)], dim=1)

    dt_ms = torch.full((B, N, 1), dt * 1000.0, dtype=dtype)

    imu_dvr_cov = torch.eye(6, dtype=dtype).expand(B, N, 6, 6).contiguous() * 1e-4
    net_vel_cov = torch.eye(3, dtype=dtype).expand(B, N, 3, 3).contiguous() * 1e-2

    # Anchor rotation = identity (since first frame in the chain is identity).
    rot_wb = torch.zeros(B, 1, 3, dtype=dtype)

    # Body-frame initial velocity prior on KF 0 — deliberately offset from the
    # IMU-consistent / observation solution so the factor visibly pulls v_0.
    vel_prior = v_true[:, 0:1] + 0.3

    # Move all to device.
    return {
        "v_true":      v_true.to(device),
        "R_true":      R_true.to(device=device),
        "imu_dv_b":    dv_pad.to(device),
        "imu_dr_b":    dr_pad.to(device),
        "imu_dt_ms":      dt_ms.to(device),
        "imu_dvr_cov": imu_dvr_cov.to(device),
        "net_vel_cov": net_vel_cov.to(device),
        "rot_wb":      pp.so3(rot_wb).to(device),
        "vel_prior":   vel_prior.to(device),
    }


@requires_cuda
def test_match_dense_noise_free():
    """Load-bearing test: same inputs → same optimized state to ~1e-3."""
    fx = _make_synthetic(B=3, N=8)
    # Add small observation noise.
    noise = torch.randn_like(fx["v_true"]) * 0.05
    net_v = fx["v_true"] + noise

    common = dict(
        net_vel_b=net_v,
        net_vel_cov_b=fx["net_vel_cov"],
        imu_dv_b=fx["imu_dv_b"],
        imu_dr_b=fx["imu_dr_b"],
        imu_dt_ms=fx["imu_dt_ms"],
        imu_dvr_cov=fx["imu_dvr_cov"],
        rot_wb=fx["rot_wb"],
        max_iter=20,
    )

    dense  = Dense_IMUPGOGraph (**common).to("cuda:0")
    sparse = Sparse_IMUPGOGraph(**common)

    info_d = dense.solve()
    info_s = sparse.solve()

    print(f"[dense]  iters={info_d['iters']}  final_loss={info_d['final_loss']:.4e}")
    print(f"[sparse] iters={info_s['iters']}  final_loss={info_s['final_loss']:.4e}")

    diff_v = (dense.get_state - sparse.get_state).abs().max().item()
    print(f"[match] max |v_dense - v_sparse| = {diff_v:.3e}")
    assert diff_v < 1e-3, f"velocity disagreement {diff_v}"

    diff_R = (dense.get_rotation.tensor() - sparse.get_rotation.tensor()).abs().max().item()
    print(f"[match] max |R_dense - R_sparse| = {diff_R:.3e}")
    assert diff_R < 1e-3, f"rotation disagreement {diff_R}"


@requires_cuda
def test_recovers_truth_under_noise():
    fx = _make_synthetic(B=2, N=8)
    net_v = fx["v_true"] + torch.randn_like(fx["v_true"]) * 0.05

    g = Sparse_IMUPGOGraph(
        net_vel_b=net_v,
        net_vel_cov_b=fx["net_vel_cov"],
        imu_dv_b=fx["imu_dv_b"],
        imu_dr_b=fx["imu_dr_b"],
        imu_dt_ms=fx["imu_dt_ms"],
        imu_dvr_cov=fx["imu_dvr_cov"],
        rot_wb=fx["rot_wb"],
        max_iter=20,
    )
    info = g.solve()
    print(f"[noise] iters={info['iters']}  final_loss={info['final_loss']:.4e}")

    v_err = (g.get_state - fx["v_true"]).norm(dim=-1).mean(dim=-1)
    print(f"[noise] mean |v_opt - v_true| per batch: {v_err.tolist()}")
    assert (v_err < 0.06).all(), f"convergence too loose: {v_err.tolist()}"


@requires_cuda
def test_match_dense_with_vel_prior():
    """Dense vs sparse must still agree once the initial velocity prior is on.

    Same load-bearing comparison as test_match_dense_noise_free, but both graphs
    receive a (B, 1, 3) vel_prior at KF 0. Verifies the vel-prior factor is
    assembled identically into H/g on both backends.
    """
    fx = _make_synthetic(B=3, N=8)
    net_v = fx["v_true"] + torch.randn_like(fx["v_true"]) * 0.05

    common = dict(
        net_vel_b=net_v,
        net_vel_cov_b=fx["net_vel_cov"],
        imu_dv_b=fx["imu_dv_b"],
        imu_dr_b=fx["imu_dr_b"],
        imu_dt_ms=fx["imu_dt_ms"],
        imu_dvr_cov=fx["imu_dvr_cov"],
        rot_wb=fx["rot_wb"],
        vel_prior=fx["vel_prior"],
        max_iter=20,
    )

    dense  = Dense_IMUPGOGraph (**common).to("cuda:0")
    sparse = Sparse_IMUPGOGraph(**common)
    dense.solve()
    sparse.solve()

    # forward() residual vector carries the extra 3-dim vel-prior block.
    expected_D = 6 * (8 - 1) + 3 * 8 + 3 + 3
    assert dense.forward().shape == (3, expected_D)
    assert sparse.forward().shape == (3, expected_D)

    v_d, v_s = dense.get_state, sparse.get_state
    diff_v = (v_d - v_s).abs().max().item()
    print(f"[vp-match] max |v_dense - v_sparse| = {diff_v:.3e}")
    assert diff_v < 1e-3, f"velocity disagreement {diff_v}"

    # The strong prior (default 1e6) should pull v_0 close to vel_prior on both.
    vp = fx["vel_prior"][:, 0]
    err_d = (v_d[:, 0] - vp).abs().max().item()
    err_s = (v_s[:, 0] - vp).abs().max().item()
    print(f"[vp-match] |v0 - prior| dense={err_d:.3e} sparse={err_s:.3e}")
    assert err_d < 1e-3 and err_s < 1e-3, f"prior not enforced (dense {err_d}, sparse {err_s})"

    diff_R = (dense.get_rotation.tensor() - sparse.get_rotation.tensor()).abs().max().item()
    print(f"[vp-match] max |R_dense - R_sparse| = {diff_R:.3e}")
    assert diff_R < 1e-3, f"rotation disagreement {diff_R}"
