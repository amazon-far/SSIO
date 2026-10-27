"""Chi-square outlier gate on the PGO velocity observations (obs_reject_chi2).

Ungated, the solver has breakdown point 0: a single bad velocity observation
drags the whole window. ``obs_reject_chi2: null`` (the shipped
default) must keep that behaviour untouched; a threshold must contain it.
"""

import torch
import pypose as pp

from Optimizer.direct_pgo import Dense_IMUPGOGraph

CHI2_3DOF_P01 = 11.34


def _make(outlier: float = 0.0, bad: tuple[int, ...] = (7,), B: int = 2, N: int = 16):
    """Window whose velocity observations are exactly IMU-consistent, plus outliers."""
    torch.manual_seed(0)
    vel = torch.cumsum(torch.randn(B, N, 3) * 0.02, dim=1)
    dv  = torch.cat([vel[:, 1:] - vel[:, :-1], torch.zeros(B, 1, 3)], dim=1)
    obs = vel.clone()
    for k in bad:
        obs[:, k] += outlier
    kwargs = dict(
        net_vel_b = obs,
        net_vel_cov_b = torch.eye(3).expand(B, N, 3, 3).clone() * 1e-4,
        imu_dv_b      = dv,
        imu_dr_b      = torch.zeros(B, N, 3),
        imu_dt_ms        = torch.full((B, N, 1), 5.0),
        imu_dvr_cov   = torch.eye(6).expand(B, N, 6, 6).clone() * 1e-6,
        rot_wb        = pp.so3(torch.zeros(B, 1, 3)),
    )
    return kwargs, vel


def test_gate_off_is_a_no_op():
    """Default (null) must reproduce the ungated solution exactly."""
    kw, truth = _make(outlier=10.0)
    off = Dense_IMUPGOGraph(**kw, obs_reject_chi2=None)
    info = off.solve()
    assert info["rejected"] == 0
    err = (off.get_state - truth).norm(dim=-1).max().item()
    assert err > 1.0, f"expected the ungated solver to be dragged by the outlier, got {err}"


def test_gate_recovers_under_a_gross_outlier():
    for outlier in (1.0, 10.0, 100.0):
        kw, truth = _make(outlier=outlier)
        g = Dense_IMUPGOGraph(**kw, obs_reject_chi2=CHI2_3DOF_P01)
        info = g.solve()
        err = (g.get_state - truth).norm(dim=-1).max().item()
        assert info["rejected"] > 0, f"outlier {outlier} was not rejected"
        assert err < 1e-3, f"outlier {outlier}: residual error {err} after rejection"


def test_gate_leaves_consistent_data_alone():
    """No outlier => nothing rejected, and the solution is unchanged."""
    kw, truth = _make(outlier=0.0)
    on = Dense_IMUPGOGraph(**kw, obs_reject_chi2=CHI2_3DOF_P01)
    info = on.solve()
    assert info["rejected"] == 0
    assert (on.get_state - truth).norm(dim=-1).max().item() < 1e-5


def test_gate_keeps_the_gauge():
    """Even at an absurd threshold the window keeps >= 2 observations, so H stays
    non-singular and no window falls back to its prior."""
    kw, _ = _make(outlier=100.0)
    g = Dense_IMUPGOGraph(**kw, obs_reject_chi2=1e-12)   # everything violates
    info = g.solve()
    kept = int((g._obs_mask > 0).sum(dim=1).min())
    assert kept >= 2, f"window stripped to {kept} observations"
    assert not bool(info["failed"].any()), "factorization failed after rejection"


def test_gate_rejects_exactly_the_injected_outliers():
    """Precision, not just recall: collateral rejections are a real cost."""
    for bad in ((7,), (3, 7, 11), (1, 5, 9, 20, 33)):
        kw, truth = _make(outlier=50.0, bad=bad, B=1, N=56)
        g = Dense_IMUPGOGraph(**kw, obs_reject_chi2=CHI2_3DOF_P01)
        info = g.solve()
        rejected = tuple(sorted((g._obs_mask[0] == 0).nonzero().flatten().tolist()))
        assert rejected == tuple(sorted(bad)), f"injected {bad}, gate removed {rejected}"
        assert info["rejected"] == len(bad)   # B=1, so the count is per-window
        assert (g.get_state - truth).norm(dim=-1).max().item() < 1e-6


def test_gate_off_is_bit_identical_to_no_gate():
    """`obs_reject_chi2: null` must reproduce the pre-gate solver exactly, not merely
    behave similarly — it is the shipped default in both graph configs."""
    kw_a, _ = _make(outlier=10.0)
    kw_b, _ = _make(outlier=10.0)
    gated_off = Dense_IMUPGOGraph(**kw_a, obs_reject_chi2=None)
    no_kwarg  = Dense_IMUPGOGraph(**kw_b)          # never passes the argument at all
    gated_off.solve(); no_kwarg.solve()
    assert torch.equal(gated_off.get_state, no_kwarg.get_state)
    assert torch.equal(gated_off.weight_matrix(), no_kwarg.weight_matrix())
