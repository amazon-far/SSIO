"""Shared step and metric helpers for SSIO.

``optimize_step`` / ``metric_AVE`` are the two trainers' common optimizer+EMA
update and monitoring metric, kept here so the stages cannot drift apart.
``integrate_trapezoid`` / ``metric_ATE`` are the position-integration pair, also
used by ``Utils/plots.py`` and ``evaluate.py`` rather than by the trainers.
"""

import torch
import pypose as pp
import jaxtyping as jt
from torch.optim.swa_utils import AveragedModel

from Model.Network import INetwork


def optimize_step(
    optimizer : torch.optim.Optimizer,
    ema_model : AveragedModel | None,
    network   : INetwork,
) -> None:
    """Shared per-step optimizer update — identical across single / dual paths."""
    optimizer.step()
    if ema_model is not None:
        ema_model.update_parameters(network)
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def metric_AVE(
    ref_rot_Twb: jt.Float[pp.LieTensor, "B M 3"],
    pred_v     : jt.Float[torch.Tensor, "B M 3"],
    gt_vel_w_M : jt.Float[torch.Tensor, "B M 3"],
) -> float:
    """**AVE** — Absolute Velocity Error: mean ‖pred_v (lifted to world) − gt_v‖.

    This is *the* AVE, and the only GT-referenced velocity metric in the training
    loop (alongside ``metric_vel_dc_bias`` below, which is its DC part). Report
    this one.

    Same per-batch formula as ``evaluate.py``'s ``AVE``, but the trainer means over
    batches and ``evaluate.py`` over sequences, so the two diverge once a sequence
    exceeds the eval loader's 256-window batch.

    Do not confuse it with ``L2norm_vel_dv_centered`` / ``L2norm_vel_dv`` from
    ``Loss/ssio.py``: those are norms of the residual against the *preintegration
    Δv reference*, not against GT, and ``_centered`` is additionally mean-centred
    per window so it cannot see the constant velocity offset. They track what the
    loss optimizes; only AVE tracks accuracy, and ``metric_vel_dc_bias`` below
    isolates the offset AVE and the centred residual each half-miss.

    PGD's counterpart residual is ``pgo_vel_residual`` / ``pgo_vel_residual_dc``
    in ``Loss/adapt.py`` — same idea against the PGO pseudo-label instead of Δv.

    Detached / no-grad: not used in supervision — the SSIO loss operates on Δv
    residuals, which are invariant to the unobservable absolute velocity offset.
    ``gt_vel_w_M`` must already be downsampled onto the network's M anchors.
    """
    pred_world = ref_rot_Twb.Exp() @ pred_v                 # (B, M, 3) world frame
    return (pred_world - gt_vel_w_M).norm(dim=-1).mean().item()


def metric_vel_dc_bias(
    ref_rot_Twb: jt.Float[pp.LieTensor, "B M 3"],
    pred_v     : jt.Float[torch.Tensor, "B M 3"],
    gt_vel_w_M : jt.Float[torch.Tensor, "B M 3"],
) -> float:
    """**DC velocity bias** — the constant part of the velocity error, vs GT.

    Mean of ``pred_v (lifted to world) - gt_v`` over TIME FIRST within each window,
    then the norm, then the mean over the batch. This and AVE split the error into
    the part unobservable to an offset-blind loss and the rest (AVE has both):

        AVE^2  ~  dc_bias^2 + (time-varying "shape" dv-loss-observable term)^2
    """
    pred_world = ref_rot_Twb.Exp() @ pred_v                 # (B, M, 3) world frame
    return (pred_world - gt_vel_w_M).mean(dim=1).norm(dim=-1).mean().item()


def integrate_trapezoid(v: torch.Tensor, t_ns: torch.Tensor) -> torch.Tensor:
    """Trapezoidal cumulative integration of a (T, 3) velocity series.

    Returns fp64 position (T, 3) anchored at the origin, absolute initial position
    being unobservable in SSIO.
    """
    assert bool(torch.all(t_ns[1:] > t_ns[:-1]))

    t_s  = (t_ns - t_ns[0]).to(torch.float64) * 1.0e-9
    dt   = (t_s[1:] - t_s[:-1]).unsqueeze(-1)                                          # (T-1, 1)
    v64  = v.to(torch.float64)
    incs = 0.5 * (v64[:-1] + v64[1:]) * dt                                             # (T-1, 3)
    return torch.cat([v64.new_zeros(1, 3), torch.cumsum(incs, dim=0)], dim=0)


@torch.no_grad()
def metric_ATE(
    pred_vel_w: jt.Float[torch.Tensor, "T 3"],
    gt_vel_w  : jt.Float[torch.Tensor, "T 3"],
    t_ns      : jt.Int[torch.Tensor, "T"],
) -> tuple[float, float]:
    """**ATE** — Absolute Trajectory Error, returned as ``(ate, ate_mean)``.

    RMSE and mean of ‖pred_pos − gt_pos‖ over origin-anchored trapezoidal
    integrals of world-frame velocity, with no SE3/Umeyama alignment. Same
    construction as ``evaluate.py``'s ``compute_metrics``.

    Takes one whole sequence, flat: the integral is cumulative, so pass the
    concatenated series and never the NaN-separated ``*_plot`` copies.
    """
    pred_pos = integrate_trapezoid(pred_vel_w, t_ns)
    gt_pos   = integrate_trapezoid(gt_vel_w,   t_ns)
    pos_err  = (pred_pos - gt_pos).norm(dim=-1)
    return (
        float(pos_err.pow(2).mean().sqrt().item()),   # RMSE of position-error norms
        float(pos_err.mean().item()),                 # mean L2 position error
    )
