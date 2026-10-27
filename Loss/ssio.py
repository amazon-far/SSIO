import torch
import pypose as pp
import jaxtyping as jt

from typing import Literal

from Interface.loss import I_SSIO_Loss

# Integration self-supervision residual metrics are named for the target they measure against:
# ``L2norm_vel_dv{,_centered}`` is ‖pred_world - ref_world‖ against the
# *preintegration Δv teacher*, not GT. It is a norm, not an error — only
# ``train/AVE`` (Utils.trainer.metric_AVE) and ``train/vel_dc_bias``
# (metric_vel_dc_bias) are GT-sourced. ``_centered`` subtracts the per-window
# mean and is therefore structurally blind to a constant velocity offset; the
# un-centred variant keeps it.


def _check_dv_fits(dv: list[tuple[int, float]], M: int) -> None:
    """Reject a window too short for every configured dv step.

    Otherwise the per-scale loop skips every step and the loss is a constant zero
    with requires_grad=False: backward() raises, or under HybridSSIOLoss the anchor
    term alone trains while the run still claims to be SSIO.
    """
    if not any(0 < step < M for step, _ in dv):
        raise ValueError(
            f"none of dv={[s for s, _ in dv]} fits the {M}-anchor window "
            f"(need 0 < step < M); raise train.window or lower the dv steps."
        )


def _parse_dv(dv: list[tuple[int, float]]) -> list[tuple[int, float]]:
    parsed: list[tuple[int, float]] = []
    for step, weight in dv:
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(f"dv step must be a positive integer, got {step!r}")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or weight < 0.0
        ):
            raise ValueError(f"dv weight must be >= 0, got {weight!r}")
        parsed.append((step, float(weight)))
    return parsed


class VarianceLoss(I_SSIO_Loss):
    """Variance-based SSIO loss, the dense limit of FixedDvLoss."""

    MODE = Literal["identity", "preint"]

    def __init__(self, weight: MODE):
        if weight not in ("identity", "preint"):
            raise ValueError(f"Unsupported weight mode {weight!r}")
        self.weight : VarianceLoss.MODE = weight

    def __call__(
        self,
        ref_rot_Twb     : jt.Float[pp.LieTensor, "B M 3"],
        ref_delta_v     : jt.Float[torch.Tensor, "B M 3"],
        ref_delta_v_cov : jt.Float[torch.Tensor, "B M 3 3"],
        pred_v          : jt.Float[torch.Tensor, "B M 3"],
        pred_v_cov      : jt.Float[torch.Tensor, "B M 3 3"],
        gt_vel_b        : jt.Float[torch.Tensor, "B M 3"],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        r"""SSIO variance Δv loss:

            e[m] = pred_world[m] - ref_world[m],   loss = Var_m(e[m])

        R at anchor 0 is used as the R_0 proxy for ref_delta_v (body-frame-at-frame-0).
        "preint" mode weights anchors by inverse trace(Σ_preint).
        """
        R          = ref_rot_Twb.Exp()                  # so(3) → SO3
        pred_world = R @ pred_v                         # body→world
        ref_world  = R[:, 0:1] @ ref_delta_v            # R[anchor 0] as R_0 proxy

        e    = pred_world - ref_world

        match self.weight:
            case "identity":
                loss = e.var(dim=1, unbiased=False).mean()

            case "preint":
                # Inverse-trace(Σ_preint) precision weights (trace is rotation-invariant);
                # 1e-6 floor handles Σ=0 at anchor 0. Weighted mean keeps offset-invariance.
                precision = 1.0 / (torch.diagonal(ref_delta_v_cov, dim1=-2, dim2=-1).sum(-1) + 1e-6)  # (B, M)
                w         = precision.unsqueeze(-1)                                                    # (B, M, 1)
                w_sum     = w.sum(dim=1, keepdim=True)                                                 # (B, 1, 1)
                mu_w      = (w * e).sum(dim=1, keepdim=True) / w_sum                                   # (B, 1, 3)
                loss      = ((w * (e - mu_w).pow(2)).sum(dim=1) / w_sum.squeeze(1)).mean()

        with torch.no_grad():
            e_centered             = e - e.mean(dim=1, keepdim=True)
            l2norm_vel_dv_centered = e_centered.norm(dim=-1).mean()
            l2norm_vel_dv          = e.norm(dim=-1).mean()

        metrics = {
            "train/loss"                   : loss.detach().item(),
            "train/L2norm_vel_dv_centered" : l2norm_vel_dv_centered.item(),
            "train/L2norm_vel_dv"          : l2norm_vel_dv.item(),
        }
        return loss, metrics


class SupervisedVelocityLoss(I_SSIO_Loss):
    r"""Supervised AirIO baseline: GT body-frame velocity loss plus AirIO's
    Gaussian-NLL covariance term. Uses velocity labels; not self-supervised.

        loss = vel_weight * vel_loss(pred_v, gt_vel_b) + cov_weight * NLL(pred_v, pred_v_cov, gt_vel_b)
    """

    def __init__(
        self,
        vel_weight : float,
        vel_loss   : Literal["L2", "L1", "Huber"] = "L2",
        huber_delta: float = 1.0,
        cov_weight : float = 0.0,
        cov_floor  : float = 1.0e-6,
        covaug     : bool  = False,
    ) -> None:
        super().__init__()
        if vel_weight < 0.0:
            raise ValueError(f"vel_weight must be >= 0, got {vel_weight}")
        if cov_weight < 0.0:
            raise ValueError(f"cov_weight must be >= 0, got {cov_weight}")
        if cov_floor <= 0.0:
            raise ValueError(f"cov_floor must be > 0, got {cov_floor}")
        if huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {huber_delta}")
        if vel_loss not in ("L2", "L1", "Huber"):
            raise ValueError(f"Unsupported vel_loss {vel_loss!r}")
        self.vel_weight  = float(vel_weight)
        self.vel_loss    = vel_loss
        self.huber_delta = float(huber_delta)
        self.cov_weight  = float(cov_weight)
        self.cov_floor   = float(cov_floor)
        self.covaug      = bool(covaug)

    def __call__(
        self,
        ref_rot_Twb     : jt.Float[pp.LieTensor, "B M 3"],
        ref_delta_v     : jt.Float[torch.Tensor, "B M 3"],
        ref_delta_v_cov : jt.Float[torch.Tensor, "B M 3 3"],
        pred_v          : jt.Float[torch.Tensor, "B M 3"],
        pred_v_cov      : jt.Float[torch.Tensor, "B M 3 3"],
        gt_vel_b        : jt.Float[torch.Tensor, "B M 3"],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # No preintegration teacher on this path, so no L2norm_vel_dv* metrics.
        match self.vel_loss:
            case "L2":
                vel_loss = torch.nn.functional.mse_loss(pred_v, gt_vel_b)
            case "L1":
                vel_loss = torch.nn.functional.l1_loss(pred_v, gt_vel_b)
            case "Huber":
                vel_loss = torch.nn.functional.huber_loss(pred_v, gt_vel_b, delta=self.huber_delta)
        total_loss = self.vel_weight * vel_loss

        # Diagonal Gaussian NLL on the residual, detached unless ``covaug`` so by
        # default it trains only the covariance head.
        var  = torch.diagonal(pred_v_cov, dim1=-2, dim2=-1).clamp(min=self.cov_floor)
        dist = pred_v - gt_vel_b
        if not self.covaug:
            dist = dist.detach()
        cov_loss = ((dist.square() / var) + var.log()).mean()
        if self.cov_weight > 0.0:
            total_loss = total_loss + self.cov_weight * cov_loss

        metrics: dict[str, float] = {
            "train/vel_loss": vel_loss.detach().item(),
            "train/cov_loss": cov_loss.detach().item(),
            "train/loss"    : total_loss.detach().item(),
        }
        return total_loss, metrics


class FixedDvLoss(I_SSIO_Loss):
    def __init__(self, dv: list[tuple[int, float]]):
        super().__init__()
        self.dv = _parse_dv(dv)

    def __call__(
        self,
        ref_rot_Twb     : jt.Float[pp.LieTensor, "B M 3"],
        ref_delta_v     : jt.Float[torch.Tensor, "B M 3"],
        ref_delta_v_cov : jt.Float[torch.Tensor, "B M 3 3"],
        pred_v          : jt.Float[torch.Tensor, "B M 3"],
        pred_v_cov      : jt.Float[torch.Tensor, "B M 3 3"],
        gt_vel_b        : jt.Float[torch.Tensor, "B M 3"],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Discrete multi-scale Δv loss (sparse baseline; VarianceLoss is the dense limit):

            de[k] = e[k+s] - e[k],   e[k] = pred_world[k] - ref_world[k]
            loss  = Σ_s weight_s · mean(de_s²)

        Differencing removes the unobservable offset. pred_v_cov unused.
        """
        R          = ref_rot_Twb.Exp()                  # SO3 [B, M]
        pred_world = R @ pred_v                         # (B, M, 3)
        ref_world  = R[:, 0:1] @ ref_delta_v            # (B, M, 3)
        e          = pred_world - ref_world             # (B, M, 3)

        M = e.shape[1]
        _check_dv_fits(self.dv, M)
        loss: torch.Tensor = e.new_zeros(())
        metrics: dict[str, float] = {}
        for step, weight in self.dv:
            if step >= M:
                continue
            de     = e[:, step:] - e[:, :-step]         # (B, M-step, 3)
            loss_s = de.pow(2).mean()
            loss   = loss + weight * loss_s
            metrics[f"train/loss_dv{step}"] = loss_s.detach().item()

        with torch.no_grad():
            e_centered             = e - e.mean(dim=1, keepdim=True)
            l2norm_vel_dv_centered = e_centered.norm(dim=-1).mean()
            l2norm_vel_dv          = e.norm(dim=-1).mean()

        metrics["train/loss"]                   = loss.detach().item()
        metrics["train/L2norm_vel_dv_centered"] = l2norm_vel_dv_centered.item()
        metrics["train/L2norm_vel_dv"]          = l2norm_vel_dv.item()
        return loss, metrics
