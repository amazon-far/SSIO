"""Stationary regularization: the stationary-sample gate and the velocity penalty on it."""

import torch
import jaxtyping as jt

from Interface.data import IMUData
from Interface.model import INetwork


class RegularizationLoss:
    """Build a static mask and apply ``weight * mean_static ||pred_v||^2``."""

    def __init__(self, gate: str = "all", speed_max: float = 0.02):
        self.gate      = gate
        self.speed_max = speed_max

    def mask(
        self,
        network  : INetwork,
        data      : IMUData | None,
        gt_vel_b : jt.Float[torch.Tensor, "B M 3"] | None = None,
    ) -> jt.Bool[torch.Tensor, "B M"]:
        """Return the static mask on the network anchor grid."""
        if self.gate == "all":
            assert data is not None
            batch, frames = data.acc.shape[0], data.acc.shape[1]
            return torch.ones(
                batch, network.anchor_count(frames),
                dtype=torch.bool, device=data.acc.device,
            )

        if self.gate == "gt":
            assert gt_vel_b is not None
            return gt_vel_b.norm(dim=-1) < self.speed_max

        stationary = None if data is None else data.stationary
        assert stationary is not None
        return network.downsample(stationary.unsqueeze(-1)).squeeze(-1)

    def __call__(
        self,
        pred_v : jt.Float[torch.Tensor, "B M 3"],
        static : jt.Bool[torch.Tensor, "B M"],
        weight : float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mean_sq_speed = (
            (pred_v.pow(2).sum(dim=-1) * static).sum()
            / static.sum().clamp(min=1)
        )
        loss = weight * mean_sq_speed
        return loss, {
            "train/regularization_loss": loss.detach().item(),
            "train/regularization_frac": float(static.to(pred_v.dtype).mean().item()),
        }
