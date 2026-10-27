import torch
import jaxtyping as jt

from Interface.loss import I_Adapt_Loss


class Adapt_MSE_Loss(I_Adapt_Loss):
    """MSE(predicted_vel, optimized_vel): fit the network to the pose graph's
    posterior velocities. Metrics also report the residual against those targets.
    """

    def __call__(
        self,
        predicted_vel: jt.Float[torch.Tensor, "B M 3"],
        optimized_vel: jt.Float[torch.Tensor, "B M 3"],
        gt_vel       : jt.Float[torch.Tensor, "B M 3"],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = torch.nn.functional.mse_loss(
            input  = predicted_vel,
            target = optimized_vel,
        )

        with torch.no_grad():
            # Residual with the PGO pseudo-label, not vs GT. Must not be read as an error.
            # Only train/AVE is an error metric.
            pgo_vel_residual = (predicted_vel - optimized_vel).norm(dim=-1).mean()
            # DC component of the same residual. The IMU chain constrains only velocity
            # differences, so a constant offset is in its null space and the solve
            # reproduces whatever DC level the network gave it. This stays tiny no
            # matter how wrong the absolute velocity is. train/vel_dc_bias measures
            # how wrong that DC level actually is.
            # Mean over time first: meaning over the batch axis too cancels
            # independent per-window offsets at ~1/sqrt(B), which under-reported a
            # true 1 m/s offset as 0.041 at train.batch=512.
            pgo_vel_residual_dc = (predicted_vel - optimized_vel).mean(dim=1).norm(dim=-1).mean()

        metrics = {
            "train/loss"       : loss.detach().item(),
            "train/pgo_vel_residual" : pgo_vel_residual.item(),
            "train/pgo_vel_residual_dc"   : pgo_vel_residual_dc.item(),
        }
        return loss, metrics
