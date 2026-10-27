import torch

import pypose as pp
import jaxtyping as jt

from .base import Base


class I_SSIO_Loss(Base):
    def __call__(
        self,
        ref_rot_Twb     : jt.Float[pp.LieTensor, "B M 3"],
        ref_delta_v     : jt.Float[torch.Tensor, "B M 3"],
        ref_delta_v_cov : jt.Float[torch.Tensor, "B M 3 3"],
        pred_v          : jt.Float[torch.Tensor, "B M 3"],
        pred_v_cov      : jt.Float[torch.Tensor, "B M 3 3"],
        gt_vel_b        : jt.Float[torch.Tensor, "B M 3"],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ...


class I_Adapt_Loss(Base):
    def __call__(
        self,
        predicted_vel: jt.Float[torch.Tensor, "B M 3"],
        optimized_vel: jt.Float[torch.Tensor, "B M 3"],
        gt_vel       : jt.Float[torch.Tensor, "B M 3"],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ...
