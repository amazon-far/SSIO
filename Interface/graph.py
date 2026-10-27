import torch
import jaxtyping as jt

from .base import Base


class I_IMUFactorGraph(Base, torch.nn.Module):
    def solve(self) -> dict: ...

    @property
    def get_state(self) -> jt.Float[torch.Tensor, "B N 3"]: ...
