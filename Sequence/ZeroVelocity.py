"""Sequence-level, IMU-only stationary mask."""

import dataclasses
import typing as tp

import torch
from omegaconf import DictConfig

from Interface.sequence import ISequence
from Interface.data     import IMUData
from Interface.stationary import detect_stationary


DataType = tp.TypeVar("DataType", bound=IMUData)


class ZeroVelocity(ISequence[DataType]):
    """Populate ``stationary`` from the full sequence below windowing wrappers."""

    CONFIG_SPEC: dict = {}

    def __init__(self, config: DictConfig, seq: ISequence[DataType]):
        self.check_config(config)
        super().__init__(config)
        self.internal_seq = seq

        full = seq[0 : len(seq)]                            # DataType, B=1
        acc  = full.acc [0, :, 0, :].unsqueeze(0)            # (1, N, 3) main-body, RAW
        gyro = full.gyro[0, :, 0, :].unsqueeze(0)            # (1, N, 3)

        # Diff the int64 ns timestamps BEFORE scaling: converting the absolute
        # ~1.7e18 ns stamp to float first collapses consecutive samples.
        t_ns = full.time_ns[0, :, 0, 0]                                      # (N,) int64 ns
        assert bool(torch.all(t_ns[1:] > t_ns[:-1]))
        dt   = float((t_ns[1:] - t_ns[:-1]).to(torch.float64).median().item() * 1e-9)

        mask = detect_stationary(acc, gyro, dt)              # (1, N) bool
        self._stationary = mask[0].cpu()                     # (N,) bool, CPU static buffer

    @property
    def seq_name(self) -> str:
        return self.internal_seq.seq_name

    def __getitem__(self, index: slice | int) -> DataType:
        sample = self.internal_seq[index]
        if isinstance(index, int):
            index = slice(index, index + 1)
        return dataclasses.replace(
            sample,
            stationary=self._stationary[index].unsqueeze(0).to(device=sample.acc.device),
        )

    def __len__(self) -> int:
        return len(self.internal_seq)

    def clip(self, start: int | None, end: int | None, step: int | None) -> None:
        self.internal_seq.clip(start, end, step)
        self._stationary = self._stationary[slice(start, end, step)]

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        self.internal_seq.to(device=device, dtype=dtype)
        if device is not None:
            self._stationary = self._stationary.to(device=device)


def attach_stationary_mask(seq: ISequence[DataType]) -> ISequence[DataType]:
    """Insert ``ZeroVelocity`` directly above a chain's base loader."""
    # ISequence does not declare the wrapper-only internal_seq link.
    parent: tp.Any = seq
    while hasattr(parent, "internal_seq"):
        if isinstance(parent, ZeroVelocity):
            raise ValueError(
                f"{seq.seq_name}: the dataset config already wraps Sequence.ZeroVelocity; "
                "the trainer attaches the stationary mask itself, so drop the wrapper node "
                "from the YAML (or set train.regularization_gate=gt / regularization off)."
            )
        if not hasattr(parent.internal_seq, "internal_seq"):
            parent.internal_seq = ZeroVelocity(DictConfig({}), parent.internal_seq)
            return seq
        parent = parent.internal_seq
    return ZeroVelocity(DictConfig({}), seq)   # bare loader, no windowing wrapper
