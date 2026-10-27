import dataclasses
import typing as tp

import torch
from omegaconf import DictConfig

from Interface.sequence import ISequence
from Interface.data     import IMUData


DataType = tp.TypeVar("DataType", bound=IMUData)


class TagSequence(ISequence[DataType]):
    """Set ``seq_id`` on every batch row; with ``window_start`` it locates each window."""

    CONFIG_SPEC = {
        "seq_id": lambda i: isinstance(i, int) and not isinstance(i, bool) and i >= 0,
    }

    def __init__(self, config: DictConfig, seq: ISequence[DataType]):
        self.check_config(config)
        super().__init__(config)
        self.internal_seq = seq
        self._seq_id: int = config.seq_id

    @property
    def seq_name(self) -> str:
        return self.internal_seq.seq_name

    def __getitem__(self, index: slice | int) -> DataType:
        data: DataType = self.internal_seq[index]
        B = data.acc.shape[0]
        seq_id = torch.full((B,), self._seq_id, dtype=torch.int64, device=data.acc.device)
        return dataclasses.replace(data, seq_id=seq_id)

    def __len__(self) -> int:
        return len(self.internal_seq)
