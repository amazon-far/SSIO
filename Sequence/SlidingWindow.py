import dataclasses
import typing as tp

import torch
from omegaconf import DictConfig

from Interface.sequence import ISequence
from Interface.data     import IMUData, cat_field


DataType = tp.TypeVar("DataType", bound=IMUData)


class SlidingWindow(ISequence[DataType]):
    CONFIG_SPEC = {
        "window_size" : lambda i: isinstance(i, int) and i > 0,
        "step_size"   : lambda i: isinstance(i, int) and i > 0,
    }

    def __init__(self, config: DictConfig, seq: ISequence[DataType]):
        self.check_config(config)
        super().__init__(config)
        self.internal_seq = seq

        n           = len(seq)
        window_size = int(self.config.window_size)
        step_size   = int(self.config.step_size)
        assert n >= window_size, (
            f"SlidingWindow: sequence {seq.seq_name!r} length {n} < window_size "
            f"{window_size}; cannot form a window.")
        # ``last_start`` is the start index of the final full window. range()'s
        # upper bound is exclusive, so the stop is ``last_start + 1`` to reach it.
        last_start = n - window_size
        self.window_start = list(range(0, last_start + 1, step_size))

    @property
    def seq_name(self) -> str:
        return self.internal_seq.seq_name

    def __getitem__(self, index: slice | int) -> DataType:
        if isinstance(index, int):
            index = slice(index, index + 1)

        window_size: int       = int(self.config.window_size)
        starts:      list[int] = self.window_start[index]

        samples = [
            self.internal_seq[s : s + window_size]
            for s in starts
        ]

        cls    = type(samples[0])
        fields = dataclasses.fields(samples[0])
        kwargs = {
            f.name: cat_field([getattr(s, f.name) for s in samples], dim=0)
            for f in fields
        }
        # Per-window flag: True only for the window anchored at frame 0 of the
        # underlying sequence (one bool per window, concatenated along B).
        kwargs["is_first_window"] = torch.tensor(
            [s == 0 for s in starts], dtype=torch.bool, device=samples[0].acc.device
        )
        kwargs["window_start"] = torch.tensor(starts, dtype=torch.int64, device=samples[0].acc.device)
        return cls(**kwargs)

    def __len__(self) -> int:
        return len(self.window_start)
