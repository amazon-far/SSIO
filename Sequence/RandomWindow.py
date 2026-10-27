import dataclasses
import typing as tp

import torch
from omegaconf import DictConfig

from Interface.sequence import ISequence
from Interface.data     import IMUData, cat_field


DataType = tp.TypeVar("DataType", bound=IMUData)


class RandomWindow(ISequence[DataType]):
    """Random fixed-size windows of consecutive IMU samples.

    Each ``__getitem__`` call samples a fresh start uniformly from
    ``[0, len(seq) - window_size]`` — the requested index is ignored.
    ``__len__`` returns ``(num_samples // window_size) * 10``: a stand-in
    "epoch size" since random sampling can never guarantee full coverage.
    """

    CONFIG_SPEC = {
        "window_size" : lambda i: isinstance(i, int) and i > 0,
    }

    def __init__(self, config: DictConfig, seq: ISequence[DataType]):
        super().__init__(config)
        # ISequence.__init__ does not validate, so without this CONFIG_SPEC is dead
        # and a stray key (a step_size left from SlidingWindow) passes silently.
        self.check_config(config)
        self.internal_seq = seq

        n = len(self.internal_seq)
        w = int(self.config.window_size)
        if n < w:
            raise ValueError(
                f"RandomWindow: internal sequence has {n} samples but window_size={w}; "
                "need len(seq) >= window_size."
            )
        self._epoch_len = (n // w) * 10

    @property
    def seq_name(self) -> str:
        return self.internal_seq.seq_name

    def __getitem__(self, index: slice | int) -> DataType:
        n_windows:   int = 1 if isinstance(index, int) else len(range(*index.indices(len(self))))
        window_size: int = int(self.config.window_size)
        max_start:   int = len(self.internal_seq) - window_size

        starts: list[int] = torch.randint(0, max_start + 1, (n_windows,)).tolist()

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
        # Per-window flag: True only when a window happens to be sampled at
        # frame 0 of the underlying sequence (one bool per window, along B).
        # Random starts make this almost always False.
        kwargs["is_first_window"] = torch.tensor(
            [s == 0 for s in starts], dtype=torch.bool, device=samples[0].acc.device
        )
        kwargs["window_start"] = torch.tensor(starts, dtype=torch.int64, device=samples[0].acc.device)
        return cls(**kwargs)

    def __len__(self) -> int:
        return self._epoch_len
