"""Populate ``input_rot`` from an external per-sequence orientation pickle.

Higher-order wrapper around ``ISequence[IMUData]``, sitting directly on the
per-dataset loader (below windowing). At construction the orientation stream
for the inner's ``seq_name`` is loaded from a pickle and validated
against the inner frame count; every read then serves the inner sample with
``input_rot`` set to the corresponding slice. The stream is a network-input
attitude override (body->world SO3): ``get_est_velocity`` conditions the
velocity network on it instead of ``gt_poses.rotation()``. ``gt_poses`` and all
other fields pass through untouched — the GT used for anchors/metrics and the
pseudo-label frame is preserved.

Pickle layout: ``{seq_name: {"orientation": (T, 4)}}``
where the (T, 4) quaternion is xyzw (ROS/pypose SO3 convention). Accepted time
lengths: ``T == N`` or legacy ``T == N-1`` (the last frame is padded by
repeating the last quaternion). Population is
opt-in via composing this wrapper in the dataset YAML; loaders NEVER auto-fill
``input_rot``.
"""

import dataclasses
import pickle
import typing as tp

import torch
import pypose as pp
from omegaconf import DictConfig

from Interface.sequence import ISequence
from Interface.data     import IMUData


DataType = tp.TypeVar("DataType", bound=IMUData)


class ExternalOrientation(ISequence[DataType]):
    CONFIG_SPEC = {
        "path": lambda v: isinstance(v, str) and len(v) > 0,
    }

    def __init__(self, config: DictConfig, seq: ISequence[DataType]):
        self.check_config(config)
        super().__init__(config)
        self.internal_seq = seq

        with open(str(self.config.path), "rb") as f:
            results = pickle.load(f)
        seq_name = seq.seq_name
        if seq_name not in results:
            raise KeyError(
                f"ExternalOrientation: sequence {seq_name!r} not in pickle "
                f"{str(self.config.path)!r} (available: {sorted(results.keys())[:8]}...). "
                "Remove this wrapper from the sequence's YAML entry to use GT rotation."
            )
        entry = results[seq_name]
        if "orientation" not in entry:
            raise KeyError(
                f"ExternalOrientation: entry for {seq_name!r} must contain "
                f"'orientation'; got {sorted(entry.keys())}"
            )

        full = seq[0 : len(seq)]                        # DataType, B=1
        ref  = full.gyro[0]                             # (N, K, 3) — device/dtype ref
        ref0 = ref.reshape(-1)[0] if ref.numel() else ref
        n    = len(seq)

        quat = torch.as_tensor(entry["orientation"])    # (T, 4) xyzw
        if quat.dim() != 2 or quat.shape[-1] != 4:
            raise ValueError(
                f"ExternalOrientation: orientation for {seq_name!r} must be (T, 4) "
                f"xyzw quaternions, got shape {tuple(quat.shape)}"
            )
        t = int(quat.shape[0])
        if t == n - 1:
            quat = torch.cat([quat, quat[-1:]], dim=0)  # pad last frame
        elif t != n:
            raise ValueError(
                f"ExternalOrientation: time mismatch for {seq_name!r} — expected "
                f"N={n} or N-1={n - 1} frames, got {t}."
            )

        # A LieTensor must be floating: match the reference dtype if floating,
        # else fall back to float32.
        ltype = ref0.dtype if ref0.dtype.is_floating_point else torch.float32
        self._input_rot = pp.SO3(
            quat.to(dtype=ltype, device=ref0.device)
        )                                               # (N, 4) SO3

    @property
    def seq_name(self) -> str:
        return self.internal_seq.seq_name

    def __getitem__(self, index: slice | int) -> DataType:
        sample = self.internal_seq[index]
        if isinstance(index, int):
            index = slice(index, index + 1)
        # Slice + unsqueeze while preserving the SO3 ltype (reconstruct from the
        # underlying tensor so the ltype cannot be lost through indexing).
        sub = pp.SO3(self._input_rot.tensor()[index]).unsqueeze(0)
        return dataclasses.replace(sample, input_rot=sub)

    def __len__(self) -> int:
        return len(self.internal_seq)

    def clip(self, start: int | None, end: int | None, step: int | None) -> None:
        self.internal_seq.clip(start, end, step)
        sl = slice(start, end, step)
        self._input_rot = pp.SO3(self._input_rot.tensor()[sl])

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        self.internal_seq.to(device=device, dtype=dtype)
        kwargs: dict = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if kwargs:
            self._input_rot = pp.SO3(self._input_rot.tensor().to(**kwargs))
