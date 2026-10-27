import os
import json

import numpy as np
import torch
import pypose as pp
from omegaconf import DictConfig

from Interface.sequence    import ISequence
from Interface.data        import IMUData_w_GT


# Column layout published by imu0_resampled_description.json:
#   0      ts_us(1)
#   1:4    gyr_compensated_rotated_in_World(3)
#   4:7    acc_compensated_rotated_in_World(3)
#   7:11   qxyzw_World_Device(4)        -- rotates body vector into world
#   11:14  pos_World_Device(3)
#   14:17  vel_World(3)
_REQUIRED_GT_COLS = (
    "qxyzw_World_Device(4)",
    "pos_World_Device(3)",
    "vel_World(3)",
)


class TLIO_Sequence(ISequence[IMUData_w_GT]):
    """One TLIO sequence folder.

    The packed ``imu0_resampled.npy`` stores IMU pre-rotated into the world
    frame using the GT orientation in the very same row. Body-frame IMU is
    only recoverable when those GT columns are present, so this loader
    requires GT and always emits ``IMUData_w_GT``.
    """

    CONFIG_SPEC = {
        "data_root":  lambda v: isinstance(v, str) and len(v) > 0,
        "seq_name":   lambda v: isinstance(v, str) and len(v) > 0,
        "device":     lambda v: isinstance(v, torch.device),
        "dtype":      lambda v: isinstance(v, torch.dtype),
        "acc_sigma":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "gyro_sigma": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
    }

    def __init__(self, config: DictConfig):
        self.check_config(config)
        super().__init__(config)

        seq_dir    = os.path.join(str(config.data_root), str(config.seq_name))
        dtype      = config.dtype
        acc_sigma  = float(config.acc_sigma)
        gyro_sigma = float(config.gyro_sigma)

        arr = np.load(os.path.join(seq_dir, "imu0_resampled.npy"))
        with open(os.path.join(seq_dir, "imu0_resampled_description.json")) as f:
            desc = json.load(f)

        cols = set(desc.get("columns_name(width)", []))
        missing = [c for c in _REQUIRED_GT_COLS if c not in cols]
        if missing:
            raise ValueError(
                f"TLIO sequence {seq_dir!r} is missing required GT columns "
                f"{missing}; body-frame IMU cannot be recovered without them."
            )

        N = arr.shape[0]

        time_ns = (arr[:, 0] * 1000.0).astype(np.int64).reshape(N, 1)

        gyro_world = torch.from_numpy(np.ascontiguousarray(arr[:, 1:4])).to(dtype)
        acc_world  = torch.from_numpy(np.ascontiguousarray(arr[:, 4:7])).to(dtype)
        q_wd       = torch.from_numpy(np.ascontiguousarray(arr[:, 7:11])).to(dtype)
        pos_wd     = torch.from_numpy(np.ascontiguousarray(arr[:, 11:14])).to(dtype)
        self._gt_vel_w = torch.from_numpy(np.ascontiguousarray(arr[:, 14:17])).to(dtype)

        # IMU is stored rotated-into-world; rotate back to body frame.
        R_dw = pp.SO3(q_wd).Inv()
        self._acc           = R_dw @ acc_world
        self._gyro          = R_dw @ gyro_world
        # Raw [N, 7] SE3 storage; wrap with pp.SE3 only at emit time.
        self._gt_poses_data = torch.cat([pos_wd, q_wd], dim=-1)

        # Single-IMU dataset, but IMUData carries an explicit K axis → K=1.
        self._time_ns = torch.from_numpy(time_ns).unsqueeze(1)           # (N, 1, 1)
        self._acc     = self._acc.unsqueeze(1)                           # (N, 1, 3)
        self._gyro    = self._gyro.unsqueeze(1)                          # (N, 1, 3)

        eye3 = torch.eye(3, dtype=dtype)
        self._acc_cov  = (eye3 * (acc_sigma ** 2)).expand(N, 1, 3, 3).contiguous()   # (N, 1, 3, 3)
        self._gyro_cov = (eye3 * (gyro_sigma ** 2)).expand(N, 1, 3, 3).contiguous()  # (N, 1, 3, 3)

        # No correction stage on this loader, so there is no corrected stream. Runs on
        # TLIO must therefore ask for the raw Δv source explicitly.
        self._corrected_acc  = None
        self._corrected_gyro = None
        self.to(device=config.device, dtype=None)


    @property
    def seq_name(self) -> str:
        """Sequence identifier — the per-sequence key in orientation pickles (Sequence.ExternalOrientation)."""
        return str(self.config.seq_name)

    def __len__(self) -> int:
        return int(self._time_ns.shape[0])

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            index = slice(index, index + 1)
        return IMUData_w_GT(
            time_ns        = self._time_ns      [index].unsqueeze(0),
            acc            = self._acc          [index].unsqueeze(0),
            gyro           = self._gyro         [index].unsqueeze(0),
            acc_cov        = self._acc_cov      [index].unsqueeze(0),
            gyro_cov       = self._gyro_cov     [index].unsqueeze(0),
            gt_poses       = pp.SE3(self._gt_poses_data[index].unsqueeze(0)),
            gt_vel_w       = self._gt_vel_w     [index].unsqueeze(0),
            corrected_acc  = None,   # no correction stage on this loader
            corrected_gyro = None,
        )

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        def _move(t: torch.Tensor) -> torch.Tensor:
            return t if device is None else t.to(device=device)

        self._time_ns       = _move(self._time_ns)   # int64; dtype intentionally skipped
        self._acc           = _move(self._acc)
        self._gyro          = _move(self._gyro)
        self._acc_cov       = _move(self._acc_cov)
        self._gyro_cov      = _move(self._gyro_cov)
        self._gt_poses_data = _move(self._gt_poses_data)
        self._gt_vel_w      = _move(self._gt_vel_w)
