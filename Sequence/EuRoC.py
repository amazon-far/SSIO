import os

import numpy as np
import torch
import pypose as pp
from omegaconf import DictConfig
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from Interface.sequence     import ISequence
from Interface.data         import IMUData_w_GT
from Sequence.debias        import run_calibrater, is_calibrater_node
from Sequence.time_sync     import sanitize_time


class EuRoC_Sequence(ISequence[IMUData_w_GT]):
    """One EuRoC sequence directory (e.g. ``MH_01_easy`` under ``data_root``).

    Mirrors the structure of ``TLIO_Sequence``: loads IMU and GT data,
    synchronizes onto a common timeline (IMU timestamps as the master grid,
    clipped to the GT/IMU overlap; GT orientation SLERPed, GT position /
    velocity linearly interpolated), optionally removes a constant acc/gyro
    bias, and emits ``IMUData_w_GT`` (body-frame IMU, world-frame poses and
    velocity).

    PyPose-gravity compatibility is automatic: raw EuRoC IMU is body-frame
    and ``rot_wb`` is ``R_w_from_body`` in z-up world, so PyPose's
    ``g_world = [0, 0, +gravity]`` cancels gravity without any axis flip.
    ``debias: false`` returns IMU readings untouched. ``true`` removes
    a constant acc/gyro bias estimated on a near-static window so that the
    window-mean acc matches ``R_wb.Inv() @ [0, 0, +gravity]``.
    """

    CONFIG_SPEC = {
        "data_root":  lambda v: isinstance(v, str) and len(v) > 0,
        "seq_name":   lambda v: isinstance(v, str) and len(v) > 0,
        "device":     lambda v: isinstance(v, torch.device),
        "dtype":      lambda v: isinstance(v, torch.dtype),
        "acc_sigma":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "gyro_sigma": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "gravity":    lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "debias":     lambda v: isinstance(v, bool),
        # ``{kind: span|set|imu, ...}``; null is invalid when debiasing. See debias.build_calibrater.
        "calibrater": is_calibrater_node,
    }

    def __init__(self, config: DictConfig):
        self.check_config(config)
        super().__init__(config)

        seq_path        = os.path.join(str(config.data_root), str(config.seq_name))
        dtype           = config.dtype
        acc_sigma       = float(config.acc_sigma)
        gyro_sigma      = float(config.gyro_sigma)
        gravity         = float(config.gravity)

        # --- raw CSVs (NO header — do not use skiprows); timestamps → seconds
        imu_path = os.path.join(seq_path, "mav0", "imu0", "data.csv")
        gt_path  = os.path.join(seq_path, "mav0", "state_groundtruth_estimate0", "data.csv")
        if not os.path.isfile(imu_path):
            raise FileNotFoundError(f"EuRoC IMU file not found: {imu_path}")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"EuRoC GT file not found: {gt_path}")
        imu_data = np.loadtxt(imu_path, dtype=float, delimiter=",")
        gt_data  = np.loadtxt(gt_path,  dtype=float, delimiter=",")

        t_imu   = imu_data[:, 0] / 1e9
        gyro_np = imu_data[:, 1:4]
        acc_np  = imu_data[:, 4:7]
        t_gt    = gt_data[:, 0] / 1e9
        gt_pos  = gt_data[:, 1:4]
        gt_quat = gt_data[:, 4:8][:, [1, 2, 3, 0]]   # wxyz in file → xyzw for pypose
        gt_vel  = gt_data[:, -9:-6]                  # match reference: negative indexing from end

        # --- common timeline: IMU timestamps clipped to the GT/IMU overlap ---
        t_imu_keep, imu_keep_idx, _ = sanitize_time(t_imu)
        t_gt_keep, gt_keep_idx, gt_dropped = sanitize_time(t_gt)
        if len(t_imu_keep) < 2 or len(t_gt_keep) < 2:
            raise ValueError("Less than 2 strictly increasing timestamps in IMU or GT stream")
        t_start = max(t_imu_keep[0], t_gt_keep[0])
        t_end   = min(t_imu_keep[-1], t_gt_keep[-1])
        if not (t_end > t_start):
            raise ValueError("No overlapping time window across streams")
        imu_mask = (t_imu_keep >= t_start) & (t_imu_keep <= t_end)
        sync_times = t_imu_keep[imu_mask]
        if len(sync_times) < 2:
            raise ValueError(f"Less than 2 synchronized time samples: {len(sync_times)}")
        acc_np = acc_np[imu_keep_idx]
        gyro_np = gyro_np[imu_keep_idx]

        # --- GT → sync_times: SLERP for orientation, linear for pos / vel ----
        # scipy's Slerp / interp1d need strictly increasing source times, so
        # all three GT streams share the sanitized GT timeline.
        if gt_dropped > 0:
            gt_quat, gt_pos, gt_vel = gt_quat[gt_keep_idx], gt_pos[gt_keep_idx], gt_vel[gt_keep_idx]
        slerp  = Slerp(t_gt_keep - t_gt_keep[0], R.from_quat(gt_quat))
        q_xyzw = torch.tensor(slerp(sync_times - t_gt_keep[0]).as_quat()).to(dtype=dtype)   # [N, 4]
        trans = torch.tensor(interp1d(
            t_gt_keep, gt_pos, axis=0,
            bounds_error=False, fill_value="extrapolate", assume_sorted=True,   # type: ignore[arg-type]
        )(sync_times)).to(dtype=dtype)                                          # [N, 3]
        self._gt_vel_w = torch.tensor(interp1d(
            t_gt_keep, gt_vel, axis=0,
            bounds_error=False, fill_value="extrapolate", assume_sorted=True,   # type: ignore[arg-type]
        )(sync_times)).to(dtype=dtype)                                          # [N, 3]
        self._gt_poses_data = torch.cat([trans, q_xyzw], dim=-1)                # [N, 7]

        # --- timestamps: preserve nanosecond integers from the raw file ------
        self._time_ns = (
            torch.from_numpy(t_imu_keep[imu_mask]).double() * 1e9
        ).round().to(torch.int64).unsqueeze(-1)

        # --- IMU (body frame, matching TLIO layout) --------------------------
        self._acc  = torch.from_numpy(acc_np [imu_mask]).to(dtype=dtype)
        self._gyro = torch.from_numpy(gyro_np[imu_mask]).to(dtype=dtype)

        rot_wb = pp.SO3(self._gt_poses_data[:, 3:7])
        self._corrected_acc, self._corrected_gyro, self.debias_report = run_calibrater(
            config, gravity, f"EuRoC {config.seq_name}",
            self._acc, self._gyro, rot_wb, self._gt_vel_w,
            time_ns=self._time_ns.view(-1),
        )

        # Single-IMU dataset, but IMUData carries an explicit K axis → K=1.
        N    = int(self._time_ns.shape[0])
        self._time_ns = self._time_ns.unsqueeze(1)                       # (N, 1, 1)
        self._acc     = self._acc.unsqueeze(1)                           # (N, 1, 3)
        self._gyro    = self._gyro.unsqueeze(1)                          # (N, 1, 3)
        if self._corrected_acc is not None:
            self._corrected_acc  = self._corrected_acc .unsqueeze(1)
            self._corrected_gyro = self._corrected_gyro.unsqueeze(1)

        # --- covariance matrices (diagonal, isotropic) ---
        eye3 = torch.eye(3, dtype=dtype)
        self._acc_cov  = (eye3 * acc_sigma ** 2).expand(N, 1, 3, 3).contiguous()   # (N, 1, 3, 3)
        self._gyro_cov = (eye3 * gyro_sigma ** 2).expand(N, 1, 3, 3).contiguous()  # (N, 1, 3, 3)

        self.to(device=config.device, dtype=None)

    # ------------------------------------------------------------------
    # ISequence protocol — identical structure to TLIO_Sequence
    # ------------------------------------------------------------------


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
            corrected_acc  = None if self._corrected_acc  is None else self._corrected_acc [index].unsqueeze(0),
            corrected_gyro = None if self._corrected_gyro is None else self._corrected_gyro[index].unsqueeze(0),
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
        if self._corrected_acc is not None:
            self._corrected_acc  = _move(self._corrected_acc)
            self._corrected_gyro = _move(self._corrected_gyro)
