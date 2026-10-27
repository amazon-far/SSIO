# The BlackBird preprocessing (``_blackbird_refer_imo``) is ported from AirIO's
# BlackBird dataset loader (https://github.com/Air-IO/AirIO); see THIRD_PARTY_NOTICES.md.

import copy
import os

import numpy as np
import torch
import pypose as pp
from omegaconf import DictConfig
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from Interface.sequence    import ISequence
from Interface.data        import IMUData_w_GT
from Sequence.debias        import run_calibrater, is_calibrater_node


class BlackBird_Sequence(ISequence[IMUData_w_GT]):
    """One BlackBird sequence folder (``seq_name`` relative to ``data_root``).

    Loads ``imu_data.csv``, ``thrust_data.csv``, and ``groundTruthPoses.csv``,
    applies the same NED→ENU + IMU extrinsic + resampling pipeline as AirIO's
    ``BlackBirddataset.refer_IMO``, and emits ``IMUData_w_GT``.

    PyPose's ``IMUPreintegrator`` convention (z-up world,
    ``g_world = [0, 0, +gravity]``, raw specific-force body-frame ``acc``) is
    satisfied by ``_blackbird_refer_imo``: it rotates the GT pose into the IMU
    body frame via ``R_w_ned`` and ``R_b_i`` so that ``rot_wb`` already maps
    body-frame IMU readings into the z-up world.

    ``debias: false`` returns IMU readings untouched. ``true`` removes
    a constant acc/gyro bias averaged over the frames inside qualifying
    near-static runs, so that their mean acc matches
    ``R_wb.Inv() @ [0, 0, +gravity]`` (see the estimator note in ``__init__``
    for why this loader uses ``set`` and not ``span``).
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

        seq_dir    = os.path.join(str(config.data_root), str(config.seq_name))
        dtype      = config.dtype
        acc_sigma       = float(config.acc_sigma)
        gyro_sigma      = float(config.gyro_sigma)
        gravity         = float(config.gravity)

        imu_path    = os.path.join(seq_dir, "imu_data.csv")
        thrust_path = os.path.join(seq_dir, "thrust_data.csv")
        gt_path     = os.path.join(seq_dir, "groundTruthPoses.csv")
        for p in (imu_path, thrust_path, gt_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"BlackBird_Sequence: missing file {p}")

        imu_data  = np.loadtxt(imu_path, dtype=float, delimiter=",")
        thrusts   = np.loadtxt(thrust_path, dtype=float, delimiter=",")
        gt_data   = np.loadtxt(gt_path, dtype=float, delimiter=",")

        times_imu, raw_imu, gt_traj_tmp = _blackbird_refer_imo(imu_data, thrusts, gt_data)

        n = int(times_imu.shape[0])
        time_ns = (times_imu * 1e9).astype(np.int64).reshape(n, 1)

        self._acc  = torch.from_numpy(np.ascontiguousarray(raw_imu[:, 4:7])).to(dtype=dtype)
        self._gyro = torch.from_numpy(np.ascontiguousarray(raw_imu[:, 1:4])).to(dtype=dtype)

        pos_w = torch.from_numpy(
            np.ascontiguousarray(interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 1:4], axis=0)(times_imu))
        ).to(dtype=dtype)
        quat_xyzw = Slerp(gt_traj_tmp[:, 0], Rotation.from_quat(gt_traj_tmp[:, 4:8]))(times_imu).as_quat()
        quat_xyzw = torch.from_numpy(np.ascontiguousarray(quat_xyzw)).to(dtype=dtype)
        self._gt_vel_w = torch.from_numpy(
            np.ascontiguousarray(interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 8:11], axis=0)(times_imu))
        ).to(dtype=dtype)

        self._gt_poses_data = torch.cat([pos_w, quat_xyzw], dim=-1)

        rot_wb = pp.SO3(self._gt_poses_data[:, 3:7])
        self._corrected_acc, self._corrected_gyro, self.debias_report = run_calibrater(
            config, gravity, f"BlackBird {config.seq_name}",
            self._acc, self._gyro, rot_wb, self._gt_vel_w,
            time_ns=torch.from_numpy(time_ns).view(-1),
        )

        # Single-IMU dataset, but IMUData carries an explicit K axis → K=1.
        self._time_ns = torch.from_numpy(time_ns).unsqueeze(1)           # (N, 1, 1)
        self._acc     = self._acc.unsqueeze(1)                           # (N, 1, 3)
        self._gyro    = self._gyro.unsqueeze(1)                          # (N, 1, 3)
        if self._corrected_acc is not None:
            self._corrected_acc  = self._corrected_acc .unsqueeze(1)
            self._corrected_gyro = self._corrected_gyro.unsqueeze(1)

        eye3 = torch.eye(3, dtype=dtype)
        self._acc_cov  = (eye3 * (acc_sigma ** 2)).expand(n, 1, 3, 3).contiguous()   # (N, 1, 3, 3)
        self._gyro_cov = (eye3 * (gyro_sigma ** 2)).expand(n, 1, 3, 3).contiguous()  # (N, 1, 3, 3)

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


def _blackbird_refer_imo(
    imu_data: np.ndarray,
    thrusts: np.ndarray,
    gt_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of AirIO's ``BlackBirddataset.refer_IMO`` — returns ``times_imu``, ``raw_imu`` (gyro, acc cols), ``gt_traj_tmp``."""
    raw_imu = np.asarray(copy.deepcopy(imu_data))
    thrusts = np.asarray(copy.deepcopy(thrusts))
    data_tmp = np.asarray(copy.deepcopy(gt_data))

    R_w_ned = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    t_w_ned = np.array([0.0, 0.0, 0.0])
    R_b_i = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t_b_i = np.array([0.0, 0.0, 0.0])

    rows: list[np.ndarray] = []
    for data_i in data_tmp:
        ts_i = data_i[0] / 1e6
        t_i = data_i[1:4]
        R_i = Rotation.from_quat(np.array([data_i[5], data_i[6], data_i[7], data_i[4]])).as_matrix()
        R_it = R_w_ned @ R_i
        t_it = t_w_ned + R_w_ned @ t_i
        t_it = t_it + R_it @ t_b_i
        R_it = R_it @ R_b_i
        q_it = Rotation.from_matrix(R_it).as_quat()
        rows.append(
            np.array(
                [ts_i, t_it[0], t_it[1], t_it[2], q_it[0], q_it[1], q_it[2], q_it[3]],
                dtype=float,
            )
        )
    data = np.stack(rows, axis=0)

    gt_times = data[:, 0]
    gt_pos = data[:, 1:4]
    v_start = ((gt_pos[1] - gt_pos[0]) / (gt_times[1] - gt_times[0])).reshape((1, 3))
    gt_vel_raw = (gt_pos[1:] - gt_pos[:-1]) / (gt_times[1:] - gt_times[:-1])[:, None]
    gt_vel_raw = np.concatenate((v_start, gt_vel_raw), axis=0)
    gt_vel_x = np.convolve(gt_vel_raw[:, 0], np.ones(5) / 5, mode="same").reshape((-1, 1))
    gt_vel_y = np.convolve(gt_vel_raw[:, 1], np.ones(5) / 5, mode="same").reshape((-1, 1))
    gt_vel_z = np.convolve(gt_vel_raw[:, 2], np.ones(5) / 5, mode="same").reshape((-1, 1))
    gt_vel = np.concatenate((gt_vel_x, gt_vel_y, gt_vel_z), axis=1)
    gt_traj_tmp = np.concatenate((data, gt_vel), axis=1)

    t_curr = raw_imu[0, 0]
    dt = 0.01
    new_times_imu = [t_curr]
    while t_curr < raw_imu[-1, 0] - dt - 0.001:
        t_curr = t_curr + dt
        new_times_imu.append(t_curr)
    new_times_imu = np.asarray(new_times_imu)
    gyro_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 1:4], axis=0)(new_times_imu)
    accel_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 4:7], axis=0)(new_times_imu)
    raw_imu = np.concatenate((new_times_imu.reshape((-1, 1)), gyro_tmp, accel_tmp), axis=1)

    times_imu = raw_imu[:, 0]
    idx_s = 0
    for ts in times_imu:
        if ts > gt_traj_tmp[0, 0] and ts > thrusts[0, 0]:
            break
        idx_s = idx_s + 1
    if idx_s >= len(times_imu):
        raise ValueError("BlackBird IMU, ground-truth, and thrust timelines do not overlap")

    idx_e = len(times_imu) - 1
    for ts in reversed(times_imu):
        if ts < gt_traj_tmp[-1, 0]:
            break
        idx_e = idx_e - 1
    if idx_e <= 0:
        raise ValueError("BlackBird IMU and ground-truth timelines do not overlap")

    times_imu = times_imu[idx_s : idx_e + 1]
    raw_imu = raw_imu[idx_s : idx_e + 1]
    return times_imu, raw_imu, gt_traj_tmp
