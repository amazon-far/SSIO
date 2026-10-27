import json
import os

import numpy as np
import torch
import pypose as pp
from omegaconf import DictConfig, OmegaConf

from Interface.base     import Opt
from Interface.sequence import ISequence
from Interface.data     import IMUData_w_GT, IMUJointsData_w_GT
from Sequence.debias import run_calibrater, is_calibrater_node


# Canonical humanoid dataset: ``humanoid_g1`` (format 3).
# One clip directory per sequence under ``data_root``:
#   <data_root>/<seq_name>/spec.json      format/length/part names/dtypes
#   <data_root>/<seq_name>/imus.npy    (P, T, 9)  float16  [acc | gyro | gt_vel]
#   <data_root>/<seq_name>/pose.npy      (P, T, 7)  float32  world SE3 (tx,ty,tz,qx,qy,qz,qw)
#   <data_root>/<seq_name>/time_ns.npy   (T,)       int64    nanoseconds, shared across parts
# P = number of body parts; the part axis is indexed by name via
# ``spec.json``'s ``names`` list.
#
# Conventions (pelvis part):
#   - ``pose`` quaternion is xyzw and is R_wb (body→world): feeding it to the
#     gyro-consistency test R_wbᵀ dR_wb/dt = skew(gyro) matches the measured
#     body rates; wxyz does not.
#   - ``acc`` is standard z-up specific force: R_wb @ acc ≈ (0, 0, +g) on
#     static frames, matching pypose IMUPreintegrator's default
#     ``gravity=+9.81007`` (it computes ``a = acc - R_bw @ g_w``). No basis
#     change is needed; downstream must use the default positive gravity.
#   - ``gt_vel`` is each part's own body-frame linear velocity (matches
#     d(position)/dt only after lifting with R_wb). The IMUData_w_GT contract
#     carries world-frame velocity, so we rotate per-frame at load.
#   - Static-frame |acc| clusters per robot session (9.74 / 9.85 / 9.95–10.01),
#     i.e. a per-session accelerometer bias up to ~0.17 m/s² — keep
#     ``debias: true`` so the all-static-frame estimate removes it.
#   - ~200 Hz nominal; dt jitters and a few clips have 80–210 ms gaps, so
#     always integrate with the true per-step dt.


class HumanoidOdom(ISequence[IMUData_w_GT]):
    """One humanoid_g1 clip, single configurable body part.

    Loads ``<data_root>/<seq_name>/`` (format-3 clip directory) and emits
    ``IMUData_w_GT`` for the configured ``part`` (canonically ``"pelvis"``)
    paired with that part's SE3 pose and world-frame velocity.

    With ``joints: true`` it emits ``IMUJointsData_w_GT`` instead, adding the G1's
    29 joints from the clip's ``joints.npy`` (aligned 1:1 with ``time_ns.npy``).
    Channels are selected by name from ``joints.json``; ``q``, ``dq``, and
    ``tau_est`` map to position, velocity, and effort.
    """

    CONFIG_SPEC = {
        "data_root":  lambda v: isinstance(v, str) and len(v) > 0,
        "seq_name":   lambda v: isinstance(v, str) and len(v) > 0,
        "part":       lambda v: isinstance(v, str) and len(v) > 0,
        "device":     lambda v: isinstance(v, torch.device),
        "dtype":      lambda v: isinstance(v, torch.dtype),
        "acc_sigma":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "gyro_sigma": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "debias":     lambda v: isinstance(v, bool),
        # ``{kind: span|set|imu, ...}``; null is invalid when debiasing. See debias.build_calibrater.
        "calibrater": is_calibrater_node,
        # World gravity magnitude used by the static-frame debias below. Wire it
        # to ${train.gravity} in the dataset YAML so the value the debias
        # subtracts is by construction the same one pypose's IMUPreintegrator
        # subtracts downstream — a mismatch leaves a constant residual in world
        # z that integrates straight into velocity/position drift.
        "gravity":    lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        # Opt-in joint loading from the clip's joints.npy; absent or false => IMU-only.
        "joints": Opt(lambda v: isinstance(v, bool)),
    }

    SUPPORTED_FORMAT = 3

    # Per-joint channels this loader understands, mapped onto JointData's fields.
    # `position`/`velocity` are required; `effort` is optional.
    REQUIRED_JOINT_CHANNELS = ("position", "velocity")

    # Static-window thresholds live on Sequence.debias.StaticSetDebias.

    def __init__(self, config: DictConfig):
        self.check_config(config)
        super().__init__(config)

        seq_name   = str(config.seq_name)
        part       = str(config.part)
        dtype      = config.dtype
        acc_sigma  = float(config.acc_sigma)
        gyro_sigma = float(config.gyro_sigma)

        clip_dir = os.path.join(str(config.data_root), seq_name)
        spec_path = os.path.join(clip_dir, "spec.json")
        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"HumanoidOdom clip spec not found: {spec_path}")
        with open(spec_path) as f:
            spec = json.load(f)

        if int(spec.get("format", -1)) != self.SUPPORTED_FORMAT:
            raise ValueError(
                f"HumanoidOdom clip {clip_dir!r}: format {spec.get('format')!r}, "
                f"expected {self.SUPPORTED_FORMAT}."
            )
        names = list(spec["names"])
        if part not in names:
            raise ValueError(
                f"HumanoidOdom clip {clip_dir!r}: part {part!r} not in {names}."
            )
        k = names.index(part)
        T = int(spec["length"])

        imus    = np.load(os.path.join(clip_dir, "imus.npy"))      # (P, T, 9) float16
        pose    = np.load(os.path.join(clip_dir, "pose.npy"))      # (P, T, 7) float32
        time_ns = np.load(os.path.join(clip_dir, "time_ns.npy"))   # (T,)      int64

        if imus.shape != (len(names), T, 9) or pose.shape != (len(names), T, 7) \
                or time_ns.shape != (T,):
            raise ValueError(
                f"HumanoidOdom clip {clip_dir!r}: shape mismatch vs spec — "
                f"imus {imus.shape}, pose {pose.shape}, time_ns {time_ns.shape}, T={T}."
            )

        # float16 storage → float32 before any math; final cast to config dtype.
        part_imus = np.ascontiguousarray(imus[k]).astype(np.float32)   # (T, 9)
        part_pose    = np.ascontiguousarray(pose[k]).astype(np.float32)      # (T, 7)
        acc_np       = part_imus[:, 0:3]
        gyro_np      = part_imus[:, 3:6]
        vel_body_np  = part_imus[:, 6:9]

        # int64 time_ns cannot be non-finite; check the float streams.
        for name, arr in (("imus", part_imus), ("pose", part_pose)):
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"HumanoidOdom clip {clip_dir!r} has non-finite entries in "
                    f"{name!r}; SSIO requires complete IMU/GT streams."
                )

        self._acc           = torch.from_numpy(acc_np ).to(dtype=dtype)          # (T, 3)
        self._gyro          = torch.from_numpy(gyro_np).to(dtype=dtype)          # (T, 3)
        self._gt_poses_data = torch.from_numpy(part_pose).to(dtype=dtype)        # (T, 7)

        # Rotate body-frame velocity into the world frame.
        rot_wb         = pp.SO3(self._gt_poses_data[:, 3:7])                     # (T,)
        vel_body       = torch.from_numpy(vel_body_np).to(dtype=dtype)           # (T, 3)
        self._gt_vel_w = rot_wb @ vel_body                                       # (T, 3)

        time_ns_t = torch.from_numpy(np.ascontiguousarray(time_ns))                # (T,) int64
        self._corrected_acc, self._corrected_gyro, self.debias_report = run_calibrater(
            config, float(config.gravity), f"HumanoidOdom {seq_name}",
            self._acc, self._gyro, rot_wb, self._gt_vel_w,
            time_ns=time_ns_t.view(-1),
        )

        # Single-part loader: K=1 in storage to satisfy the IMUData K axis.
        self._time_ns = time_ns_t.view(T, 1, 1)                                 # (T, 1, 1)
        self._acc     = self._acc.unsqueeze(1)                                  # (T, 1, 3) as measured
        self._gyro    = self._gyro.unsqueeze(1)                                 # (T, 1, 3) as measured
        if self._corrected_acc is not None:
            self._corrected_acc  = self._corrected_acc .unsqueeze(1)
            self._corrected_gyro = self._corrected_gyro.unsqueeze(1)

        eye3 = torch.eye(3, dtype=dtype)
        self._acc_cov  = (eye3 * (acc_sigma  ** 2)).expand(T, 1, 3, 3).contiguous()   # (T, 1, 3, 3)
        self._gyro_cov = (eye3 * (gyro_sigma ** 2)).expand(T, 1, 3, 3).contiguous()   # (T, 1, 3, 3)

        # ``self._joints`` is None for the IMU-only path and a channel->tensor map
        # otherwise; ``__getitem__`` switches emitted type on it.
        self._joints: dict[str, torch.Tensor] | None = None
        if OmegaConf.select(config, "joints", default=False):
            self._joints = self._load_joints(clip_dir, seq_name, time_ns, dtype)

        self.to(device=config.device, dtype=None)

    # joints.json channel name -> JointData field.
    _JOINT_CHANNELS = {"q": "position", "dq": "velocity", "tau_est": "effort"}

    def _load_joints(
        self, clip_dir: str, seq_name: str, clip_time_ns: np.ndarray,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Read ``<clip_dir>/joints.npy``, ``(T, 29, C)`` aligned 1:1 with ``time_ns.npy``.

        Returns ``{JointData field: (T, J, 1)}`` for the channels ``joints.json`` names.
        """
        with open(os.path.join(clip_dir, "joints.json")) as f:
            meta = json.load(f)
        joints = np.load(os.path.join(clip_dir, "joints.npy"), allow_pickle=False)   # (T, J, C)
        if joints.shape[0] != clip_time_ns.size:
            raise ValueError(
                f"HumanoidOdom {seq_name}: joints.npy has {joints.shape[0]} frames but "
                f"time_ns.npy has {clip_time_ns.size}; they must align 1:1."
            )

        out: dict[str, torch.Tensor] = {}
        # Index channels BY NAME, as joints.json lists them, never by position.
        for name, (i, _unit) in meta["channels"].items():
            field = self._JOINT_CHANNELS.get(name)
            if field is None:
                continue
            chan = np.ascontiguousarray(joints[:, :, i]).astype(np.float32)   # (T, J)
            if not np.isfinite(chan).all():
                raise ValueError(
                    f"HumanoidOdom {seq_name}: non-finite {name} in joints.npy; "
                    f"the joints network requires a complete proprioceptive stream."
                )
            out[field] = torch.from_numpy(chan).to(dtype=dtype).unsqueeze(-1)  # (T, J, 1)

        missing = [c for c in self.REQUIRED_JOINT_CHANNELS if c not in out]
        if missing:
            raise ValueError(
                f"HumanoidOdom {seq_name}: joints.npy lacks required joint channel(s) "
                f"{missing}; joints.json lists {list(meta['channels'])}."
            )
        return out

    @property
    def seq_name(self) -> str:
        return str(self.config.seq_name)

    def __len__(self) -> int:
        return int(self._time_ns.shape[0])

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            index = slice(index, index + 1)
        common = dict(
            time_ns  = self._time_ns      [index].unsqueeze(0),
            acc      = self._acc          [index].unsqueeze(0),
            gyro     = self._gyro         [index].unsqueeze(0),
            acc_cov  = self._acc_cov      [index].unsqueeze(0),
            gyro_cov = self._gyro_cov     [index].unsqueeze(0),
            corrected_acc  = None if self._corrected_acc  is None else self._corrected_acc [index].unsqueeze(0),
            corrected_gyro = None if self._corrected_gyro is None else self._corrected_gyro[index].unsqueeze(0),
            gt_poses = pp.SE3(self._gt_poses_data[index].unsqueeze(0)),
            gt_vel_w = self._gt_vel_w     [index].unsqueeze(0),
        )
        if self._joints is None:
            return IMUData_w_GT(**common)

        effort = self._joints.get("effort")
        return IMUJointsData_w_GT(
            **common,
            # Joints ride the same clock as the IMU: one time_ns per clip, and the
            # (T, 1) shape drops the K axis the IMU stack carries.
            joint_time_ns = self._time_ns[index, 0].unsqueeze(0),
            position      = self._joints["position"][index].unsqueeze(0),
            velocity      = self._joints["velocity"][index].unsqueeze(0),
            # Preserve an absent effort channel instead of fabricating zeros.
            effort        = None if effort is None else effort[index].unsqueeze(0),
        )

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        def _move(t: torch.Tensor) -> torch.Tensor:
            return t if device is None else t.to(device=device)

        self._time_ns       = _move(self._time_ns)   # int64; dtype intentionally skipped
        self._acc            = _move(self._acc)
        self._gyro           = _move(self._gyro)
        if self._corrected_acc is not None:
            self._corrected_acc  = _move(self._corrected_acc)
            self._corrected_gyro = _move(self._corrected_gyro)
        self._acc_cov       = _move(self._acc_cov)
        self._gyro_cov      = _move(self._gyro_cov)
        self._gt_poses_data = _move(self._gt_poses_data)
        self._gt_vel_w      = _move(self._gt_vel_w)
        if self._joints is not None:
            self._joints = {k: _move(v) for k, v in self._joints.items()}
