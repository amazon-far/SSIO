"""ANYmal GrandTour multi-IMU + joint-state sequence loader for SSIO
**utilizing the HuggingFace Zarr format release**. 

Each "mission" is a per-topic Zarr store holding:
 -  several IMUs (all rigid relative to base unlike MIPO though)
 -  12 leg joints (pos/vel/effort loaded; joint acceleration is not loaded)
 -  per-foot state fields (NOT loaded by this loader)
    1.  a filtered contact flag from the actual force sensor (``{FOOT}_contact``/``_state``)
    2.  estimated contact wrench (``{FOOT}_wrench_force``/``_wrench_torque``)
    3.  estimated contact normal, surface coeff friction, and restitution coeff
 -  GNSS-INS sensor box ground truth for non-indoor sequences/sections (not the Prism GT)

    data/anymal_imu                     base-mounted body IMU, 400 Hz  (K stack idx 0)
    data/{stim320,adis,alphasense,...}  auxiliary Boxi IMUs, 100-500 Hz (all rigid to base so
                                        not proprioceptively as interesting). Can select which ones
                                        to add to the K IMU fields via ``imu_topics`` in the config.
                                        (default is just the main IMU, which is always idx 0). All
                                        all IMUs are rotated into the robot ``base`` using the TF tree
    data/anymal_state_state_estimator   12 joints (pos/vel/effort) + per-foot states, 400 Hz (frame: odom)
    data/cpt7_ie_tc_odometry            GNSS-INS GT pose+velocity, 200 Hz  (Outdoor only, frame: enu_origin)
    data/prism_position                 total-station GT position, ~23 Hz
    metadata/tf.yaml                    complete static TF tree

GrandTour has exact extrinsics and a metric robot-body GT with a real
velocity, so this loader applies the GrandTour calibration directly without extra
per-sequence fitting other than <5ms per-IMU clock offsets (read from the time alignment
file). Aligns to the cpt7 GNSS-INS clock.

``gt_source`` is chosen PER SEQUENCE in the dataset config between the GNSS-INS and the
onboard state estimator. The WHOLE sequence uses this one GT
source for its entire length.
 -  ``"cpt7_ie"`` (for OUTDOOR where GNSS is available): the metric robot-body GT.
    GNSS-INS pose and velocity twist. Intended for supervised training and all reported eval
    metrics.
 -  ``"state_estimator"`` (for INDOOR / GNSS-denied). A substitute that uses the
    onboard proprioceptive state estimate: pose in ``odom`` (drifts globally) and body
    ``twist_lin`` rotated to world. Intended for use only as unsupervised pseudo-label ori
    source for the 8 indoor (or largely indoor) missions that take it (ARC-6, ARC-7,
    DEC-1..4, ETH-2, IND-11) that appear only in the label-free SSL pool,
    NEVER in the supervised Sup/Val/Test sets. The drifting estimate should never be a
    supervised target.
"""

import json
import os
from typing import NamedTuple

import numpy as np
import torch
import pypose as pp
import zarr
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as _Rot

from Interface.base         import Opt
from Interface.sequence     import ISequence
from Interface.data         import IMUJointsData_w_GT
from Sequence.debias        import build_calibrater, disabled_report, is_calibrater_node
from Sequence.time_sync     import synchronize_timeline, resample_list


# 12 leg joints in order (see anymal_state group 'description'),
# HAA, HFE, and KFE for LF, RF, LH, and RH
_EXPECTED_JOINTS = [
    "LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE",
]

_MAIN_IMU = "anymal_imu"      # base-mounted body IMU (400 Hz), canonical K=0
_STATE    = "anymal_state_state_estimator"
_GT_CPT7  = "cpt7_ie_tc_odometry"


class _GTData(NamedTuple):
    """Common ground-truth streams at native GT rate, already in the robot base / world
    frame. Populated by the per-source builders so downstream code is source-agnostic."""
    time: np.ndarray   # (M,)   seconds
    pos:  np.ndarray   # (M, 3) base position (ENU for cpt7_ie, odom for state_estimator)
    quat: np.ndarray   # (M, 4) xyzw base orientation
    vel:  np.ndarray   # (M, 3) base velocity in world


def _se3_from_tf(tf: DictConfig, child: str) -> pp.LieTensor:
    """Build ``T_parent_child`` (pose of ``child`` in its ``base_frame_id`` parent) as a
    pypose SE3 from a single ``tf.yaml`` entry. Rotation is stored as named ``{w,x,y,z}``."""
    e = tf[child]
    r, t = e["rotation"], e["translation"]
    q_xyzw = torch.tensor([float(r["x"]), float(r["y"]), float(r["z"]), float(r["w"])], dtype=torch.float64)
    trans  = torch.tensor([float(t["x"]), float(t["y"]), float(t["z"])], dtype=torch.float64)
    return pp.SE3(torch.cat([trans, q_xyzw]))


def _T_base_from_tf(tf: DictConfig, child: str, base: str = "base", max_depth: int = 8) -> pp.LieTensor:
    """Compose the static transform chain from ``child`` up to the robot ``base`` frame and
    return ``T_base_child``."""
    accum_tf = pp.SE3(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
    cur = child
    for _ in range(max_depth):
        if cur not in tf:
            raise RuntimeError(f"tf chain: frame {cur!r} not in tf.yaml (resolving {child!r} -> {base!r})")
        accum_tf = _se3_from_tf(tf, cur) @ accum_tf                      # T_parent_cur @ T_cur_child = T_parent_child
        parent = str(tf[cur].get("base_frame_id", ""))
        if parent == base:
            return accum_tf
        cur = parent
    raise RuntimeError(f"tf chain from {child!r} did not reach {base!r} within {max_depth} hops")

def _largest_true_span(mask: np.ndarray) -> tuple[int, int]:
    """Return [start, end) of the longest contiguous run of True in a 1D bool mask."""
    best_len = best_start = best_end = 0
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > best_len:
                best_len, best_start, best_end = j - i, i, j
            i = j
        else:
            i += 1
    return best_start, best_end


class ANYmal_Sequence(ISequence[IMUJointsData_w_GT]):
    """One ANYmal GrandTour mission as ``IMUJointsData_w_GT``.

    K IMUs, 12 joints, metric body GT."""

    CONFIG_SPEC = {
        "data_root":   lambda v: isinstance(v, str) and len(v) > 0,
        "seq_name":    lambda v: isinstance(v, str) and len(v) > 0,   # mission dir name
        "device":      lambda v: isinstance(v, torch.device),
        "dtype":       lambda v: isinstance(v, torch.dtype),
        "acc_sigma":   lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "gyro_sigma":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "num_imus":    lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1,
        "num_joints":  lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1,
        "target_rate": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "gt_source":   lambda v: v in ("cpt7_ie", "state_estimator"),
        # Optional configs (defaults applied via OmegaConf.select in __init__).
        "debias":          Opt(lambda v: isinstance(v, bool)),
        "calibrater":      Opt(is_calibrater_node),
        "gravity":         Opt(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0),
        "imu_topics":      Opt(lambda v: v is not None),   # list of topic names, main first
        "imu_gt_offsets_json": Opt(lambda v: v is None or isinstance(v, str)),  # path to per-IMU GT-align offsets
        "max_gt_speed":    Opt(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0),
    }

    def __init__(self, config: DictConfig):
        self.check_config(config)
        super().__init__(config)

        device      = config.device
        dtype       = config.dtype
        acc_sigma   = float(config.acc_sigma)
        gyro_sigma  = float(config.gyro_sigma)
        num_imus    = int(config.num_imus)
        num_joints  = int(config.num_joints)
        target_rate = float(config.target_rate)
        gt_source   = str(config.gt_source)
        debias  = bool(OmegaConf.select(config, "debias", default=False))
        gravity = float(OmegaConf.select(config, "gravity", default=9.81007))
        if not debias and OmegaConf.select(config, "calibrater", default=None) is not None:
            raise ValueError(
                "`calibrater` has no effect without `debias: true` — remove it or set `debias: true`."
            )

        if num_joints != len(_EXPECTED_JOINTS):
            raise ValueError(
                f"ANYmal_Sequence expects num_joints={len(_EXPECTED_JOINTS)}, "
                f"got num_joints={num_joints}"
            )
        # imu_topics selects which sensor-box IMU topics to load for stacked IMUs
        # (index 0 must be the main base IMU and the rest are auxiliary IMUs).
        # Default is main-IMU-only
        imu_topics = OmegaConf.select(config, "imu_topics", default=None)
        imu_topics = [str(t) for t in imu_topics] if imu_topics is not None else [_MAIN_IMU]
        if imu_topics[0] != _MAIN_IMU:
            raise ValueError(f"imu_topics[0] must be the base IMU {_MAIN_IMU!r}, got {imu_topics[0]!r}")
        if len(imu_topics) != num_imus:
            raise ValueError(f"num_imus={num_imus} but imu_topics has {len(imu_topics)} entries")

        mission_dir = os.path.join(str(config.data_root), str(config.seq_name))
        data_dir    = os.path.join(mission_dir, "data")
        tf_path     = os.path.join(mission_dir, "metadata", "tf.yaml")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"ANYmal zarr data dir not found: {data_dir}")
        if not os.path.isfile(tf_path):
            raise FileNotFoundError(f"ANYmal tf.yaml not found: {tf_path}")

        root = zarr.open_group(data_dir, mode="r")
        tf   = OmegaConf.load(tf_path)   # complete static transform tree

        # For each selected IMU, reads and rotates each into the robot base frame
        imu_raw = []
        for topic in imu_topics:
            if topic not in root:
                raise RuntimeError(f"ANYmal mission {config.seq_name!r}: missing IMU topic {topic!r}")
            imu_group = root[topic]
            t_imu = np.asarray(imu_group["timestamp"][:], dtype=float)
            a_imu = np.asarray(imu_group["lin_acc"][:],   dtype=float)
            w_imu = np.asarray(imu_group["ang_vel"][:],   dtype=float)
            if t_imu.size == 0:
                raise RuntimeError(f"ANYmal mission {config.seq_name!r}: empty IMU topic {topic!r}")
            # base<-sensor rotation composed up the tf chain to base.
            R_bs = _T_base_from_tf(tf, topic).rotation().matrix().numpy()   # (3,3)
            a_base = a_imu @ R_bs.T
            w_base = w_imu @ R_bs.T
            imu_raw.append({"time": t_imu, "acc": a_base, "gyro": w_base})

        # joints
        if _STATE not in root:
            raise RuntimeError(f"ANYmal mission {config.seq_name!r}: missing {_STATE!r}")
        state_group = root[_STATE]
        # Joint columns are assumed in _EXPECTED_JOINTS order
        j_time = np.asarray(state_group["timestamp"][:], dtype=float)
        j_pos  = np.asarray(state_group["joint_positions"][:],  dtype=float)   # (M,12) rad
        j_vel  = np.asarray(state_group["joint_velocities"][:], dtype=float)   # (M,12) rad/s
        j_eff  = np.asarray(state_group["joint_efforts"][:],    dtype=float)   # (M,12) Nm
        if j_pos.shape[1] != len(_EXPECTED_JOINTS):
            raise ValueError(f"{_STATE} has {j_pos.shape[1]} leg joints, expected {len(_EXPECTED_JOINTS)}")

        if gt_source == "cpt7_ie":
            gt = self._build_GNSS_gt(root, tf, config)
        elif gt_source == "state_estimator":
            gt = self._build_state_estim_gt(root, config)
        else:
            raise ValueError(f"unknown gt_source {gt_source!r}")

        # Optional per-IMU IMU<->GT time-offset shift from precomputed, constant-per-mission offsets JSON.
        # Applied ONLY for cpt7_ie GT (the state_estimator GT shares the ANYmal onboard clock. The
        # joints/proprioception from state estimate use the main-IMU (anymal_imu) onboard clock so
        # same offset applied. If no JSON path is provided, NO offsets are applied.
        # JSON lives at the dataset root (DATA_ROOT/grand_tour/) rather than in the repo,
        # and is regenerated with Scripts/measure_anymal_imu_gt_offsets_per_imu.py.
        offsets_path = OmegaConf.select(config, "imu_gt_offsets_json", default=None)
        if gt_source == "cpt7_ie" and offsets_path is not None:
            per_imu_off = ANYmal_Sequence._load_imu_gt_offsets(str(offsets_path))
            for i, topic in enumerate(imu_topics):
                imu_raw[i]["time"] = imu_raw[i]["time"] + per_imu_off[topic]   # each IMU, its own offset
            j_time = j_time + per_imu_off[_MAIN_IMU]                           # joints ride the main-IMU clock

        # Synchronize onto one uniform timeline, then resample GT, IMUs, and joints
        sync_times = synchronize_timeline(
            [gt.time] + [s["time"] for s in imu_raw] + [j_time],
            target_rate=target_rate,
        )
        N = len(sync_times)
        K = num_imus

        # GT: pos linear, quat SLERP, vel linear (all already in the base/world frame).
        gt_res = resample_list(
            {"pos":  {"time": gt.time, "data": gt.pos},
             "quat": {"time": gt.time, "data": gt.quat, "quat": True},
             "vel":  {"time": gt.time, "data": gt.vel}},
            sync_times,
            )
        gt_trans = gt_res["pos"].to(torch.float64)                 # (N,3) base position, world
        gt_quat  = gt_res["quat"].tensor().to(torch.float64)       # (N,4) xyzw base orientation
        vel_np   = gt_res["vel"].to(torch.float64).numpy()         # (N,3) base velocity, world

        # resample IMUs to grid
        acc_list, gyro_list = [], []
        for s in imu_raw:
            r = resample_list(
                {"acc": {"time": s["time"], "data": s["acc"]},
                 "gyro": {"time": s["time"], "data": s["gyro"]}},
                sync_times,
            )
            acc_list.append(r["acc"].to(torch.float64))
            gyro_list.append(r["gyro"].to(torch.float64))
        acc  = torch.stack(acc_list,  dim=1)             # (N, K, 3)
        gyro = torch.stack(gyro_list, dim=1)             # (N, K, 3)

        # Debias the base IMU against rot_wb.
        # Only corrected_* changes; acc/gyro remain as measured.
        time_ns = (torch.from_numpy(sync_times).double() * 1.0e9).round().to(torch.int64)
        if debias:
            calibrater = build_calibrater(config, gravity)
            rot_wb     = pp.SO3(gt_quat)
            gt_vel_w   = torch.from_numpy(vel_np)
            # Only the base IMU (K=0) is debiased; any stacked IMUs stay raw in corrected_*.
            a_0, g_0, report = calibrater(acc[:, 0, :], gyro[:, 0, :], rot_wb, gt_vel_w, time_ns=time_ns)
            corrected_acc  = acc .clone()                                # (N, K, 3)
            corrected_gyro = gyro.clone()
            corrected_acc [:, 0, :] = a_0
            corrected_gyro[:, 0, :] = g_0
            self.debias_report          = report
            self.debias_reports_per_imu = [report] + [disabled_report(N)] * (K - 1)
        else:
            # None prevents a corrected-stream request from silently receiving raw IMU.
            corrected_acc = corrected_gyro = None
            self.debias_report          = disabled_report(N)
            self.debias_reports_per_imu = [self.debias_report] * K

        # resample joints
        jr = resample_list(
            {"pos": {"time": j_time, "data": j_pos},
             "vel": {"time": j_time, "data": j_vel},
             "eff": {"time": j_time, "data": j_eff}},
            sync_times,
        )

        # store everything (time-major (N,K,...) / (N,J,1))
        self._time_ns       = time_ns.view(N, 1, 1).expand(N, K, 1).contiguous()
        self._joint_time_ns = time_ns.view(N, 1)

        self._acc  = acc.to(dtype=dtype)                            # (N, K, 3) as measured
        self._gyro = gyro.to(dtype=dtype)                           # (N, K, 3) as measured
        self._corrected_acc  = None if corrected_acc  is None else corrected_acc .to(dtype=dtype)
        self._corrected_gyro = None if corrected_gyro is None else corrected_gyro.to(dtype=dtype)

        eye3 = torch.eye(3, dtype=dtype)
        self._acc_cov  = (eye3 * acc_sigma  ** 2).expand(N, K, 3, 3).contiguous()
        self._gyro_cov = (eye3 * gyro_sigma ** 2).expand(N, K, 3, 3).contiguous()

        self._gt_poses_data = torch.cat([gt_trans, gt_quat], dim=-1).to(dtype=dtype)
        self._gt_vel_w      = torch.from_numpy(vel_np).to(dtype=dtype)

        self._joint_pos = jr["pos"].to(dtype=dtype).unsqueeze(-1)   # (N, 12, 1)
        self._joint_vel = jr["vel"].to(dtype=dtype).unsqueeze(-1)
        self._joint_eff = jr["eff"].to(dtype=dtype).unsqueeze(-1)

        self.to(device=device, dtype=None)

    @staticmethod
    def _load_imu_gt_offsets(path: str) -> dict:
        """Per-IMU constant GT-alignment offsets (seconds to ADD to that IMU's timestamps),
        keyed by IMU topic name, read from the JSON's ``per_imu[<name>]['offset_to_add_s']``.
        Lookup is direct by topic name, so a missing topic raises KeyError at use."""
        with open(path) as f:
            per_imu = json.load(f)["per_imu"]
        return {name: float(d["offset_to_add_s"]) for name, d in per_imu.items()}

    @staticmethod
    def _build_GNSS_gt(root, tf, config: DictConfig) -> "_GTData":
        """GNSS-INS (Inertial Explorer) GT. It solves for ``cpt7_imu`` == ``box_base``, so the pose
        is transferred box->base (static tf) and the box velocity gets the rigid-body lever arm to
        the base. ``twist_lin`` is reported in cpt7's fixed [180,0,90]-deg nav frame (not in tf.yaml
        but a fixed NED <-> ENU constant)."""
        if _GT_CPT7 not in root:
            raise RuntimeError(
                f"ANYmal mission {config.seq_name!r}: gt_source='cpt7_ie' but "
                f"{_GT_CPT7!r} is absent (possibly a GNSS-denied mission)"
                )
        cpt7_group = root[_GT_CPT7]
        t         = np.asarray(cpt7_group["timestamp"][:],  dtype=float)
        box_pos   = np.asarray(cpt7_group["pose_pos"][:],   dtype=float)     # ENU, box_base point
        box_quat  = np.asarray(cpt7_group["pose_orien"][:], dtype=float)     # xyzw, ENU
        twist_nav = np.asarray(cpt7_group["twist_lin"][:],  dtype=float)     # cpt7 fixed nav frame

        # pose: box_ENU -> base_ENU  (T_enu_base = T_enu_boxbase @ inv(T_base_boxbase))
        box_pose  = pp.SE3(torch.cat([torch.tensor(box_pos,  dtype=torch.float64),
                                      torch.tensor(box_quat, dtype=torch.float64)], dim=-1))
        base_pose = box_pose @ _se3_from_tf(tf, "box_base").Inv()
        base_pos  = base_pose.translation().numpy()
        base_quat = base_pose.rotation().tensor().numpy()          # xyzw

        # velocity: twist (nav frame) -> ENU box velocity, then + lever arm omega x (base - box).
        R_cpt7_enu = _Rot.from_euler("xyz", [180, 0, 90], degrees=True).as_matrix()  # nav -> ENU
        vel = (R_cpt7_enu @ twist_nav.T).T                                            # ENU box velocity
        r_world = base_pos - box_pos                                                  # base - box (ENU)
        # omega = base-orientation derivative in ENU (forward difference). This GT's attitude seems
        # clean (max ~2.7 rad/s) and high-freq enough that this is fine.
        # A central difference would need uniform-rate resampling first; with the small
        # lever arm the forward difference is sufficient.
        q  = pp.SO3(torch.tensor(base_quat, dtype=torch.float64))
        dR = (q[1:] @ q[:-1].Inv()).Log().numpy()                                     # (M-1,3) ENU incr. rot
        omega = np.zeros_like(r_world)
        if len(omega) > 1:
            omega[:-1] = dR / np.diff(t)[:, None]
            omega[-1]  = omega[-2]
        vel = vel + np.cross(omega, r_world)                                          # base velocity, ENU
        gt = _GTData(time=t, pos=base_pos, quat=base_quat, vel=vel)

        # Plausible-GT guard trims to the largest span of physically-plausible poses
        # (by position-to-position step speed). A high step-speed means bad GT
        # (e.g. Inertial Explorer divergence) despite the GNSS-INS-derived GT being the
        # proper clean path for supervision. The state_estimator path skips this since
        # its odometry position has sparse relocalization jumps and only its orientation
        # are consumed during training, so trimming on position jumps would discard good
        # IMU, vel, and orientation over independent estimator solution pose jumps.
        return ANYmal_Sequence._apply_gt_trim(
            gt, str(config.seq_name),
            max_gt_speed=float(OmegaConf.select(config, "max_gt_speed", default=4.0)))

    @staticmethod
    def _build_state_estim_gt(root, config: DictConfig) -> "_GTData":
        """Build GT from onboard proprioceptive state-estimator (for GNSS-denied missions for
        unsupervised use only, primarily for orientation over small windows). Estimates
        are in the ``base`` frame; GT velocity needs only a body->world rotation."""
        g = root[_STATE]
        t      = np.asarray(g["timestamp"][:],  dtype=float)
        pos    = np.asarray(g["pose_pos"][:],   dtype=float)       # odom, base
        quat   = np.asarray(g["pose_orien"][:], dtype=float)       # xyzw
        v_body = np.asarray(g["twist_lin"][:],  dtype=float)       # base body-frame velocity
        vel = (pp.SO3(torch.tensor(quat, dtype=torch.float64))
               @ torch.tensor(v_body, dtype=torch.float64)).numpy()
        return _GTData(time=t, pos=pos, quat=quat, vel=vel)

    @staticmethod
    def _apply_gt_trim(gt: "_GTData", seq_name: str, max_gt_speed: float, margin_time: float = 0.5) -> "_GTData":
        """Trim the GT to the largest contiguous span whose base pose-to-pose speed is physically
        plausible. GNSS-INS (Inertial Explorer) solutions can diverge at sequence boundaries (e.g.
        LMB-1's final ~0.4 s ramps to ~36 m/s) from lack of trailing GPS.
        <50% plausible is a hard error (GT unusable, not just glitchy at an edge).
        Note:   This only validates position step-speed since the attitude was clean across whole
                dataset (max ~2.7 rad/s)."""
        dt = np.diff(gt.time)
        N = len(gt.time)
        step_speed = np.linalg.norm(np.diff(gt.pos, axis=0), axis=1) / dt
        valid = np.append(step_speed <= max_gt_speed, step_speed[-1] <= max_gt_speed)  # (N,)
        start, stop = _largest_true_span(valid)
        if not ((start > 0) or (stop < N)):
            return gt                                     # nothing to trim
        margin = max(1, int(round(margin_time / float(np.median(dt)))))   # trim including margin (default 0.5 s)
        if start > 0: start = min(start + margin, N)
        if stop < N: stop = max(stop - margin, 0)
        if (stop - start) < 0.5 * N:
            raise RuntimeError(
                f"ANYmal {seq_name!r}: physically-plausible GT span {stop-start}/{N} < 50% "
                f"(max step-speed {step_speed.max():.1f} m/s), mission's GT unusable"
            )
        print(f"[ANYmal] {seq_name}: GT-plausibility trim -> [{start},{stop}) of {N} "
              f"({100*(N-(stop-start))/N:.2f}% dropped; max step-speed {step_speed.max():.1f} m/s)")
        return _GTData(time=gt.time[start:stop], pos=gt.pos[start:stop], quat=gt.quat[start:stop], vel=gt.vel[start:stop])

    # ------------------------------------------------------------------
    # ISequence protocol
    # ------------------------------------------------------------------

    @property
    def seq_name(self) -> str:
        """Sequence identifier — the per-sequence key in orientation pickles (Sequence.ExternalOrientation)."""
        return str(self.config.seq_name)

    def __len__(self) -> int:
        return int(self._time_ns.shape[0])

    def __getitem__(self, index: slice | int) -> IMUJointsData_w_GT:
        if isinstance(index, int):
            index = slice(index, index + 1)
        return IMUJointsData_w_GT(
            time_ns        = self._time_ns      [index].unsqueeze(0),
            acc            = self._acc          [index].unsqueeze(0),
            gyro           = self._gyro         [index].unsqueeze(0),
            acc_cov        = self._acc_cov      [index].unsqueeze(0),
            gyro_cov       = self._gyro_cov     [index].unsqueeze(0),
            gt_poses       = pp.SE3(self._gt_poses_data[index].unsqueeze(0)),
            gt_vel_w       = self._gt_vel_w     [index].unsqueeze(0),
            corrected_acc  = None if self._corrected_acc  is None else self._corrected_acc [index].unsqueeze(0),
            corrected_gyro = None if self._corrected_gyro is None else self._corrected_gyro[index].unsqueeze(0),
            joint_time_ns  = self._joint_time_ns[index].unsqueeze(0),
            position       = self._joint_pos    [index].unsqueeze(0),
            velocity       = self._joint_vel    [index].unsqueeze(0),
            effort         = self._joint_eff    [index].unsqueeze(0),
        )

    def clip(self, start: int | None, end: int | None, step: int | None) -> None:
        sl = slice(start, end, step)
        self._time_ns        = self._time_ns       [sl]
        self._acc            = self._acc           [sl]
        self._gyro           = self._gyro          [sl]
        self._acc_cov        = self._acc_cov       [sl]
        self._gyro_cov       = self._gyro_cov      [sl]
        self._gt_poses_data  = self._gt_poses_data [sl]
        self._gt_vel_w       = self._gt_vel_w      [sl]
        self._joint_time_ns  = self._joint_time_ns [sl]
        self._joint_pos      = self._joint_pos     [sl]
        self._joint_vel      = self._joint_vel     [sl]
        self._joint_eff      = self._joint_eff     [sl]
        if self._corrected_acc is not None:
            self._corrected_acc  = self._corrected_acc [sl]
            self._corrected_gyro = self._corrected_gyro[sl]

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        def _move(t: torch.Tensor) -> torch.Tensor:
            kwargs: dict = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None and t.dtype.is_floating_point:
                kwargs["dtype"] = dtype
            return t.to(**kwargs) if kwargs else t

        self._time_ns        = _move(self._time_ns)
        self._acc            = _move(self._acc)
        self._gyro           = _move(self._gyro)
        self._acc_cov        = _move(self._acc_cov)
        self._gyro_cov       = _move(self._gyro_cov)
        self._gt_poses_data  = _move(self._gt_poses_data)
        self._gt_vel_w       = _move(self._gt_vel_w)
        self._joint_time_ns  = _move(self._joint_time_ns)
        self._joint_pos      = _move(self._joint_pos)
        self._joint_vel      = _move(self._joint_vel)
        self._joint_eff      = _move(self._joint_eff)
        if self._corrected_acc is not None:
            self._corrected_acc  = _move(self._corrected_acc)
            self._corrected_gyro = _move(self._corrected_gyro)
