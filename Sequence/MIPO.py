"""MIPO multi-IMU + joint-state sequence loader for SSIO.

One MIPO trial (a deterministic per-topic HDF5 produced offline from a rosbag)
holds five IMUs, sixteen joints, and a motion-capture GT pose, each on its own
native timeline:

    /unitree_hardware/imu     main body IMU, ~400 Hz   (index 0 of the K stack)
    /WT901_4{7,8,9,50}_Data   four auxiliary IMUs, ~200 Hz
    /unitree_hardware/joint_foot  16 joints (pos/vel/effort), ~400 Hz
    /natnet_ros/Shuo_Go1/pose     GT body pose, ~60-70 Hz (no velocity stream)

The per-topic group layout is ``time`` + ``linear_acceleration (F, 3)`` +
``angular_velocity (F, 3)`` for an IMU, and ``time`` + ``position (G, 3)`` +
``orientation (G, 4)`` xyzw for the GT; ``attrs['imu_topics']`` is the json IMU
topic list and ``attrs['gt_topic']`` names the GT group. See
``Doc/mipo_hdf5_format.md`` for the full HDF5 schema.

The loader reads the raw streams, applies kalibr calibration (per-sequence GT
time offset + body←sensor extrinsic, both read from ``calibration_file``),
synchronizes everything onto one uniform ``target_rate`` timeline by
interpolation (SLERP for the GT quaternion), derives GT velocity by
finite-difference + Savitzky-Golay, and emits ``IMUJointsData_w_GT`` — the
K-IMU stack, the J-joint stack, and body GT.

Conventions (match the other ``Sequence/`` loaders):
    * IMU is body-frame; GT is world/global; GT quaternion is xyzw (pypose order).
    * After ``_apply_kalibr_calibration`` the stored GT rotation is
      ``R_world<-IMU0_sensor``, not ``R_world<-body``: the consensus ``T_BS`` is
      applied to the GT pose (``q_WS = q_WB * q_BS``) so that raw IMU data stays
      untouched. World
      velocity is differentiated from world GT position. This satisfies PyPose's
      z-up ``g_world = [0,0,+gravity]`` convention (same as EuRoC), so the
      ``Sequence.debias`` estimators only ever remove a constant bias — they never
      re-orient axes. No sign flip is applied to the accelerometer. Only IMU 0 can
      be debiased; the aux IMUs have no extrinsics, so there is no shared frame to
      remove gravity in.

Joint columns are reordered to a fixed canonical order (``_EXPECTED_JOINTS``)
so a given joint channel maps to the same physical joint across all sequences
— required because the joints network flattens J into Conv1d channels. They
are then z-scored per group (``joint_norm``, on by default) with the statistics
the released checkpoints were trained with; ``joint_norm: false`` keeps them raw.

Outdoor ``ahrs_gps`` sequences use onboard attitude as orientation-only GT and
retain phone GPS as a separate ENU shape-evaluation track.
"""

import json
import os

import h5py
import numpy as np
import torch
import pypose as pp
from omegaconf import DictConfig, OmegaConf
from scipy.signal import savgol_filter

from Interface.base         import Opt
from Interface.sequence     import ISequence
from Interface.data         import IMUJointsData_w_GT
from Sequence.debias        import (STATUS_STATIC, SelectionDebias,
                                    build_calibrater, disabled_report,
                                    is_calibrater_node)
from Sequence.time_sync     import synchronize_timeline, resample_list


# Canonical joint order: 12 revolute leg joints then 4 foot-contact "joints"
# (the *_foot entries are fixed joints carrying contact-sensor signal).
_EXPECTED_JOINTS = [
    "FL0", "FL1", "FL2", "FR0", "FR1", "FR2",
    "RL0", "RL1", "RL2", "RR0", "RR1", "RR2",
    "FL_foot", "FR_foot", "RL_foot", "RR_foot",
]

# Joint standardization groups. One shared (mean, std) per group keeps the
# relative magnitudes within a group while pulling the contact effort (raw sensor
# counts, ~233 ± 289) down to the scale of the leg channels.
_LEG_JOINTS     = slice(0, 12)
_CONTACT_JOINTS = slice(12, 16)

# Statistics the released MIPO checkpoints were trained with: computed on the raw
# joints of the paper's MIPO_Train split (36 sequences). The network's input
# contract, so they live here rather than in a data file; ``joint_stats_file``
# overrides them (same keys) for a model trained on other statistics.
_DEFAULT_JOINT_STATS = {
    "leg_pos":     {"mean": -0.34287641059613677,   "std": 1.1801247648104463},
    "leg_vel":     {"mean": -0.00014636684348431014, "std": 2.182366737941409},
    "leg_eff":     {"mean": 2.130135116730853,       "std": 5.217969268767851},
    "contact_eff": {"mean": 233.3156380113291,       "std": 288.7649325149753},
}


def _standardize_joints(
    pos  : torch.Tensor,     # (N, J, 1)
    vel  : torch.Tensor,     # (N, J, 1)
    eff  : torch.Tensor,     # (N, J, 1)
    stats: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group z-score of the joint channels (``_EXPECTED_JOINTS`` order).

    Leg position / velocity / effort each use their own group; the foot-contact
    channels carry signal only in effort, so their position and velocity pass through.
    """
    def zscore(x: torch.Tensor, group: str) -> torch.Tensor:
        return (x - stats[group]["mean"]) / stats[group]["std"]

    pos, vel, eff = pos.clone(), vel.clone(), eff.clone()
    pos[:, _LEG_JOINTS]     = zscore(pos[:, _LEG_JOINTS],     "leg_pos")
    vel[:, _LEG_JOINTS]     = zscore(vel[:, _LEG_JOINTS],     "leg_vel")
    eff[:, _LEG_JOINTS]     = zscore(eff[:, _LEG_JOINTS],     "leg_eff")
    eff[:, _CONTACT_JOINTS] = zscore(eff[:, _CONTACT_JOINTS], "contact_eff")
    return pos, vel, eff

# Body-frame tilt correction for onboard attitude estimates.
_AHRS_TILT_ROLL_DEG  = 0.795
_AHRS_TILT_PITCH_DEG = 0.131


def _lla_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray,
                refloc: np.ndarray) -> np.ndarray:
    """Project WGS84 (lat deg, lon deg, alt m) to local-tangent ENU at refloc.

    ENU axis rows, refloc = [lat0, lon0, alt0]. Implemented with plain numpy so
    the loader does not depend on pyproj.
    """
    a, e2 = 6378137.0, 6.69437999014e-3          # WGS84 semi-major axis, ecc^2

    def ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
        n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        return np.stack(
            [(n + alt) * np.cos(lat) * np.cos(lon),
             (n + alt) * np.cos(lat) * np.sin(lon),
             (n * (1.0 - e2) + alt) * np.sin(lat)], axis=-1)

    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    p = ecef(lat, lon, np.asarray(alt_m, dtype=float))
    lat0, lon0 = np.radians(float(refloc[0])), np.radians(float(refloc[1]))
    p0 = ecef(np.float64(lat0), np.float64(lon0), np.float64(refloc[2]))
    rot = np.array([
        [-np.sin(lon0),                np.cos(lon0),               0.0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [ np.cos(lat0) * np.cos(lon0),  np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
    ])
    return (p - p0) @ rot.T


def _compute_gt_velocity(
    pos: np.ndarray, t: np.ndarray, window_time: float,
) -> np.ndarray:
    """World-frame GT velocity by central finite-difference + Savitzky-Golay.

    MIPO motion capture provides no velocity stream, so velocity is
    differentiated from the resampled GT position. ``window_time`` (seconds)
    sets the SavGol smoothing window; <= 0 disables smoothing.
    """
    n = len(t)
    dt  = np.diff(t)
    vel = np.zeros_like(pos)
    if n >= 3:
        vel[1:-1] = (pos[2:] - pos[:-2]) / (dt[1:, None] + dt[:-1, None])
    vel[0]  = (pos[1] - pos[0])   / dt[0]
    vel[-1] = (pos[-1] - pos[-2]) / dt[-1]

    if window_time > 0.0 and n > 11:
        period = float(np.median(dt))
        wl     = int(np.clip(round(window_time / period), 5, n))
        if wl % 2 == 0:
            wl -= 1
        if wl >= 5:
            for axis in range(3):
                vel[:, axis] = savgol_filter(vel[:, axis], wl, polyorder=3, mode="mirror")
    return vel


def _load_mipo_hdf5(path: str, num_imus: int, num_joints: int,
                    gt_source: str = "file") -> dict:
    """Read raw per-topic streams from a MIPO HDF5 file.

    Returns numpy arrays at native rates. IMUs are ordered main-body-first;
    joint columns are permuted to ``_EXPECTED_JOINTS`` order.

    ``gt_source`` selects where the GT streams come from:
      - ``"file"``: the HDF5's ``gt_topic`` group (indoor motion capture, or a
        pre-written outdoor GT group), position + orientation on one timeline.
      - ``"ahrs_gps"``: GT orientation from the main-body IMU's onboard
        attitude filter (``orientation`` dataset, IMU timeline) and GT
        translation from the raw phone GPS (``/phone/Position``, phone-clock
        timeline, returned as lat/lon/alt for the caller to project and
        time-shift). Used for outdoor sequences, which have no robot GT.
    """
    with h5py.File(path, "r") as f:
        topics_attr = f.attrs["imu_topics"]
        if isinstance(topics_attr, bytes):
            topics_attr = topics_attr.decode("utf-8")
        imu_topics = json.loads(topics_attr)
        if len(imu_topics) != num_imus:
            raise ValueError(
                f"MIPO HDF5 {path!r} has {len(imu_topics)} IMU topics, "
                f"config expects num_imus={num_imus}"
            )
        if MIPO_Sequence.MAIN_IMU_TOPIC not in imu_topics:
            raise ValueError(
                f"Main body IMU topic {MIPO_Sequence.MAIN_IMU_TOPIC!r} "
                f"not found in HDF5 imu_topics {imu_topics}"
            )
        # Main body IMU first — it is the canonical IMU 0 for preintegration/PGO.
        imu_topics = [MIPO_Sequence.MAIN_IMU_TOPIC] + [
            t for t in imu_topics if t != MIPO_Sequence.MAIN_IMU_TOPIC
        ]

        imu = []
        for topic in imu_topics:
            g = f[topic.lstrip("/")]
            stream = {
                "time": np.asarray(g["time"][()],                dtype=float),
                "acc":  np.asarray(g["linear_acceleration"][()], dtype=float),
                "gyro": np.asarray(g["angular_velocity"][()],    dtype=float),
            }
            if stream["time"].size == 0:
                raise RuntimeError(f"MIPO HDF5 {path!r}: empty IMU topic {topic!r}")
            imu.append(stream)

        gps = None
        if gt_source == "file":
            gt_topic = f.attrs.get("gt_topic", MIPO_Sequence.DEFAULT_GT_TOPIC)
            if isinstance(gt_topic, bytes):
                gt_topic = gt_topic.decode("utf-8")
            gg = f[gt_topic.lstrip("/")]
            gt_time = np.asarray(gg["time"][()],        dtype=float)
            gt_pos  = np.asarray(gg["position"][()],    dtype=float)
            gt_quat = np.asarray(gg["orientation"][()], dtype=float)   # xyzw (ROS)
            if gt_time.size == 0:
                raise RuntimeError(f"MIPO HDF5 {path!r}: empty GT topic {gt_topic!r}")
        elif gt_source == "ahrs_gps":
            # Orientation: onboard attitude filter of the main-body IMU, on the
            # IMU timeline. Translation is filled later from GPS; gt_pos stays
            # None here (the caller projects lat/lon/alt to ENU and time-shifts
            # the phone clock onto the rosbag clock).
            mg = f[MIPO_Sequence.MAIN_IMU_TOPIC.lstrip("/")]
            gt_time = np.asarray(mg["time"][()],        dtype=float)
            gt_quat = np.asarray(mg["orientation"][()], dtype=float)   # xyzw (ROS)
            gt_pos  = None
            norms = np.linalg.norm(gt_quat, axis=1)
            if norms.min() < 0.5:
                raise RuntimeError(
                    f"MIPO HDF5 {path!r}: main-IMU orientation is not populated "
                    f"(min quaternion norm {norms.min():.3f}); ahrs_gps GT mode "
                    f"requires the onboard attitude filter stream"
                )
            pg = f["phone/Position"]
            gps = {
                "time": np.asarray(pg["time"][()], dtype=float),
                "lat":  np.asarray(pg["lat"][()],  dtype=float),
                "lon":  np.asarray(pg["lon"][()],  dtype=float),
                "alt":  np.asarray(pg["alt"][()],  dtype=float),
            }
            if gps["time"].size < 2:
                raise RuntimeError(f"MIPO HDF5 {path!r}: /phone/Position missing or too short")
        else:
            raise ValueError(f"unknown gt_source {gt_source!r}")

        jg    = f[MIPO_Sequence.JOINT_TOPIC.lstrip("/")]
        names = [n.decode() if isinstance(n, bytes) else n for n in jg["names"][()]]
        if len(names) != num_joints:
            raise ValueError(
                f"MIPO HDF5 {path!r} has {len(names)} joints, "
                f"config expects num_joints={num_joints}"
            )
        # Permute joint columns to the canonical order; missing → ValueError.
        perm = [names.index(j) for j in _EXPECTED_JOINTS]
        joint = {
            "time": np.asarray(jg["time"][()],         dtype=float),
            "pos":  np.asarray(jg["position"][()],     dtype=float)[:, perm],
            "vel":  np.asarray(jg["velocity"][()],     dtype=float)[:, perm],
            "eff":  np.asarray(jg["effort"][()],       dtype=float)[:, perm],
        }
        if joint["time"].size == 0:
            raise RuntimeError(f"MIPO HDF5 {path!r}: empty joint topic")

    return {"imu": imu, "gt_time": gt_time, "gt_pos": gt_pos,
            "gt_quat": gt_quat, "gps": gps, "joint": joint}


class _ExplicitWindowDebias(SelectionDebias):
    """Averages the constant bias over a caller-chosen ``[start, end)`` frame range.

    Only the outdoor ``ahrs_gps`` GT mode needs this: its ``gt_vel_w`` is all-zero
    by construction, so every GT-speed selector in :mod:`Sequence.debias` would
    judge the whole trot static. ``MIPO_Sequence.__init__`` picks the window from
    main-IMU gyro energy instead and passes it here, keeping MIPO on the one
    estimator interface (and so on the one ``DebiasReport``) as every other loader.

    ``window`` indexes the synchronized timeline, which all K IMUs share.
    """

    def __init__(self, gravity: float, *, window: tuple[int, int]):
        super().__init__(gravity)
        self.window = (int(window[0]), int(window[1]))

    def _select(self, acc, gyro, speed, *, time_ns=None):
        del speed, time_ns
        start, end = self.window
        n = int(acc.shape[0])
        if not (0 <= start < end <= n):
            raise ValueError(
                f"{self.name}: invalid window [{start}, {end}) for n={n}"
            )
        idx = torch.arange(start, end, device=acc.device)
        return idx, STATUS_STATIC, idx


class MIPO_Sequence(ISequence[IMUJointsData_w_GT]):
    """One MIPO trial — five IMUs, sixteen joints, body GT — as ``IMUJointsData_w_GT``."""

    MAIN_IMU_TOPIC   = "/unitree_hardware/imu"
    JOINT_TOPIC      = "/unitree_hardware/joint_foot"
    DEFAULT_GT_TOPIC = "/natnet_ros/Shuo_Go1/pose"

    CONFIG_SPEC = {
        "data_root":   lambda v: isinstance(v, str) and len(v) > 0,
        # Full relative path incl. subdir + ".h5" — MUST equal the orientation pickle key.
        "seq_name":    lambda v: isinstance(v, str) and len(v) > 0,
        "device":      lambda v: isinstance(v, torch.device),
        "dtype":       lambda v: isinstance(v, torch.dtype),
        # Preintegration noise stds: accel m/s^2, gyro rad/s.
        "acc_sigma":   lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "gyro_sigma":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        "num_imus":    lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1,
        "num_joints":  lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1,
        # Hz; 200.0 to match the orientation pickle timeline.
        "target_rate": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
        # SavGol window (s) for the differentiated GT velocity; 0 disables smoothing.
        "velocity_window_time": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0,
        # Calibration JSON (kalibr): per-sequence GT time offset + consensus T_BS.
        # Required for indoors and must be absent for outdoor.
        "calibration_file": lambda v: isinstance(v, str) and len(v) > 0,
        # Optional configs (defaults applied via OmegaConf.select in __init__).
        "outdoor":              Opt(lambda v: isinstance(v, bool)),
        "debias":               Opt(lambda v: isinstance(v, bool)),
        # ``{kind: span|set|imu, ...}``; null is invalid when debiasing. See
        # debias.build_calibrater. The MIPO dataset YAMLs set ``{kind: imu,
        # debias_gyro: false}``.
        "calibrater":           Opt(is_calibrater_node),
        "gravity":              Opt(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0),
        "outdoor_gt_source":    Opt(lambda v: v in ("file", "ahrs_gps")),
        "gps_time_offset_file": Opt(lambda v: isinstance(v, str) and len(v) > 0),
        "ahrs_tilt_correction": Opt(lambda v: isinstance(v, bool)),
        # Per-group joint z-scoring (default true: the released checkpoints expect it).
        # ``joint_stats_file`` replaces ``_DEFAULT_JOINT_STATS`` with a JSON of the same keys.
        "joint_norm":           Opt(lambda v: isinstance(v, bool)),
        "joint_stats_file":     Opt(lambda v: isinstance(v, str) and len(v) > 0),
    }

    def __init__(self, config: DictConfig):
        # `outdoor=True` skips kalibr time-offset + T_BS application entirely
        # (outdoor sequences have no mocap GT, so no extrinsic alignment applies
        # and there is no per-sequence time-offset calibration). When outdoor,
        # the `calibration_file` key must be omitted.
        #
        # `outdoor_gt_source` (outdoor only) selects the GT construction:
        #   - "file" (default): read the HDF5's gt_topic group as-is (e.g. a
        #     `/gps_ins_gt` topic written by an offline filter).
        #   - "ahrs_gps": GT orientation = onboard attitude filter of the main
        #     IMU (optionally tilt-corrected by the indoor-measured constant);
        #     GT translation = ZERO and GT velocity = zeros (outdoor has no robot
        #     GT — the phone was operator-carried, not robot-mounted). gt_poses
        #     is therefore a coherent orientation-only container. The zero
        #     velocity is also the physically correct rest-start prior for the
        #     adapt PGO first window (all outdoor sequences begin at rest). The
        #     phone GPS is held separately on `gps_track()` as a shape-evaluation
        #     reference, time-aligned onto the rosbag clock by the AHRS-yaw
        #     offset in `gps_time_offset_file`; offline evaluation reads it and
        #     aligns (Umeyama / start-pose) before comparing.
        outdoor = bool(OmegaConf.select(config, "outdoor", default=False))
        gt_source = str(OmegaConf.select(config, "outdoor_gt_source", default="file"))
        # calibration_file is conditionally required: needed indoors, and must be
        # ABSENT outdoors (no mocap GT, so no kalibr time-offset/T_BS). Drop it from
        # the spec when outdoor so a present key errors and an absent one is fine.
        spec = dict(self.CONFIG_SPEC)
        if outdoor:
            spec.pop("calibration_file", None)
        self.check_config(config, spec)
        super().__init__(config)

        if gt_source != "file" and not outdoor:
            raise ValueError("outdoor_gt_source is only valid with outdoor: true")
        if gt_source == "ahrs_gps" and "gps_time_offset_file" not in config:
            raise ValueError("outdoor_gt_source 'ahrs_gps' requires gps_time_offset_file")
        ahrs_tilt_correction = bool(
            OmegaConf.select(config, "ahrs_tilt_correction", default=True)
        )

        dtype       = config.dtype
        acc_sigma   = float(config.acc_sigma)
        gyro_sigma  = float(config.gyro_sigma)
        num_imus    = int(config.num_imus)
        num_joints  = int(config.num_joints)
        target_rate = float(config.target_rate)
        debias = bool(OmegaConf.select(config, "debias", default=False))
        joint_norm = bool(OmegaConf.select(config, "joint_norm", default=True))
        gravity        = float(OmegaConf.select(config, "gravity",        default=9.81007))
        # ``Sequence.debias.build_calibrater`` dereferences ``debias`` and
        # ``calibrater``; both are optional here, so hand it a node assembled from
        # what is present instead of ``config``.
        #
        # ``resolve=True`` before re-parenting: a detached subtree cannot resolve
        # ${...} interpolations against the root, and build_calibrater reads the
        # node's values. EuRoC/BlackBird/HumanoidOdom pass ``config`` itself and so
        # keep interpolation working; MIPO must not silently lose it.
        has_calibrater = "calibrater" in config and config.calibrater is not None
        if has_calibrater and "debias" not in config:
            raise ValueError(
                "`calibrater` has no effect without `debias: true` — remove it or "
                "set `debias`. Unlike EuRoC/BlackBird/HumanoidOdom, which require "
                "both keys, MIPO makes both optional, so a lone `calibrater` would "
                "silently pass raw IMU through."
            )
        if debias and gt_source == "ahrs_gps" and has_calibrater:
            raise ValueError(
                "outdoor_gt_source 'ahrs_gps' selects the explicit-window "
                "calibrater (gt_vel_w is all-zero, so GT-speed selectors would "
                "judge the whole trot static); remove `calibrater`."
            )
        calibrater_node = OmegaConf.create({
            "debias":     debias,
            "calibrater": (OmegaConf.to_container(config.calibrater, resolve=True)
                           if has_calibrater else None),
        })

        if num_joints != len(_EXPECTED_JOINTS):
            raise ValueError(
                f"MIPO_Sequence supports exactly {len(_EXPECTED_JOINTS)} joints, "
                f"got num_joints={num_joints}"
            )

        seq_path = os.path.join(str(config.data_root), str(config.seq_name))
        if not os.path.isfile(seq_path):
            raise FileNotFoundError(f"MIPO HDF5 not found: {seq_path}")

        raw = _load_mipo_hdf5(seq_path, num_imus, num_joints, gt_source=gt_source)

        # Outdoor sequences skip kalibr entirely (see __init__ note).
        if not outdoor:
            self._apply_kalibr_calibration(raw, config)

        # Phone GPS stays separate from orientation-only GT and is shifted into
        # the rosbag clock by the configured additive offset.
        self._gps_time_bag = None            # (M,) rosbag-clock seconds, or None
        self._gps_enu      = None            # (M, 3) local-tangent ENU, or None
        if gt_source == "ahrs_gps":
            gps = raw["gps"]
            seq_key = os.path.splitext(str(config.seq_name))[0]
            with open(str(config.gps_time_offset_file)) as f:
                yaw = json.load(f).get("sequences", {})
            ye = yaw.get(seq_key)
            if ye is None or not ye.get("trusted", True):
                # Untrusted (e.g. CMU garage GPS outage) or absent: the GPS
                # reference is unusable, but training does not need it (GT is
                # orientation-only), so proceed without a GPS track. Eval skips
                # sequences that have no trusted offset.
                print(
                    f"[MIPO] {config.seq_name}: no trusted AHRS-yaw GPS offset; "
                    f"GPS shape-eval track unavailable (training unaffected)"
                )
            else:
                use_begin = bool(ye.get("begin_reliable", False))
                offset = float(ye["offset_begin_s"] if use_begin else ye["offset_s"])
                refloc = np.array([gps["lat"][0], gps["lon"][0], gps["alt"][0]])
                self._gps_time_bag = torch.from_numpy(gps["time"] + offset)
                self._gps_enu = torch.from_numpy(
                    _lla_to_enu(gps["lat"], gps["lon"], gps["alt"], refloc)
                ).to(torch.float64)
        else:
            gt_trans_time = raw["gt_time"]
            gt_trans_np   = raw["gt_pos"]

        # --- synchronize all streams onto one uniform timeline ---
        # In ahrs_gps mode the GT is orientation-only, so only the attitude
        # timeline (main-IMU) plus the IMU/joint streams enter the intersection;
        # the phone GPS lives on the separate eval accessor and never touches
        # the training container. In "file" mode the GT translation shares
        # raw["gt_time"], which stays in the intersection.
        sync_times = synchronize_timeline(
            [raw["gt_time"]]
            + [s["time"] for s in raw["imu"]]
            + [raw["joint"]["time"]],
            target_rate=target_rate,
        )
        N = len(sync_times)
        K = num_imus

        resample_spec = {"ori": {"time": raw["gt_time"], "data": raw["gt_quat"], "quat": True}}
        if gt_source != "ahrs_gps":
            resample_spec["trans"] = {"time": gt_trans_time, "data": gt_trans_np}
        gt_res  = resample_list(resample_spec, sync_times)
        gt_quat = gt_res["ori"].tensor().to(torch.float64)      # (N, 4) xyzw
        if gt_source == "ahrs_gps":
            # Orientation-only GT: translation zeroed (the GPS shape-eval
            # reference is exposed via gps_track(), not packed into gt_poses).
            gt_trans = torch.zeros((N, 3), dtype=torch.float64)
        else:
            gt_trans = gt_res["trans"].to(torch.float64)        # (N, 3)

        if gt_source == "ahrs_gps":
            if ahrs_tilt_correction:
                # Constant attitude-filter tilt correction, indoor-measured
                # (see _AHRS_TILT_*_DEG). R_corrected = R_ahrs * Exp(v).
                v = torch.deg2rad(torch.tensor(
                    [_AHRS_TILT_ROLL_DEG, _AHRS_TILT_PITCH_DEG, 0.0],
                    dtype=torch.float64,
                ))
                corr = pp.so3(v).Exp()
                gt_quat = (pp.SO3(gt_quat) @ corr).tensor()
            # GT velocity is intentionally all-zero: there is no usable outdoor
            # velocity signal. Self-supervised losses ignore it, and the adapt
            # PGO first-window prior receives the physically correct rest-start
            # value (all outdoor sequences begin at rest). Do NOT differentiate
            # the GPS translation here: it is operator-carried, quantized, and
            # on a seconds-uncertain clock.
            vel_np = np.zeros((N, 3), dtype=np.float64)
        else:
            vel_np = _compute_gt_velocity(
                gt_trans.numpy(), sync_times, float(config.velocity_window_time),
            )

        acc_list, gyro_list = [], []
        for s in raw["imu"]:
            r = resample_list(
                {
                    "acc":  {"time": s["time"], "data": s["acc"]},
                    "gyro": {"time": s["time"], "data": s["gyro"]},
                },
                sync_times,
            )
            acc_list.append(r["acc"].to(torch.float64))
            gyro_list.append(r["gyro"].to(torch.float64))
        acc  = torch.stack(acc_list,  dim=1)                    # (N, K, 3)
        gyro = torch.stack(gyro_list, dim=1)                    # (N, K, 3)

        # Optional constant-bias removal, through the shared estimators in
        # ``Sequence.debias`` so MIPO reports like every other loader. `debias`
        # defaults False, and then NO correction stage runs at all: `corrected_*`
        # stays None (never aliased to the as-measured stream) and the published
        # report is ``disabled_report``.
        #
        # In ahrs_gps mode gt_vel_w is all-zero, so every GT-speed selector in
        # ``Sequence.debias`` would judge the whole trot static. Select the window
        # explicitly from main-IMU gyro energy instead (all outdoor sequences
        # start at rest). Note the ordering invariant: the tilt correction was
        # already applied to gt_quat above, so the bias computed here against the
        # corrected rotation is the residual per-sequence accelerometer bias — the
        # two corrections compose without double counting.

        # Only K=0 is debiased; the aux streams in ``corrected_*`` are raw copies.
        # IMU 0 is the stream preintegration and PGO run on (``dv_imu_input``),
        # and the network's own input is raw. ``_apply_kalibr_calibration`` rotated GT by the
        # single consensus ``T_BS``, so ``rot_ws`` is R_world<-IMU0_sensor, not
        # R_world<-body; and the four WT901 aux IMUs are never rotated into a
        # common frame (``_load_mipo_hdf5`` returns their raw streams and the
        # calibration file holds one T_BS). Debiasing IMU k>=1 against IMU 0's
        # rotation would subtract gravity expressed in the wrong frame, i.e.
        # estimate b_k + (R_S0_W - R_SkW) g_w and inject a constant of up to 1 g
        # into every aux sample -- invisible in a loss curve. Fix by sourcing
        # per-IMU extrinsics and rotating each stream into body first (see
        # ``Sequence/ANYmal.py``, which does exactly that before calibrating).
        time_ns = (torch.from_numpy(sync_times).double() * 1.0e9).round().to(torch.int64)
        rot_ws = pp.SO3(gt_quat)                                # (N,) R_world<-IMU0_sensor
        gt_vel_w = torch.from_numpy(vel_np)                     # (N, 3) world frame
        if debias:
            if gt_source == "ahrs_gps":
                win = round(2.0 * target_rate)                 # 2 s of samples
                g2 = gyro[:, 0, :].norm(dim=-1).pow(2).numpy()
                energy = np.convolve(g2, np.ones(win) / win, mode="valid")
                s0 = int(np.argmin(energy))
                still_rms = float(np.sqrt(energy[s0]))
                if still_rms > 0.1:
                    raise RuntimeError(
                        f"{config.seq_name}: stillest 2 s window has gyro rms "
                        f"{still_rms:.3f} rad/s; refusing motion-contaminated bias "
                        "calibration"
                    )
                calibrater = _ExplicitWindowDebias(gravity, window=(s0, s0 + win))
            else:
                calibrater = build_calibrater(calibrater_node, gravity)

            a_0, g_0, report = calibrater(acc[:, 0, :], gyro[:, 0, :], rot_ws, gt_vel_w, time_ns=time_ns)
            self._corrected_acc  = acc .clone()                         # (N, K, 3)
            self._corrected_gyro = gyro.clone()
            self._corrected_acc [:, 0, :] = a_0
            self._corrected_gyro[:, 0, :] = g_0
            self.debias_report          = report
            self.debias_reports_per_imu = [report] + [disabled_report(N)] * (K - 1)
        else:
            # No correction stage ran, so there is no corrected stream to serve.
            # NOT aliased to the as-measured stream: a consumer asking for
            # `corrected` must fail loudly rather than silently receive raw IMU.
            self._corrected_acc = self._corrected_gyro = None
            self.debias_report          = disabled_report(N)
            self.debias_reports_per_imu = [self.debias_report] * K

        jr = resample_list(
            {
                "pos": {"time": raw["joint"]["time"], "data": raw["joint"]["pos"]},
                "vel": {"time": raw["joint"]["time"], "data": raw["joint"]["vel"]},
                "eff": {"time": raw["joint"]["time"], "data": raw["joint"]["eff"]},
            },
            sync_times,
        )

        # --- store (time-major (N, K, ...) / (N, J, 1)) ---
        self._time_ns       = time_ns.view(N, 1, 1).expand(N, K, 1).contiguous()
        self._joint_time_ns = time_ns.view(N, 1)

        self._acc  = acc.to(dtype=dtype)                        # (N, K, 3) as measured
        self._gyro = gyro.to(dtype=dtype)                       # (N, K, 3) as measured
        if self._corrected_acc is not None:
            self._corrected_acc  = self._corrected_acc .to(dtype=dtype)
            self._corrected_gyro = self._corrected_gyro.to(dtype=dtype)

        eye3 = torch.eye(3, dtype=dtype)
        self._acc_cov  = (eye3 * acc_sigma  ** 2).expand(N, K, 3, 3).contiguous()
        self._gyro_cov = (eye3 * gyro_sigma ** 2).expand(N, K, 3, 3).contiguous()

        self._gt_poses_data = torch.cat([gt_trans, gt_quat], dim=-1).to(dtype=dtype)
        self._gt_vel_w      = torch.from_numpy(vel_np).to(dtype=dtype)

        self._joint_pos = jr["pos"].to(dtype=dtype).unsqueeze(-1)   # (N, J, 1)
        self._joint_vel = jr["vel"].to(dtype=dtype).unsqueeze(-1)
        self._joint_eff = jr["eff"].to(dtype=dtype).unsqueeze(-1)

        if joint_norm:
            joint_stats = _DEFAULT_JOINT_STATS
            stats_file = OmegaConf.select(config, "joint_stats_file", default=None)
            if stats_file is not None:
                with open(str(stats_file)) as f:
                    joint_stats = json.load(f)
            self._joint_pos, self._joint_vel, self._joint_eff = _standardize_joints(
                self._joint_pos, self._joint_vel, self._joint_eff, joint_stats)

        self.to(device=config.device, dtype=None)

    # Kalibr inlier set boundary for time offsets: fitted offsets above this
    # magnitude are not trusted and the '_consensus' offset (median over
    # inliers) is applied instead.
    _MAX_TRUSTED_OFFSET_S = 0.5

    @classmethod
    def _apply_kalibr_calibration(cls, raw: dict, config: DictConfig) -> None:
        """Apply kalibr per-sequence GT time offset and global T_BS extrinsic.

        Mutates ``raw["gt_time"]`` / ``raw["gt_quat"]`` / ``raw["gt_pos"]`` in place,
        applying calibration file extrinsics.
        """
        seq_key = os.path.splitext(os.path.basename(str(config.seq_name)))[0]
        with open(str(config.calibration_file)) as f:
            calib = json.load(f)
        if seq_key not in calib:
            raise ValueError(
                f"No calibration entry for sequence {seq_key!r} in {config.calibration_file}"
            )
        consensus = calib["_consensus"]
        consensus_off = consensus["time_offset_s"]
        entry = calib[seq_key]
        status = entry["status"]
        fitted = float(entry["time_offset_s"])
        if status == "optimized" and abs(fitted) <= cls._MAX_TRUSTED_OFFSET_S:
            raw["gt_time"] = raw["gt_time"] + fitted
        else:
            raw["gt_time"] = raw["gt_time"] + float(consensus_off)
            print(
                f"[MIPO] {seq_key}: kalibr fit not trusted (status={status!r}, "
                f"fitted={fitted:+.3f}s); applying consensus offset {float(consensus_off):+.3f}s"
            )

        tbs = consensus["T_BS"]
        R_BS = pp.SO3(torch.tensor(list(tbs["quaternion_xyzw"]), dtype=torch.float64))
        t_BS =        torch.tensor(list(tbs["translation"]),     dtype=torch.float64)
        q_WB = pp.SO3(torch.tensor(raw["gt_quat"], dtype=torch.float64))
        p_WB =        torch.tensor(raw["gt_pos"],  dtype=torch.float64)
        raw["gt_quat"] = (q_WB * R_BS).tensor().numpy()
        raw["gt_pos"]  = (p_WB + (q_WB @ t_BS)).numpy()

    # ------------------------------------------------------------------
    # ISequence protocol
    # ------------------------------------------------------------------

    # Always-present tensors, moved/clipped unconditionally. ``_corrected_acc`` /
    # ``_corrected_gyro`` are deliberately NOT here: they are None when no
    # correction stage ran, so every site handles them with an explicit guard.
    _TENSOR_ATTRS = (
        "_time_ns", "_acc", "_gyro", "_acc_cov", "_gyro_cov",
        "_gt_poses_data", "_gt_vel_w",
        "_joint_time_ns", "_joint_pos", "_joint_vel", "_joint_eff",
    )

    @property
    def seq_name(self) -> str:
        """Sequence identifier — the per-sequence key in orientation pickles (Sequence.ExternalOrientation)."""
        return str(self.config.seq_name)

    def __len__(self) -> int:
        return int(self._time_ns.shape[0])

    def gps_track(self):
        """Shape-evaluation GPS reference for ahrs_gps outdoor sequences.

        Returns ``(time_bag_s, enu)`` numpy arrays on the rosbag clock (the
        AHRS-yaw offset already applied), or ``None`` when no trusted GPS
        reference exists (indoor 'file' mode, or an untrusted outdoor offset
        such as the CMU-garage GPS outage). This is NOT part of the per-window
        GT container — offline evaluation consumes it directly so the exact
        same clock alignment used by the loader is used for scoring.
        """
        if self._gps_time_bag is None or self._gps_enu is None:
            return None
        return self._gps_time_bag.numpy(), self._gps_enu.numpy()

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

    # Not on ISequence, but ``Sequence.ZeroVelocity`` and
    # ``Sequence.ExternalOrientation`` forward ``clip`` to whatever they wrap, so
    # a loader that may sit under either needs it.
    def clip(self, start: int | None, end: int | None, step: int | None) -> None:
        sl = slice(start, end, step)
        for attr in self._TENSOR_ATTRS:
            setattr(self, attr, getattr(self, attr)[sl])
        if self._corrected_acc is not None:
            self._corrected_acc  = self._corrected_acc [sl]
            self._corrected_gyro = self._corrected_gyro[sl]

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        def _move(t: torch.Tensor) -> torch.Tensor:
            return t if device is None else t.to(device=device)

        for attr in self._TENSOR_ATTRS:
            setattr(self, attr, _move(getattr(self, attr)))
        if self._corrected_acc is not None:
            self._corrected_acc  = _move(self._corrected_acc)
            self._corrected_gyro = _move(self._corrected_gyro)
