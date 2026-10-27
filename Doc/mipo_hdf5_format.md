# MIPO Indoor — HDF5 Format

Indoor MIPO trials, one HDF5 per source ROS bag. Recorded on a Unitree Go1 with OptiTrack motion-capture ground truth. The original data is from the [Cerberus2.0 repo](https://github.com/ShuoYangRobotics/Cerberus2.0) and the "Multi-IMU Proprioceptive Odometry for Legged Robots" related paper, hence the adopted shorthand, MIPO. That data was saved as a rosbag and can be retrieved on that repo through the linked google drive. For ease of use, we converted this data to HDF5 (converter not included in this repository) and release it, with the calibration below, as [Open-IO/mipo-go1-odometry](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry). In the HDF5 files, each per-topic stream is stored at its native rate with its own ROS timestamp axis. Synchronization, resampling, filtering, and calibration are not applied.

## Directory layout

```
mipo_hdf5/indoor/
├── mipo_time_offsets.json
├── mipo_joint_stats.json
├── 20230615_risqh_gait_analysis/
│   ├── 20230615-risqh-flyingtrot-04-04-square.h5
│   └── ...
├── 20230620_risqh_robustness_test/
│   └── *.h5
├── 20230625_gait/
│   └── *.h5
└── 230622_robustness/
    └── *.h5
```
One `.h5` per ROS bag, with the bag basename preserved.

## Root attributes

| Attribute | Type | Meaning |
|---|---|---|
| `imu_topics` | JSON-encoded string list | The 5 IMU topic names included. |
| `joint_topic` | string | `/unitree_hardware/joint_foot` |
| `gt_topic` | string | `/natnet_ros/Shuo_Go1/pose` |
| `num_imus` | int | 5 (1 main-body + 4 aux near each foot). |
| `message_counts` | JSON-encoded dict | `{topic: count}` for every preserved topic. |
| `source_bag_rel` | string | Source bag path relative to the dataset root. |
| `compression` | string | `gzip`. |
| `created_utc` | string | ISO-8601 export timestamp. |

## Groups

Group paths mirror the ROS topic tree (leading `/` stripped). Every leaf is a contiguous numeric array (time-major along axis 0). Covariances are flattened row-major and are also the zero-placeholders that were present in the rosbags to comply with `sensor_msgs/Imu` format.

All `time` arrays are ROS message stamps in POSIX seconds (float64).

### `unitree_hardware/imu/` — body IMU, ~400 Hz

`sensor_msgs/Imu` schema.

| Dataset | Shape | Units |
|---|---|---|
| `linear_acceleration` | (N, 3) | m/s², sensor body frame, gravity included |
| `angular_velocity` | (N, 3) | rad/s, sensor body frame |
| `orientation` | (N, 4) | quaternion (x, y, z, w), Unitree onboard AHRS. Populated on most sequences (71/79); a handful of bags ship it as all-zero — check before using. |
| `linear_acceleration_cov` | (N, 9) | row-major 3×3 (all zero) |
| `angular_velocity_cov` | (N, 9) | row-major 3×3 (all zero) |
| `orientation_cov` | (N, 9) | row-major 3×3 (all zero) |
| `time` | (N,) | sec |

### `WT901_47_Data/`, `WT901_48_Data/`, `WT901_49_Data/`, `WT901_50_Data/` — leg IMUs, ~333 Hz

Four WitMotion WT901 IMUs mounted on the calves near the robot feet. Same field set as the body IMU. `*_cov` fields are all zero. `orientation` is also all zero on every sequence.

### `unitree_hardware/joint_foot/` — joints + foot contact, ~400 Hz

16 channels = 12 actuated joints + 4 foot-contact channels. These are taken directly in the format of the ROS bag, which are likely from the unitree SDK directly. The following are deduced from that:
 - The foot contact channels are integer force count values (still stored as floats).
 - These force values are saved in the `effort` fields of the last 4 channels of the joints topic.
 - Otherwise, these last four channels have zero position and velocity with the exception of the 5 `convexmpc` bags in `20230620_risqh_robustness_test/` which seem to have a 0/1 contact flags stored in the velocity field.

| Dataset | Shape | Notes |
|---|---|---|
| `names` | (16,) | Joint names, in storage order. |
| `position` | (N, 16) | rad (cols 12–15 zero) |
| `velocity` | (N, 16) | rad/s (cols 12–15 zero on most bags, 0/1 per-foot contact flags for `convexmpc` bags) |
| `effort` | (N, 16) | cols 0–11 = joint torques (N·m), cols 12–15 = raw foot-force counts (integer-valued, range ≈ 0–1500). |
| `time` | (N,) | sec |

Joint name → meaning:

| Idx | Name | Joint |
|---|---|---|
| 0 | FL0 | Front-left hip abduction/adduction |
| 1 | FL1 | Front-left thigh (hip pitch) |
| 2 | FL2 | Front-left calf (knee) |
| 3–5 | FR0/1/2 | Front-right hip-abad / thigh / calf |
| 6–8 | RL0/1/2 | Rear-left hip-abad / thigh / calf |
| 9–11 | RR0/1/2 | Rear-right hip-abad / thigh / calf |
| 12 | FL_foot | Front-left foot — force only, `effort[:, 12]` |
| 13 | FR_foot | Front-right foot — force only, `effort[:, 13]` |
| 14 | RL_foot | Rear-left foot — force only, `effort[:, 14]` |
| 15 | RR_foot | Rear-right foot — force only, `effort[:, 15]` |


### `natnet_ros/Shuo_Go1/pose/` — OptiTrack GT pose, ~60-70 Hz (varies)

| Dataset | Shape | Notes |
|---|---|---|
| `position` | (N, 3) | world-frame XYZ, meters |
| `orientation` | (N, 4) | quaternion (x, y, z, w), world ← body |
| `time` | (N,) | sec |

No velocity stream.

## `mipo_time_offsets.json`

Per-sequence kalibr results, keyed by bag basename. Each entry has `time_offset_s` (GT-clock → IMU-clock offset, seconds), `status` (`"optimized"` or `"prior"`), and `T_BS` (body←sensor quaternion & translation). The per-sequence `T_BS` values are the raw kalibr fits and not used. Only the `_consensus` entry's `T_BS` is used for the SE(3) calibration offset; the MIPO loader reads it from this file (`calibration_file`). A per-sequence `time_offset_s` is applied only when `status == "optimized"` and its magnitude is within the loader's trust bound; otherwise the `_consensus` entry's `time_offset_s` is applied. The MIPO dataloaders in this package already apply this calibration.

This file ships with the Hugging Face dataset.
