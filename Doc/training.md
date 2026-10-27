# Training Without a Recipe

The paper recipes in `Config/experiment/` (see the README) set every option below. This
page is for choosing configuration groups by hand: new datasets, ablations, or changes
to the method.

Every run writes a Hydra output directory under `$EXP_DIR` with the resolved
configuration (`.hydra/config.yaml`), periodic checkpoints (`checkpoint_<step>.pt`), and
the best checkpoints by validation loss (`best_model.pt`) and validation AVE
(`best_ave_model.pt`). Any value can be overridden from the command line, e.g.
`train.batch=256 train.lr=5e-4 wandb.mode=disabled`.

## Integration Self-Supervision (`train_ssio.py`)

```bash
python train_ssio.py network=airio_imu_only loss=ssio_dv_1_3_9_27 \
    train.regularization_gate=imu train.regularization_weight=<WEIGHT> \
    dataset@train_dataset=EuRoC_Train_Randomized dataset@eval_dataset=EuRoC_Valid
```

* **Supervision horizons.** `loss=ssio_dv_*` sums velocity-change residuals over a
  geometric series of window lengths, measured in network steps of 9 IMU samples. For
  example, `ssio_dv_1_3_9_27` supervises gaps of 1, 3, 9, and 27 steps. Choose the
  longest gap per dataset so that the teacher's integration error stays small. TLIO uses
  `loss=ssio_variance`.
* **Stationary regularization.** `train.regularization_gate=imu` detects stationary intervals from
  the raw IMU with thresholds shared across datasets; tune `train.regularization_weight` per
  dataset. The default `train.regularization_weight=0` disables the term.
* **Teacher stream.** The default `train.dv_imu_input=corrected` integrates the
  bias-corrected IMU. TLIO has no corrected stream; use
  `train.dv_imu_input=raw` there (see [`debias.md`](debias.md)).

## Pose Graph Distillation (`train_pgd.py`)

```bash
python train_pgd.py network=airio_imu_only \
    dataset@train_dataset=EuRoC_Train_Randomized dataset@eval_dataset=EuRoC_Valid \
    train.network_factor_sigma=1.0 train.resume_from=file:<PATH_TO_SSIO_RUN>/best_model.pt
```

The network configuration must match the `train_ssio.py` checkpoint. For each training
trajectory, the network predicts velocity over stitched windows, one pose graph over the
whole trajectory chains those predictions with IMU preintegration factors, and the
posterior velocities are cached as targets for the training windows.

* **Label refresh.** `train.label_refresh_every_steps: null` (default) solves the graphs
  once, before training; a positive value re-solves them with the current network every
  that many steps.
* **Network factor weight.** `train.network_factor_sigma` (required) is the isotropic
  std (m/s) of the network's velocity factors in the graph. The covariance head is not
  trained by self-supervision, so the graph does not use it. Small values keep the labels
  close to the network's predictions; large values let the IMU chain dominate. The paper recipes use
  0.2 to 2.0.
* **Training windows.** `train.step_size` (default 252) must be a multiple of the network
  stride (9), so window anchors fall on the trajectory's anchor grid.

Two graph backends with the same factors are available:

* `graph=sparse_imupgo` (default): block-tridiagonal Hessian, solved with the cuDSS
  direct solver in fp64; linear in the number of keyframes. Linux and CUDA only.
* `graph=dense_imupgo`: dense Cholesky in fp64, cubic in the number of keyframes. Runs on
  CPU or GPU; use it only for short trajectories such as the quick-start sample.

## Proprioceptive Input

On legged platforms, a joint-aware network lets the student read joint positions,
velocities, and torques in addition to the IMU:

| Platform   | Network                          | Student inputs |
|------------|----------------------------------|----------------|
| MIPO       | `airio_joints_imu_only`          | 5 IMUs, 16 joints |
| GrandTour  | `airio_joints_imu_only_anymal`   | 1 IMU, 12 joints |
| Humanoid   | `airio_joints_imu_only_humanoid` | 1 IMU, 29 joints |

`modality.use_aux_imu` and `modality.use_joints` restrict a multi-sensor dataset to the
main IMU or drop joint states. For the IMU-only baseline on a legged dataset, combine
`network=airio_imu_only` with `modality.use_aux_imu=false modality.use_joints=false`.

## Use of Reference Data

The training losses use no velocity or position labels. The following inputs still come
from each dataset's reference trajectory, as in the paper:

* **Attitude.** To use gyroscope-integrated attitude instead, wrap the loader in
  `Sequence.WindowIntegratedOrientation`.
* **Bias calibration.** The shipped dataset configs select stationary frames from IMU
  statistics alone; removing gravity from the accelerometer still uses the reference
  attitude. See [`debias.md`](debias.md) for the setting per dataset.
* **Pose graph anchor.** Each trajectory's graph is anchored on the reference rotation
  and velocity at its first keyframe.
* **Model selection.** `best_ave_model.pt` is selected with reference velocity on the
  validation split; `best_model.pt` is selected with the self-supervised validation loss.

Some settings are supervised by design: the `airio*` recipes (`loss=supervised_only`)
and `train.regularization_gate=gt`.

## Code Structure

Datasets, networks, losses, optimizers, schedulers, and pose graph backends are wired
through [Hydra](https://hydra.cc) configuration groups against typed interfaces, so
replacing a component is a configuration change. Swappable groups:
`dataset@train_dataset`, `dataset@eval_dataset`, `network`, `loss`, `optimizer`,
`scheduler`, and `graph` (PGD only). Defaults are in `Config/train_ssio.yaml` and
`Config/train_pgd.yaml`; `+experiment=<dataset>/<recipe>` applies a paper recipe on top.

```
Config/        Hydra entry points and configuration groups
Doc/           Training, troubleshooting, bias calibration, and MIPO HDF5 notes
Interface/     Typed interfaces for data, sequences, models, losses, and graphs
Model/         AirIO velocity networks
Loss/          Integration self-supervision, distillation, and stationary losses
Optimizer/     Dense and sparse IMU pose graph optimizers
Sequence/      Dataset loaders and composable window / modifier wrappers
Scripts/       Bias fitting, GrandTour time offsets, and the test fixture
Utils/         Training loop, checkpointing, scheduling, and plotting
tests/         Unit and closed-loop tests
```

**Tensors.** Batched IMU fields have shape `(B, N, K, ...)`: batch, time, and IMU index.
Single-IMU datasets use `K = 1`.

**Frames.** The network predicts velocity in the IMU body frame. The world frame is z-up
with gravity along −z.

**IMU streams.** `acc` / `gyro` are raw measurements. `corrected_acc` / `corrected_gyro`
are the output of bias calibration and are `None` when no correction runs.
`network.imu_input` selects the network's input stream (default `raw`), and
`train.dv_imu_input` selects the teacher's stream (default `corrected`).
