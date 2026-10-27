# Bias Calibration

With `debias: true`, a dataset loader estimates one constant accelerometer and gyroscope
bias per sequence from its static frames and serves the corrected stream as
`corrected_acc` / `corrected_gyro`. The raw `acc` / `gyro` are never modified. The
estimators live in `Sequence/debias.py`.

## Where the corrected stream is used

| Consumer | Stream | Selected by |
|---|---|---|
| Δv teacher (`get_ref_delta_v`) | `corrected_*` | `train.dv_imu_input` (default `corrected`) |
| Network input | `acc` / `gyro` | `network.imu_input` (`raw` in every shipped network config) |
| `evaluate.py` preintegration overlay | same as the teacher | the run's `train.dv_imu_input` |

So calibration changes the self-supervised Δv target, not what the network sees.

`corrected_* is None` only when `debias: false`. When an estimator finds no usable static
frames it declines, returns the raw stream unchanged, and records a `skipped_*` status in
the sequence's `debias_report`; the loader prints that status at load time.

## Estimators

Select one with the `calibrater` node of a dataset config; `debias: true` without one is
an error. The keys other than `kind` are passed to the estimator's constructor: every
estimator accepts `debias_gyro: false` (estimate the accelerometer bias only), and `imu`
also accepts the detector thresholds (`a_std_max`, `gyro_mean_max`, `gyro_std_max`,
`grav_dev_max`, `win_s`).

| `calibrater` | Class | Static frames | Reads GT |
|---|---|---|---|
| `{kind: span}` | `StaticSpanDebias` | Every frame between the first and last frame with GT speed < 0.001 m/s | yes |
| `{kind: set}` | `StaticSetDebias` | Every frame covered by a 90-frame run with GT speed < 0.02 m/s and \|gyro\| < 0.05 rad/s throughout | yes |
| `{kind: imu}` | `StaticIMUDebias` | Frames that `Interface.stationary.detect_stationary` marks static: over a 1 s centred window, std ‖a‖ < 0.20 m/s², mean ‖ω‖ < 0.10 rad/s, std ‖ω‖ < 0.05 rad/s and \|mean ‖a‖ − g\| < 0.40 m/s². The same detector as the `train.regularization_gate=imu` mask; the window length in samples comes from the timestamps | no (gravity removal still uses the attitude) |

```yaml
debias     : true
calibrater :
  kind     : imu
```

The bias is

```python
g_in_body = R_wb[idx].Inv() @ [0, 0, gravity]
acc_bias  = (acc[idx] - g_in_body).mean(0)
gyro_bias = gyro[idx].mean(0)
```

Set the loader's `gravity` to `${train.gravity}` so that calibration and preintegration
subtract the same value.

## Choosing an estimator

- **`span`** works when a sequence's static frames form one block, such as a rest
  period at the start. If the sequence stops again later, the span between the first
  and last static frame includes the motion in between, which is then absorbed into the
  bias.
- **`set`** uses only genuinely static runs.
- **`imu`** selects static frames from IMU statistics alone, for a calibration that uses
  no reference velocity.

Any estimate absorbs motion or attitude error over the selected frames as bias. The
shipped configs all use `imu` (EuRoC, Blackbird, Humanoid; GrandTour with
`gyro_mean_max: 0.005`; MIPO with `debias_gyro: false`, main IMU only).
