# Troubleshooting

## Environment

* **PyTorch 2.14 or newer** breaks the PyPose cumulative operators used by IMU
  preintegration (an empty `torch.arange` now raises). Use PyTorch < 2.14; tested with
  PyTorch 2.5.1 and PyPose 0.6.8.
* **No GPU.** Everything except `graph=sparse_imupgo` runs on CPU. Use
  `graph=dense_imupgo` for pose graph distillation (short trajectories only), and pass
  `'meta.device=${as_device:cpu}'`.
* **Windows.** Run the evaluator as `python -X utf8 evaluate.py ...`. Without UTF-8 mode
  it fails with a `UnicodeEncodeError` when printing metric names.

## Data

* **`DATA_ROOT` is not set.** There is no implicit data path; every dataset config
  resolves under `$DATA_ROOT`, and a missing variable raises an error.
* **`corrected_acc` is `None`.** The config sets `debias: false`, or the loader found no
  static frames, so there is no bias-corrected stream. Pass `train.dv_imu_input=raw`.

## Tests

```bash
python -m pytest -m "not closedloop"   # unit tests
python -m pytest tests/closedloop      # closed-loop training and evaluation on the bundled sample
```

Tests that require CUDA are skipped on CPU-only machines. If ROS is sourced, set
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to prevent its pytest plugin from breaking test
collection.
