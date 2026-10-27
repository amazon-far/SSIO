"""Populate ``input_rot`` with per-window gyro-integrated orientation.

Wrapper mixin that sits above the specific window class (``SlidingWindow`` or
``RandomWindow``) with their ``(B, S, K, ...)`` batched windows.
For each window the body->world rotation is cumprod integrated forward from the
GT rot at the beginning of the window, anchoring it to mimic a low-drift attitude
filter (e.g. AHRS, EKF, sliding-window PGO) that would be used during deployment.
The result is written to ``input_rot`` (the network-input attitude override read
by ``get_est_velocity``); ``gt_poses`` is left UNTOUCHED, so the GT used for
anchors/metrics and the pseudo-label frame is preserved.
 -  Orientation reads only the main-body IMU (K index 0), so multi-IMU stacks (K>1)
    do not need special handling.
"""

import dataclasses

import torch
import pypose as pp
from omegaconf import DictConfig

from Interface.sequence import ISequence
from Interface.data     import IMUData_w_GT


class WindowIntegratedOrientation(ISequence[IMUData_w_GT]):
    CONFIG_SPEC = {}

    def __init__(self, config: DictConfig, seq: ISequence[IMUData_w_GT]):
        self.check_config(config)
        super().__init__(config)
        self.internal_seq = seq

    @property
    def seq_name(self) -> str:
        return self.internal_seq.seq_name

    @torch.no_grad()
    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        sample = self.internal_seq[index] # (B, S, K, ...) windowed batch

        if sample.corrected_gyro is None:
            # `corrected_gyro` is None whenever no correction stage ran. Falling back
            # to `sample.gyro` is NOT acceptable here: this wrapper's whole output is
            # an integrated attitude, so silently integrating the as-measured gyro
            # would change what `input_rot` means (bias-driven drift) with no signal
            # anywhere. Fail instead, naming the fix.
            raise ValueError(
                f"{type(self).__name__} integrates the CORRECTED gyro, but "
                f"{self.internal_seq.seq_name!r} serves corrected_gyro=None (no "
                "correction stage ran). Set `debias: true` on the loader config."
            )
        gyro = sample.corrected_gyro[:, :, 0] # (B, S, 3) main-body
        _, S, _ = gyro.shape
        if S < 2:
            # S<2: no interval to integrate, pass GT rotation through unchanged
            return sample                                          

        R0 = sample.gt_poses.rotation()[:, 0:1] # (B, 1) SO3 per-window GT anchor
        time_ns = sample.time_ns[:, :, 0, :] # (B, S, 1) per-window main-body clock (ns)
        dt = (time_ns[:, 1:] - time_ns[:, :-1]).to(gyro.dtype) * 1.0e-9 # (B, S-1, 1) seconds

        dR = pp.so3(gyro[:, :-1] * dt).Exp() # (B, S-1) per-step body rotations
        # Seed R0 as the first element of the product so the cumprod integrates
        # *from* the GT anchor and comes out as absolute orientation directly.
        seeded  = pp.SO3(torch.cat([R0.tensor(), dR.tensor()], dim=1))  # (B, S): [R0, dR0, ...]
        est_rot = pp.cumprod(seeded, dim=1, left=False)                 # (B, S) absolute; frame 0 == R0

        # Non-destructive: gt_poses is preserved; the integrated attitude becomes
        # the network-input override read by get_est_velocity.
        return dataclasses.replace(sample, input_rot=est_rot)

    def __len__(self) -> int:
        return len(self.internal_seq)
