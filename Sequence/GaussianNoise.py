import dataclasses
import typing as tp

import torch
from omegaconf import DictConfig

from Interface.sequence import ISequence
from Interface.data     import IMUData


DataType = tp.TypeVar("DataType", bound=IMUData)


class GaussianNoise(ISequence[DataType]):
    """Additive Gaussian white-noise augmentation on the raw IMU streams.

    A higher-order sequence (like ``SlidingWindow`` / ``RandomWindow``): it
    wraps an inner ``ISequence`` and, on every ``__getitem__``, returns the
    inner sample with fresh zero-mean Gaussian noise added to ``acc`` and
    ``gyro``. The accelerometer and gyroscope noise standard deviations are
    controlled independently from the Hydra config (``acc_noise_std`` /
    ``gyro_noise_std``), each expressed in the same physical unit as its channel
    (m/s^2 for ``acc``, rad/s for ``gyro``).

    Noise is resampled on each access (repeated reads of the same index differ,
    matching ``RandomWindow``'s use of the global RNG) and is *not* folded into
    the ``acc_cov`` / ``gyro_cov`` fields, which keep describing the underlying
    sensor model. Raw and corrected both shifted by the same realization, as in
    ``Sequence.Bias``. Every other field — including ``is_first_window`` — passes
    through untouched.

    Place it anywhere in the wrapper chain; it is shape-agnostic, so both
    ``GaussianNoise(SlidingWindow(...))`` and ``SlidingWindow(GaussianNoise(...))``
    are valid.
    """

    def __init__(self, config: DictConfig, seq: ISequence[DataType]):
        super().__init__(config)
        self.internal_seq = seq

    @property
    def seq_name(self) -> str:
        return self.internal_seq.seq_name

    def __getitem__(self, index: slice | int) -> DataType:
        sample: DataType = self.internal_seq[index]

        acc_std:  float = float(self.config.acc_noise_std)
        gyro_std: float = float(self.config.gyro_noise_std)

        # ``randn_like`` inherits each field's device and dtype, so the noise
        # lands on the same accelerator / precision as the underlying data.
        acc_noise  = torch.randn_like(sample.acc)  * acc_std
        gyro_noise = torch.randn_like(sample.gyro) * gyro_std
        return dataclasses.replace(
            sample,
            acc            =sample.acc            + acc_noise,
            gyro           =sample.gyro           + gyro_noise,
            corrected_acc  =None if sample.corrected_acc  is None else sample.corrected_acc  + acc_noise,
            corrected_gyro =None if sample.corrected_gyro is None else sample.corrected_gyro + gyro_noise,
        )

    def __len__(self) -> int:
        return len(self.internal_seq)
