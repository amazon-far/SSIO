import dataclasses
import typing as tp

import torch
from omegaconf import DictConfig

from Interface.sequence import ISequence
from Interface.data     import IMUData


DataType = tp.TypeVar("DataType", bound=IMUData)


class Bias(ISequence[DataType]):
    def __init__(self, config: DictConfig, seq: ISequence[DataType]):
        super().__init__(config)
        self.internal_seq = seq
        acc_bias_scale : float = float(self.config.acc_bias )
        gyro_bias_scale: float = float(self.config.gyro_bias)
        self.acc_bias  = torch.randn(1, 3, dtype=torch.float)
        self.gyro_bias = torch.randn(1, 3, dtype=torch.float)
        
        self.acc_bias  *= acc_bias_scale / self.acc_bias.norm()
        self.gyro_bias *= gyro_bias_scale / self.gyro_bias.norm()

    @property
    def seq_name(self) -> str:
        return self.internal_seq.seq_name

    def __getitem__(self, index: slice | int) -> DataType:
        sample: DataType = self.internal_seq[index]

        # As-measured and corrected both shifted: a sensor offset sits underneath the
        # correction chain, which cannot be re-run after the fact.
        acc_bias  = self.acc_bias .to(sample.acc)
        gyro_bias = self.gyro_bias.to(sample.gyro)
        return dataclasses.replace(
            sample,
            acc            =sample.acc            + acc_bias,
            gyro           =sample.gyro           + gyro_bias,
            corrected_acc  =None if sample.corrected_acc  is None else sample.corrected_acc  + acc_bias,
            corrected_gyro =None if sample.corrected_gyro is None else sample.corrected_gyro + gyro_bias,
        )

    def __len__(self) -> int:
        return len(self.internal_seq)
