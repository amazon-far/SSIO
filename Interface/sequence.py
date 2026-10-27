import typing as tp

from omegaconf import DictConfig
from torch.utils.data import Dataset

from .base import Base


DataType = tp.TypeVar("DataType")


class ISequence(Base, Dataset[DataType]):
    def __init__(self, config: DictConfig):
        self.config = config

    @property
    def seq_name(self) -> str:
        """Human-readable per-sequence identifier (the dataset's ``seq_name`` key).

        Loaders return their configured name; higher-order wrappers forward to
        their ``internal_seq`` so the name survives any wrapper nesting.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement seq_name; loaders must "
            "return their configured name, wrappers must forward to internal_seq."
        )

    def __getitem__(self, index: slice | int) -> DataType: ...

    def __len__(self) -> int: ...
