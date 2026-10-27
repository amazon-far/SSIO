import torch

from omegaconf import DictConfig, ListConfig

from Interface.base import Opt, T_ConfigSpec


_is_bool = lambda value: type(value) is bool
_is_number = lambda value: (
    isinstance(value, (int, float)) and not isinstance(value, bool)
)
_is_nonnegative = lambda value: _is_number(value) and value >= 0
_is_positive = lambda value: _is_number(value) and value > 0
_is_positive_int = lambda value: type(value) is int and value > 0
_is_optional_nonnegative = lambda value: value is None or _is_nonnegative(value)
_is_optional_positive_int = lambda value: value is None or _is_positive_int(value)
_is_optional_string = lambda value: value is None or isinstance(value, str)
_is_target_config = lambda value: (
    isinstance(value, DictConfig)
    and "_target_" in value
    and isinstance(value._target_, str)
)
_is_sequences = lambda value: (
    isinstance(value, ListConfig)
    and len(value) > 0
    and all(_is_target_config(item) for item in value)
)
_is_dataset_config = lambda value: (
    isinstance(value, DictConfig)
    and "sequences" in value
    and _is_sequences(value.sequences)
)


_COMMON_ROOT_SPEC: dict[str, T_ConfigSpec] = {
    "modality": {
        "use_aux_imu": _is_bool,
        "use_joints": _is_bool,
    },
    "train_dataset": _is_dataset_config,
    "eval_dataset": _is_dataset_config,
    "network": _is_target_config,
    "optimizer": _is_target_config,
    "scheduler": _is_target_config,
    "loss": _is_target_config,
    "meta": {
        "device": lambda value: isinstance(value, torch.device),
        "dtype": lambda value: isinstance(value, torch.dtype),
    },
    "wandb": {
        "project": lambda value: isinstance(value, str) and bool(value),
        "mode": lambda value: (
            isinstance(value, str)
            and value in {"online", "offline", "disabled"}
        ),
        "name": _is_optional_string,
    },
}

_COMMON_TRAIN_SPEC: dict[str, T_ConfigSpec] = {
    "lr": _is_positive,
    "decay": _is_nonnegative,
    "batch": _is_positive_int,
    "epoch": _is_positive_int,
    "window": _is_positive_int,
    "ema": _is_bool,
    "gravity": _is_positive,
    "checkpoint_every_epochs": _is_positive_int,
    "eval_plot_every_epochs": _is_positive_int,
    "max_train_steps": _is_optional_positive_int,
    "dv_imu_input": lambda value: (
        isinstance(value, str)
        and value in {"corrected", "raw"}
    ),
    "resume_from": _is_optional_string,
}

SSIO_CONFIG_SPEC: T_ConfigSpec = {
    **_COMMON_ROOT_SPEC,
    "train": {
        **_COMMON_TRAIN_SPEC,
        "num_workers": lambda value: type(value) is int and value >= 0,
        "data_ratio": lambda value: _is_number(value) and 0 < value <= 1,
        "regularization_weight": _is_nonnegative,
        "regularization_speed_max": _is_positive,
        "regularization_gate": lambda value: (
            isinstance(value, str)
            and value in {"all", "imu", "gt"}
        ),
        "regularization_sched_enable": _is_bool,
        "regularization_sched_start": _is_optional_nonnegative,
        "regularization_sched_end": _is_optional_nonnegative,
    },
}

ADAPT_CONFIG_SPEC: T_ConfigSpec = {
    **_COMMON_ROOT_SPEC,
    "graph": _is_target_config,
    "train": {
        **_COMMON_TRAIN_SPEC,
        "step_size": _is_positive_int,
        "label_refresh_every_steps": _is_optional_positive_int,
        "label_batch_windows": _is_positive_int,
        "network_factor_sigma": _is_positive,
    },
}
