import typing as tp

import torch
from omegaconf   import DictConfig, OmegaConf
from hydra.utils import instantiate as __hydra_instantiate

from .base import validate_hydra_config


GuardType = tp.TypeVar("GuardType")


def instantiate(of: type[GuardType], config: DictConfig, *args, **kwargs) -> GuardType:
    instance = __hydra_instantiate(config, *args, **kwargs)
    runtime_cls = tp.get_origin(of) or of
    assert isinstance(instance, runtime_cls), (
        f"instantiate(...) expected {of!r} but got {type(instance)!r}"
    )

    return instance


# -----------------------------------------------------------------------------
# Environment resolution
# -----------------------------------------------------------------------------
# Paths are read from environment variables. Use in yaml as:
#     data_root: ${host_env:DATA_ROOT}/TLIO           # export DATA_ROOT=/path/to/datasets
#     exp_dir:   ${host_env:EXP_DIR,./experiments}    # inline default
# SSIO_ROOT (this repository's root) needs no export.
# -----------------------------------------------------------------------------

import os as _os

_FALLBACK: dict[str, str] = {
    "EXP_DIR": "./experiments",
    "SSIO_ROOT": _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")),
}

def host_env(key: str, default: str | None = None) -> str:
    if key in _os.environ:
        return _os.environ[key]
    if default is not None:
        return default
    if key in _FALLBACK:
        return _FALLBACK[key]
    raise KeyError(f"host_env: environment variable {key!r} is not set")


OmegaConf.register_new_resolver("host_env", host_env, replace=True)


# -----------------------------------------------------------------------------
# Device / dtype resolution
# -----------------------------------------------------------------------------
# Use in yaml as:
#     device: ${as_device:'cuda:0'}    # falls back to cpu when CUDA absent
#     dtype : ${as_dtype:'fp32'}
# -----------------------------------------------------------------------------

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
    "fp64": torch.float64, "float64": torch.float64, "double": torch.float64,
}


def as_device(spec: str) -> torch.device:
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def as_dtype(spec: str) -> torch.dtype:
    key = spec.lower()
    if key not in _DTYPE_ALIASES:
        raise KeyError(f"as_dtype: unknown dtype {spec!r}; known: {sorted(_DTYPE_ALIASES)}")
    return _DTYPE_ALIASES[key]


OmegaConf.register_new_resolver("as_device", as_device, replace=True)
OmegaConf.register_new_resolver("as_dtype",  as_dtype,  replace=True)


__all__ = [
    "instantiate",
    "validate_hydra_config",
    "host_env",
    "as_device",
    "as_dtype",
]
