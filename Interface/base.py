import typing as tp

from functools import wraps
from omegaconf import DictConfig


#: A predicate receives the raw node, ``DictConfig`` included: a sub-config whose
#: keys depend on a value inside it (an estimator's kwargs) has no dict spec, and
#: only its consumer can check it. Literal predicates still reject a ``DictConfig``
#: in their own ``isinstance``, so nothing is loosened for them.
T_LiteralSpec: tp.TypeAlias = tp.Callable[[tp.Any], bool]
T_ConfigSpec:  tp.TypeAlias = dict[str, "T_ConfigSpec"] | T_LiteralSpec | "Opt"


class Opt:
    """Marks a ``CONFIG_SPEC`` key as optional.

    The wrapped spec (a predicate or a nested dict) is validated only when the key
    is present. Unknown keys still fail.
    """
    __slots__ = ("spec",)

    def __init__(self, spec: T_ConfigSpec) -> None:
        self.spec = spec


def check_config(
    config: tp.Any,
    spec: T_ConfigSpec,
    path: str = "cfg",
) -> None:
    if isinstance(spec, Opt):
        raise TypeError(
            f"Opt(...) may only appear inside a config spec dict, got {path}"
        )

    if isinstance(spec, dict):
        if not isinstance(config, DictConfig):
            raise TypeError(
                f"{path} must be a config mapping, got {type(config).__name__}"
            )

        spec_keys = set(spec)
        required_keys = {k for k, v in spec.items() if not isinstance(v, Opt)}
        config_keys = {str(k) for k in config}

        missing = required_keys - config_keys
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")
        extra = config_keys - spec_keys
        if extra:
            raise KeyError(f"{path} has unexpected keys: {sorted(extra)}")

        for key, sub_spec in spec.items():
            if isinstance(sub_spec, Opt):
                if key not in config_keys:
                    continue
                sub_spec = sub_spec.spec
            check_config(config[key], sub_spec, f"{path}.{key}")
        return

    if not spec(config):
        raise ValueError(f"{path}={config!r} failed config validation")


def validate_hydra_config(
    spec: T_ConfigSpec,
) -> tp.Callable[
    [tp.Callable[[DictConfig], tp.Any]],
    tp.Callable[[DictConfig], tp.Any],
]:
    def decorate(
        func: tp.Callable[[DictConfig], tp.Any],
    ) -> tp.Callable[[DictConfig], tp.Any]:
        @wraps(func)
        def validated(cfg: DictConfig) -> tp.Any:
            check_config(cfg, spec)
            return func(cfg)

        return validated

    return decorate


class Base:
    CONFIG_SPEC: T_ConfigSpec

    @classmethod
    def check_config(
        cls,
        config: tp.Any,
        spec_override: T_ConfigSpec | None = None,
    ) -> None:
        spec: T_ConfigSpec = cls.CONFIG_SPEC if spec_override is None else spec_override
        check_config(config, spec)
