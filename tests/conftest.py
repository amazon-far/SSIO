"""pytest path setup and the smoke-fixture accessor.

Prepends the repo root to ``sys.path`` so ``Utils``, ``Sequence``, ``Model`` …
resolve regardless of pytest's rootdir.

It also owns the single accessor for the committed 10 s EuRoC smoke fixture: the
session-scoped ``smoke`` fixture below. Everything about that fixture — its path,
its frame count, the dataset group names, and how to instantiate them — is reached
through ``smoke``, so every suite that touches the fixture agrees on it and none has
to import anything from this file.
"""
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(scope="session")
def smoke() -> types.SimpleNamespace:
    """Everything the 10 s EuRoC smoke fixture offers, as one namespace.

    Attributes:
        ``root``          ``data_root`` for the loader (it appends ``seq_name``).
        ``seq_name``      ``"MH_02_easy"``.
        ``frames``        2000 — the frame count the loader yields. Every
                          ``window_size`` claim in the smoke dataset YAMLs is
                          arithmetic on this; ``tests/closedloop`` asserts it
                          (``test_data_contract.py``'s window/anchor tests).
        ``config_root``   directory to put on Hydra's searchpath so the two smoke
                          dataset groups resolve. They are test-only, so they live
                          under ``fixtures/config/dataset/`` rather than polluting
                          ``Config/dataset/`` with a group no experiment uses.
        ``train_dataset`` / ``valid_dataset``  the smoke Hydra dataset group names.
        ``load(name, window=1024, dtype="fp32") -> list[ISequence]``
                          Instantiate a shipped dataset YAML through the same
                          ``Interface.instantiate`` path Hydra uses at train time,
                          with ``data_root`` repointed at the fixture's absolute
                          path.
    """
    root = os.path.join(_HERE, "fixtures", "euroc_mh02_10s")
    if not os.path.isdir(root):
        pytest.fail(
            f"smoke fixture missing: {root}\n"
            "It is committed; regenerate with Scripts/make_smoke_fixture.py if it was deleted."
        )
    config_root = os.path.join(_HERE, "fixtures", "config")
    config_dir  = os.path.join(config_root, "dataset")

    def load(dataset_name: str, window: int = 1024, dtype: str = "fp32") -> list:
        # Imported lazily: an unconditional `import Interface` here would pull
        # torch + pypose into every session that only runs the flat unit tests.
        from omegaconf import OmegaConf

        import Interface  # noqa: F401  — registers host_env / as_device / as_dtype
        from Interface import instantiate
        from Interface.sequence import ISequence

        path = os.path.join(config_dir, f"{dataset_name}.yaml")
        if not os.path.isfile(path):
            pytest.fail(f"dataset config not found: {path}")
        # allow_objects + no OmegaConf.resolve(): ${meta.device} resolves to a
        # torch.device, which is not an OmegaConf primitive, so eager resolution
        # raises. Hydra resolves lazily inside instantiate, same as at train time.
        cfg = OmegaConf.create(
            {"train": {"window": window, "gravity": 9.81007},
             "meta": {"device": "${as_device:'cpu'}", "dtype": f"${{as_dtype:'{dtype}'}}"}},
            flags={"allow_objects": True},
        )
        cfg.dataset = OmegaConf.load(path)
        for node in cfg.dataset.sequences:
            node.seq.config.data_root = root
        return [instantiate(ISequence, node) for node in cfg.dataset.sequences]

    return types.SimpleNamespace(
        root=root,
        config_root=config_root,
        seq_name="MH_02_easy",
        frames=2000,
        train_dataset="EuRoC_Smoke_Train",
        valid_dataset="EuRoC_Smoke_Valid",
        load=load,
    )
