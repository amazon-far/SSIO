"""Train smoke test — integration self-supervision runs in-process and writes a loadable checkpoint.

The run itself happens once, in the ``trained_run`` session fixture; the tests here
assert what it left behind. The CLI equivalent is verified to work, so a failure in
this file is a real defect in ``train_ssio.py``, its Hydra config, or the
checkpoint contract between training and ``evaluate.py``:

    EXP_DIR=<tmp> PYTHONPATH=. python train_ssio.py \\
        'hydra.searchpath=[file://tests/fixtures/config]' \\
        dataset@train_dataset=EuRoC_Smoke_Train dataset@eval_dataset=EuRoC_Smoke_Valid \\
        train.epoch=1 train.batch=4 train.max_train_steps=2 train.dv_imu_input=raw \\
        wandb.mode=disabled

Nothing here asserts a loss or metric value: two steps of a randomly initialised
network carry no signal, and pinning such a number would only encode the current
seed. What is asserted is wiring — that the run reaches its own checkpoint-saving
code, and that what it saves is what the evaluator can load.
"""
import os

import pytest
import torch
from omegaconf import OmegaConf

from .conftest import TRAIN_STEPS

pytestmark = pytest.mark.closedloop


def test_run_writes_exactly_one_checkpoint_named_for_its_step(trained_run):
    """``checkpoint_{step:05d}.pt`` appears, once, for the step the cap stopped on.

    ``save_checkpoint_artifact`` derives the filename from the global step counter, and
    ``evaluate.pick_checkpoint`` parses that number back out — the name is the only
    contract between the two scripts, so it is asserted literally.
    """
    checkpoints = sorted(
        f for f in os.listdir(trained_run.run_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    )
    assert checkpoints == [f"checkpoint_{TRAIN_STEPS:05d}.pt"], (
        f"expected one checkpoint for step {TRAIN_STEPS}; found {checkpoints}"
    )


def test_step_cap_stopped_the_run_mid_epoch(trained_run):
    """The 10-window epoch really does exceed ``TRAIN_STEPS`` batches.

    Guards the premise of the mid-epoch stop: with ``train.batch=4`` over
    ``epoch_windows`` windows the epoch is more than ``TRAIN_STEPS`` steps, so the
    run must have taken the ``stop = True`` branch inside the batch loop.
    """
    steps_per_epoch = -(-trained_run.epoch_windows // int(trained_run.cfg.train.batch))
    assert steps_per_epoch > TRAIN_STEPS, (
        f"epoch is only {steps_per_epoch} step(s) at batch {trained_run.cfg.train.batch}; "
        "the max_train_steps cap no longer interrupts an epoch, so the mid-epoch "
        "stop-and-checkpoint path is untested."
    )


def test_checkpoint_loads_strictly_into_a_freshly_built_network(trained_run):
    """The saved state_dict matches ``cfg.network`` key-for-key.

    This is the train→eval contract: ``evaluate.main`` does a strict
    ``load_state_dict`` on a network built from the stored ``cfg.network``. EMA is on
    by default, so the file must hold ``ema_model.module``'s keys — unprefixed — and
    not ``AveragedModel``'s ``module.*`` / ``n_averaged``.
    """
    import evaluate as ev
    from Interface     import instantiate
    from Model.Network import INetwork

    cfg = OmegaConf.load(trained_run.cfg_path)
    state_dict = torch.load(ev.pick_checkpoint(trained_run.run_dir, None),
                            map_location=cfg.meta.device)

    assert not any(k.startswith("module.") or k == "n_averaged" for k in state_dict), (
        "checkpoint carries AveragedModel wrapper keys; evaluate.py's strict load "
        f"would reject it. keys: {sorted(state_dict)[:5]}"
    )
    network = instantiate(INetwork, cfg.network).to(cfg.meta.device, cfg.meta.dtype)
    network.load_state_dict(state_dict)  # strict by default — raises on any mismatch
    assert all(torch.isfinite(p).all() for p in network.parameters()), (
        "some parameters are NaN/Inf after two training steps"
    )


def test_stored_config_is_the_smoke_recipe_and_an_evaluate_target(trained_run):
    """``.hydra/config.yaml`` is present and describes the run we think we made.

    ``evaluate.py`` reconstructs everything (network, dtype, device, gravity) from this
    file alone, so a run dir without it is unscoreable. The dataset assertions pin the
    intent of the two smoke groups: a ``RandomWindow`` train pool and a
    ``SlidingWindow`` eval pool over the committed fixture, debias off.
    """
    assert os.path.isfile(trained_run.cfg_path), (
        f"{trained_run.cfg_path} missing — evaluate.py cannot score this run dir"
    )
    cfg = OmegaConf.load(trained_run.cfg_path)

    assert "hydra" not in cfg, "hydra node leaked into the dumped config"
    assert int(cfg.train.max_train_steps) == TRAIN_STEPS
    assert cfg.wandb.mode == "disabled"
    assert str(cfg.meta.device) == "cpu", (
        "the smoke run must stay on CPU so it is host-independent and repeatable"
    )

    train_node = cfg.train_dataset.sequences[0]
    eval_node  = cfg.eval_dataset.sequences[0]
    assert train_node._target_.endswith("RandomWindow.RandomWindow")
    assert eval_node ._target_.endswith("SlidingWindow.SlidingWindow")
    for node in (train_node, eval_node):
        assert node.seq._target_.endswith("EuRoC.EuRoC_Sequence")
        # Off, not merely declining: `corrected_*` stays None, so the run has to name
        # its Δv stream (`train.dv_imu_input=raw`).
        assert node.seq.config.debias is False, (
            "smoke fixture requires debias: false; StaticSpanDebias has no static "
            "span to calibrate on in the first 10 s of MH_02_easy"
        )
