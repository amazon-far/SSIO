"""Eval closed loop — ``evaluate.py``'s own functions, on the checkpoint we just trained.

Every assertion here goes through the shipped module: ``apply_data_override``,
``pick_checkpoint``, ``run_sequence``, ``compute_metrics``. Nothing is re-implemented,
so the test cannot drift from the script the way a hand-rolled metric would; if
``evaluate.py`` renames a key or changes a horizon, this file fails.

What is asserted is the metric *contract* — which keys exist, which are finite, which
are NaN because the fixture cannot support them — never a value. The network is two
optimizer steps old; its AVE is noise, and pinning it would test the seed.

Structural NaN: the clip is 9.995 s and the default ``train.window=1024`` gives a
single 5.12 s eval window, so ``dv_err_10s`` cannot exist. That is a property of the
fixture, not a bug, and it is asserted together with the duration that causes it.
"""
import math

import pytest
from omegaconf import OmegaConf

from .conftest import dataset_yaml

pytestmark = pytest.mark.closedloop

#: Keys ``compute_metrics`` must return for the network, given ``_DV_WINDOWS_S`` =
#: (1, 2, 10) s and ``_RVTE_HORIZONS_S`` = (1, 2, 5) s. Spelled out rather than derived
#: from the module constants: these strings are the schema of ``result.json`` and of
#: every downstream report, so a silent rename should fail here.
EXPECTED_KEYS = {
    "AVE", "ATE", "ATE_mean",
    "dv_err_1s", "dv_err_2s", "dv_err_10s",
    "trend_mean_dt_sec",
    "RVTE_1s", "nRVTE_1s",
    "RVTE_2s", "nRVTE_2s",
    "RVTE_5s", "nRVTE_5s",
    "mean_nRVTE",
}
#: The one key that is NaN by construction on this fixture (see module docstring).
STRUCTURAL_NAN = {"dv_err_10s"}


def test_metric_dict_has_exactly_the_documented_keys(evaluator):
    metrics = evaluator.score()
    assert set(metrics) == EXPECTED_KEYS, (
        f"unexpected: {sorted(set(metrics) - EXPECTED_KEYS)}, "
        f"missing: {sorted(EXPECTED_KEYS - set(metrics))}"
    )
    assert all(isinstance(v, float) for v in metrics.values()), (
        "compute_metrics must return plain floats (result.json is json.dump'd): "
        f"{ {k: type(v).__name__ for k, v in metrics.items() if not isinstance(v, float)} }"
    )


def test_finite_everywhere_except_the_structurally_impossible_horizon(evaluator):
    metrics = evaluator.score()
    not_finite = {k for k, v in metrics.items() if not math.isfinite(v)}
    assert not_finite == STRUCTURAL_NAN, (
        f"non-finite metrics {sorted(not_finite)} != expected {sorted(STRUCTURAL_NAN)}"
    )
    # Error magnitudes are non-negative by definition; a negative one means a sign or
    # norm bug, which no amount of training would fix.
    for key in ("AVE", "ATE", "ATE_mean", "dv_err_1s", "dv_err_2s",
                "RVTE_1s", "RVTE_2s", "RVTE_5s", "nRVTE_1s", "nRVTE_2s", "nRVTE_5s",
                "mean_nRVTE"):
        assert metrics[key] >= 0.0, f"{key} is negative: {metrics[key]}"


def test_dv_err_10s_is_nan_because_the_eval_window_is_shorter_than_10_s(evaluator):
    """Tie the NaN to its cause, so a fixture change that removes it also fails here.

    ``SlidingWindow`` with ``step == window == 1024`` places exactly one window in the
    2000-frame clip, spanning ~5 s of anchors. ``compute_metrics`` reports NaN rather
    than aborting when a ΔV horizon exceeds the duration.
    """
    pred_w, _gt_w, _gt_rot_w, t_ns_w, *_ = evaluator.run()
    windows, anchors, _ = pred_w.shape
    assert windows == 1, (
        f"expected a single eval window at window=1024 over 2000 frames, got {windows}"
    )
    assert anchors >= 2, "compute_metrics needs at least 2 anchors"

    t_ns = t_ns_w.reshape(-1)
    duration_s = float(t_ns[-1] - t_ns[0]) * 1e-9
    assert 2.0 <= duration_s < 10.0, (
        f"eval span is {duration_s:.3f} s; the dv_err_10s NaN assertion below only "
        "holds while the span is under 10 s"
    )
    assert math.isnan(evaluator.score()["dv_err_10s"])


def test_pick_checkpoint_defaults_to_the_highest_step(trained_run, tmp_path):
    """No ``--checkpoint`` → the newest weights; explicit name / abs path honoured.

    A run dir normally holds many ``checkpoint_<step>.pt``; the smoke run writes one, so
    extra names are planted here to prove the choice is by step number and not by mtime
    or lexicographic order.
    """
    import evaluate as ev

    real = ev.pick_checkpoint(trained_run.run_dir, None)
    assert real.endswith(f"checkpoint_{trained_run.steps:05d}.pt")

    # Same file addressed three ways.
    assert ev.pick_checkpoint(trained_run.run_dir, f"checkpoint_{trained_run.steps:05d}.pt") == real
    assert ev.pick_checkpoint(trained_run.run_dir, real) == real

    with pytest.raises(FileNotFoundError):
        ev.pick_checkpoint(trained_run.run_dir, "checkpoint_99999.pt")
    with pytest.raises(FileNotFoundError):
        ev.pick_checkpoint(str(tmp_path), None)  # empty dir

    # Step ordering, on a dir of empty placeholder files: 9 must not beat 10.
    for step in (2, 9, 10):
        (tmp_path / f"checkpoint_{step:05d}.pt").touch()
    assert ev.pick_checkpoint(str(tmp_path), None).endswith("checkpoint_00010.pt")


def test_apply_data_override_swaps_the_eval_set_and_nothing_else(trained_run):
    """``--data`` replaces ``cfg.eval_dataset``; ``None`` is a no-op; a bad path raises.

    The override is what lets one checkpoint be scored on a different pool, so the test
    swaps in the *train* group and checks the sequences that come out really are the
    other window sampler — proving the swap reaches instantiation, not just the tree.
    """
    import evaluate as ev
    from Interface          import instantiate
    from Interface.data     import IMUData_w_GT
    from Interface.sequence import ISequence

    cfg = OmegaConf.load(trained_run.cfg_path)
    untouched = OmegaConf.to_yaml(cfg)
    ev.apply_data_override(cfg, None)
    assert OmegaConf.to_yaml(cfg) == untouched, "None override must not modify the config"

    with pytest.raises(FileNotFoundError):
        ev.apply_data_override(cfg, str(trained_run.run_dir) + "/no_such_dataset.yaml")

    ev.apply_data_override(cfg, dataset_yaml("EuRoC_Smoke_Train"))
    assert cfg.eval_dataset.sequences[0]._target_.endswith("RandomWindow.RandomWindow")
    # Interpolations inside the override (${train.window}, ${meta.device}) must resolve
    # against the STORED config — that is the whole point of assigning into the tree.
    swapped = instantiate(ISequence[IMUData_w_GT], cfg.eval_dataset.sequences[0])
    assert type(swapped).__name__ == "RandomWindow"
    assert swapped.seq_name == "MH_02_easy"
