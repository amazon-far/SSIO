"""Inference consistency — scoring the same checkpoint twice must give the same numbers.

This is the "performance consistency of model inference" half of the smoke suite. It is
worth its own file because an eval path that is only *approximately* repeatable makes
every A/B comparison in the experiment log unfalsifiable: a 2% difference between two
checkpoints means nothing if re-scoring one of them moves it by 2%.

The eval path is expected to be *bit* identical, not merely close, and the tests assert
that. Three things make it so and each is covered below:

  * ``SlidingWindow`` is deterministic — no RNG, unlike the ``RandomWindow`` used for
    training, whose window starts come from the global torch RNG.
  * ``run_sequence`` is under ``@torch.no_grad`` with ``shuffle=False`` and the network
    in ``eval()``; no dropout/batchnorm state evolves.
  * ``compute_metrics`` is a pure function of the returned tensors.

If a future change introduces sampling, augmentation, or non-deterministic reduction
into the eval path, the exact-equality assertions here fail immediately rather than
quietly widening the noise floor of every reported result.
"""
import math

import pytest
import torch

pytestmark = pytest.mark.closedloop


def _identical(left: dict[str, float], right: dict[str, float]) -> list[str]:
    """Keys whose values differ, treating NaN == NaN.

    Plain ``==`` on the dicts is useless here: ``dv_err_10s`` is NaN by construction and
    ``nan != nan``, so an unaware comparison reports a false difference every time.
    """
    assert set(left) == set(right), (
        f"key sets differ: {sorted(set(left) ^ set(right))}"
    )
    return [
        k for k in left
        if not (left[k] == right[k] or (math.isnan(left[k]) and math.isnan(right[k])))
    ]


def test_scoring_the_same_checkpoint_twice_is_bit_identical(evaluator):
    first  = evaluator.score()
    second = evaluator.score()
    differing = _identical(first, second)
    assert not differing, (
        "eval is not deterministic; these metrics changed between two identical runs: "
        + ", ".join(f"{k}: {first[k]!r} -> {second[k]!r}" for k in differing)
    )
    # Sanity: the comparison above is not vacuous — the run really did produce numbers.
    assert math.isfinite(first["AVE"]) and first["AVE"] > 0.0


def test_a_fresh_sequence_instance_yields_the_same_tensors(evaluator):
    """Repeatability is not an artifact of reusing one dataset object.

    ``evaluator.score()`` reuses a single ``ISequence``; a cached tensor inside it would
    make the previous test pass even if loading were non-deterministic. Rebuilding the
    sequence from the same config re-reads the fixture CSVs and must land on the same
    bits — which also pins the loader (debias off, sigma constants) as deterministic.
    """
    reference = evaluator.run()
    rebuilt   = evaluator.run(seq=evaluator.build_sequence())

    names = ("pred_world", "gt_world", "gt_rot_log", "t_ns",
             "preint_vel", "preint_pos", "preint_rot_log")
    for name, a, b in zip(names, reference, rebuilt):
        assert torch.equal(a, b), f"run_sequence().{name} differs on a rebuilt sequence"


def test_a_reloaded_checkpoint_scores_identically(evaluator):
    """Load → score is reproducible across network instances, not just across calls.

    Covers the other half of the consistency claim: ``evaluate.py`` builds the network
    fresh on every invocation, so "the same checkpoint scores the same" only holds if
    ``load_state_dict`` restores the model exactly and no buffer is left at its
    construction-time value.
    """
    reloaded = evaluator.build_network()
    for name, a, b in zip(("pred_world", "gt_world"),
                          evaluator.run(), evaluator.run(net=reloaded)):
        assert torch.equal(a, b), f"{name} differs after reloading the checkpoint"

    differing = _identical(evaluator.score(), evaluator.score(net=reloaded))
    assert not differing, (
        "a second network loaded from the same checkpoint scores differently: "
        f"{differing}"
    )


# The preintegrator's repeatability is covered above:
# test_a_fresh_sequence_instance_yields_the_same_tensors compares all seven
# ``run_sequence`` outputs, ``preint_*`` included.
