"""The ``acc`` / ``corrected_acc`` split, and the network flag that selects between them.

Covers the plumbing directly, because the smoke fixture ships ``debias: false`` and
therefore only ever exercises the identity case in
``tests/closedloop/test_data_contract.py``. Here the applied case is constructed, so a
regression that discards the calibrater output fails a test instead of silently training
the network on the wrong stream.

See README.md "Conventions" for why the pair is named by content rather than by reader.
"""

import inspect

import pytest
import torch

from Sequence.debias import STATUS_DISABLED, disabled_report
from Model.airio     import select_imu


# ---------------------------------------------------------------- debias off


def test_disabled_report_is_published_and_not_applied():
    """``debias: false`` still has to publish a report.

    Consumers read ``debias_report`` unconditionally, so the ``else`` branch
    cannot simply leave the attribute unset.
    """
    r = disabled_report(17)
    assert r.status == STATUS_DISABLED
    assert not r.applied
    assert r.n_total == 17
    assert r.n_used == 0
    assert r.acc_bias == [] and r.gyro_bias == []


# ---------------------------------------------------------------- selector


class _IMU:
    def __init__(self):
        self.acc            = torch.zeros(2, 1, 3)
        self.gyro           = torch.zeros(2, 1, 3)
        self.corrected_acc  = torch.ones(2, 1, 3)
        self.corrected_gyro = torch.ones(2, 1, 3)


@pytest.mark.parametrize("stream, expected", [("raw", 0.0), ("corrected", 1.0)])
def test_select_imu_returns_the_named_pair(stream, expected):
    acc, gyro = select_imu(_IMU(), stream)
    assert torch.all(acc  == expected)
    assert torch.all(gyro == expected)


def test_the_stream_has_no_constructor_default():
    """Every network must be told its stream; there is no default to fall back on.

    A run directory saved by an older config has no ``imu_input`` key. With a default,
    ``hydra.utils.instantiate`` would build the network on whatever the default names
    and evaluate it on a stream it never trained on: right shapes, right dtypes,
    plausible AVE, wrong number. Without one it raises, and the run is re-evaluated with
    both streams passed as overrides — see README.md "Conventions" for what each stream
    holds.
    """
    from Model.airio import AirIONetwork, AirIONetworkIMUOnly

    for cls in (AirIONetwork, AirIONetworkIMUOnly):
        default = inspect.signature(cls.__init__).parameters["imu_input"].default
        assert default is inspect.Parameter.empty, (
            f"{cls.__name__}.imu_input defaults to {default!r}; a legacy config with no "
            "imu_input key would instantiate silently on that stream"
        )
