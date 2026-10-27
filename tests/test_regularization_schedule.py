import pytest

from Utils.lr_schedule import regularization_schedule_weight


def test_boundary_values():
    # E=30, start=0.2, end=0.8 => start_e=6, end_e=24.
    E = 30
    assert regularization_schedule_weight(5,  E, start=0.2, end=0.8) == 1.0
    assert regularization_schedule_weight(6,  E, start=0.2, end=0.8) == 1.0
    assert regularization_schedule_weight(15, E, start=0.2, end=0.8) == pytest.approx(0.5)
    assert regularization_schedule_weight(24, E, start=0.2, end=0.8) == 0.0
    assert regularization_schedule_weight(29, E, start=0.2, end=0.8) == 0.0


def test_fraction_absolute_equivalence():
    # fraction 0.2/0.8 of 30 == absolute 6/24.
    E = 30
    for e in range(E):
        frac = regularization_schedule_weight(e, E, start=0.2, end=0.8)
        absol = regularization_schedule_weight(e, E, start=6,   end=24)
        assert frac == pytest.approx(absol)


def test_start_equals_end_step_no_div0():
    # start==end => pure step, no ZeroDivision, no raise.
    E = 30
    assert regularization_schedule_weight(5,  E, start=0.5, end=0.5) == 1.0
    assert regularization_schedule_weight(15, E, start=0.5, end=0.5) == 0.0
    assert regularization_schedule_weight(16, E, start=0.5, end=0.5) == 0.0


def test_disabled_returns_one():
    E = 30
    assert regularization_schedule_weight(10, E, start=0.2, end=0.8, enable=False) == 1.0
    assert regularization_schedule_weight(10, E, start=None, end=None, enable=False) == 1.0


def test_enabled_requires_both_endpoints():
    with pytest.raises(ValueError):
        regularization_schedule_weight(10, 30, start=None, end=0.8)
    with pytest.raises(ValueError):
        regularization_schedule_weight(10, 30, start=0.2, end=None)


def test_start_after_end_raises():
    with pytest.raises(ValueError):
        regularization_schedule_weight(5, 30, start=0.8, end=0.2)


def test_negative_raises():
    with pytest.raises(ValueError):
        regularization_schedule_weight(5, 30, start=-1, end=10)
