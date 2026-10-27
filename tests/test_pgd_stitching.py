"""``train_pgd.stitched_window_starts``: teacher windows tile every trajectory anchor."""
import types

import pytest

from train_pgd import stitched_window_starts


def _net(start_index=13, stride=9):
    # anchor_count mirrors the AirIO grid: anchors at start_index + k*stride inside the window.
    return types.SimpleNamespace(
        stride=stride, start_index=start_index,
        anchor_count=lambda n: (n - 1 - start_index) // stride + 1,
    )


@pytest.mark.parametrize("n_frames", [1024, 1025, 2000, 5000, 12345])
def test_windows_cover_every_anchor_once_on_the_grid(n_frames):
    net, window = _net(), 1024
    starts = stitched_window_starts(n_frames, window, net)
    m_win = net.anchor_count(window)
    covered = set()
    for s in starts:
        assert s % net.stride == 0 and s + window <= n_frames
        covered.update(range(s // net.stride, s // net.stride + m_win))
    m = starts[-1] // net.stride + m_win
    assert covered == set(range(m))
    # The covered grid reaches the last full window: no anchor beyond it fits a window.
    assert (n_frames - window) // net.stride * net.stride == starts[-1]


def test_short_trajectory_raises():
    with pytest.raises(ValueError):
        stitched_window_starts(1000, 1024, _net())
