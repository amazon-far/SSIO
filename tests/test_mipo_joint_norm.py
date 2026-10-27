"""MIPO joint standardization (``joint_norm``): the input contract of the released MIPO joint checkpoints."""
import json
import os

import pytest
import torch

from Interface.base import Opt
from Sequence.MIPO import (_CONTACT_JOINTS, _DEFAULT_JOINT_STATS, _EXPECTED_JOINTS, _LEG_JOINTS,
                           MIPO_Sequence, _standardize_joints)


def _raw_joints(n: int = 50):
    g = torch.Generator().manual_seed(0)
    j = len(_EXPECTED_JOINTS)
    pos = torch.randn(n, j, 1, generator=g)
    vel = torch.randn(n, j, 1, generator=g) * 2.0
    eff = torch.randn(n, j, 1, generator=g) * 5.0
    eff[:, _CONTACT_JOINTS] = eff[:, _CONTACT_JOINTS] * 50.0 + 233.0   # contact counts are O(100)
    return pos, vel, eff


def test_groups_cover_the_canonical_order():
    assert _LEG_JOINTS == slice(0, 12) and _CONTACT_JOINTS == slice(12, 16)
    assert all(name.endswith("_foot") for name in _EXPECTED_JOINTS[_CONTACT_JOINTS])
    assert not any(name.endswith("_foot") for name in _EXPECTED_JOINTS[_LEG_JOINTS])


def test_per_group_zscore():
    pos, vel, eff = _raw_joints()
    s = _DEFAULT_JOINT_STATS
    p, v, e = _standardize_joints(pos, vel, eff, s)

    def z(x, group):
        return (x - s[group]["mean"]) / s[group]["std"]

    torch.testing.assert_close(p[:, _LEG_JOINTS],     z(pos[:, _LEG_JOINTS], "leg_pos"))
    torch.testing.assert_close(v[:, _LEG_JOINTS],     z(vel[:, _LEG_JOINTS], "leg_vel"))
    torch.testing.assert_close(e[:, _LEG_JOINTS],     z(eff[:, _LEG_JOINTS], "leg_eff"))
    torch.testing.assert_close(e[:, _CONTACT_JOINTS], z(eff[:, _CONTACT_JOINTS], "contact_eff"))
    # Foot-contact channels carry signal only in effort: position / velocity pass through.
    torch.testing.assert_close(p[:, _CONTACT_JOINTS], pos[:, _CONTACT_JOINTS])
    torch.testing.assert_close(v[:, _CONTACT_JOINTS], vel[:, _CONTACT_JOINTS])


def test_inputs_not_mutated():
    pos, vel, eff = _raw_joints()
    before = [t.clone() for t in (pos, vel, eff)]
    _standardize_joints(pos, vel, eff, _DEFAULT_JOINT_STATS)
    for t, b in zip((pos, vel, eff), before):
        torch.testing.assert_close(t, b)


def test_config_spec_accepts_joint_keys():
    # Optional keys: existing dataset YAMLs (which omit both) keep validating.
    spec = MIPO_Sequence.CONFIG_SPEC
    assert isinstance(spec["joint_norm"], Opt) and isinstance(spec["joint_stats_file"], Opt)
    assert spec["joint_norm"].spec(True) and spec["joint_norm"].spec(False)
    assert not spec["joint_norm"].spec("yes")
    assert spec["joint_stats_file"].spec("/some/stats.json")
    assert not spec["joint_stats_file"].spec("")


def test_default_stats_match_dataset_file():
    """The built-in statistics equal the ``mipo_joint_stats.json`` the checkpoints were trained with."""
    root = os.environ.get("DATA_ROOT")
    path = root and os.path.join(root, "mipo_hdf5", "indoor", "mipo_joint_stats.json")
    if not path or not os.path.isfile(path):
        pytest.skip("mipo_joint_stats.json not available under $DATA_ROOT")
    with open(path) as f:
        stats = json.load(f)
    for group, ms in _DEFAULT_JOINT_STATS.items():
        assert stats[group]["mean"] == ms["mean"] and stats[group]["std"] == ms["std"], group
