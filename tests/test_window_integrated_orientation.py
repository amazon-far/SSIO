"""Tests for ``Sequence.WindowIntegratedOrientation``.

What is worth pinning is the one way this wrapper can be wrong while staying
finite, right-shaped and silent: anchoring the integration somewhere other than
each window's own start-frame GT (a global anchor, a neighbouring window's
anchor, or free-running drift), and disturbing the streams it must pass through.
"""

import torch
import pypose as pp
from omegaconf import OmegaConf

from Interface.sequence import ISequence
from Interface.data     import IMUData_w_GT
from Sequence.SlidingWindow import SlidingWindow
from Sequence.RandomWindow  import RandomWindow
from Sequence.WindowIntegratedOrientation import WindowIntegratedOrientation


GRAVITY = 9.81
OMEGA_Z = 0.5          # rad/s
DT_S    = 0.005        # 200 Hz
N       = 400          # 2 s

WINDOW  = 100
STEP    = 50


class _SyntheticSequence(ISequence[IMUData_w_GT]):
    """Minimal base loader following the storage convention under ``Sequence/``:
    constant z-spin raw gyro, yaw-only GT rotation."""

    CONFIG_SPEC = {}

    def __init__(self, n: int = N, k: int = 1,
                 corrected_gyro_z: float | None = None,
                 gt_yaw: float = 0.0, gt_yaw_rate: float = 0.0):
        """``corrected_gyro_z`` overrides the corrected gyro's z-rate (None keeps
        the corrected streams equal to raw). GT rotation is a yaw of
        ``gt_yaw + gt_yaw_rate * t`` about z (``gt_yaw_rate=0`` gives a constant
        ``gt_yaw``, both 0 gives identity GT)."""
        super().__init__(OmegaConf.create({}))
        dtype = torch.float32

        t_s = torch.arange(n, dtype=torch.float64) * DT_S
        self._time_ns = (t_s * 1e9).round().to(torch.int64).reshape(n, 1, 1).expand(n, k, 1).contiguous()

        acc  = torch.tensor([0.0, 0.0, GRAVITY], dtype=dtype).expand(n, 3)
        gyro = torch.tensor([0.0, 0.0, OMEGA_Z], dtype=dtype).expand(n, 3)
        self._acc  = acc .unsqueeze(1).expand(n, k, 3).contiguous()
        self._gyro = gyro.unsqueeze(1).expand(n, k, 3).contiguous()

        eye3 = torch.eye(3, dtype=dtype)
        self._acc_cov  = (eye3 * 1e-4).expand(n, k, 3, 3).contiguous()
        self._gyro_cov = (eye3 * 1e-4).expand(n, k, 3, 3).contiguous()

        # GT rotation: yaw of (gt_yaw + gt_yaw_rate * t) about z.
        trans   = torch.linspace(0, 1, n, dtype=dtype).unsqueeze(-1).expand(n, 3)
        gt_ang  = torch.zeros(n, 3, dtype=dtype)
        gt_ang[:, 2] = (gt_yaw + gt_yaw_rate * t_s).to(dtype)
        quat    = pp.so3(gt_ang).Exp().tensor()                                        # (n, 4) xyzw
        self._gt_poses_data = torch.cat([trans, quat], dim=-1).contiguous()
        self._gt_vel_w      = torch.full((n, 3), 2.0, dtype=dtype)

        self._corrected_acc  = self._acc
        self._corrected_gyro = self._gyro
        if corrected_gyro_z is not None:
            self._corrected_gyro = torch.tensor(
                [0.0, 0.0, corrected_gyro_z], dtype=dtype,
            ).expand(n, k, 3).contiguous()

    @property
    def seq_name(self) -> str:
        return "synthetic_spin"

    def __len__(self) -> int:
        return int(self._time_ns.shape[0])

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            index = slice(index, index + 1)
        return IMUData_w_GT(
            time_ns        = self._time_ns [index].unsqueeze(0),
            acc            = self._acc     [index].unsqueeze(0),
            gyro           = self._gyro    [index].unsqueeze(0),
            acc_cov        = self._acc_cov [index].unsqueeze(0),
            gyro_cov       = self._gyro_cov[index].unsqueeze(0),
            gt_poses       = pp.SE3(self._gt_poses_data[index].unsqueeze(0)),
            gt_vel_w       = self._gt_vel_w[index].unsqueeze(0),
            corrected_acc  = self._corrected_acc [index].unsqueeze(0),
            corrected_gyro = self._corrected_gyro[index].unsqueeze(0),
        )


def _wrap(window_cls=SlidingWindow, **seq_kwargs) -> WindowIntegratedOrientation:
    base = _SyntheticSequence(**seq_kwargs)
    if window_cls is SlidingWindow:
        win = SlidingWindow(OmegaConf.create({"window_size": WINDOW, "step_size": STEP}), base)
    else:
        win = RandomWindow(OmegaConf.create({"window_size": WINDOW}), base)
    return WindowIntegratedOrientation(OmegaConf.create({}), win)


def _analytic_local(yaw0: float = 0.0) -> pp.LieTensor:
    """Per-window absolute rotation: GT yaw anchor, then in-window z-spin."""
    ang = torch.zeros(WINDOW, 3, dtype=torch.float64)
    ang[:, 2] = yaw0 + OMEGA_Z * torch.arange(WINDOW, dtype=torch.float64) * DT_S
    return pp.so3(ang).Exp()


def _geodesic_max(got_quat: torch.Tensor, expected: pp.LieTensor) -> float:
    err = pp.SO3(expected.tensor().to(torch.float32)).Inv() @ pp.SO3(got_quat)
    return err.Log().norm(dim=-1).max().item()


def test_each_window_integrates_from_its_own_gt_start():
    # Identity GT everywhere: each window re-anchors to its own identity start, so
    # the local rotation is the same for every window regardless of absolute start.
    seq   = _wrap()
    inner = seq.internal_seq
    batch = seq[0:len(seq)]
    rot   = batch.input_rot                              # (B, WINDOW) SO3
    B     = rot.shape[0]
    assert B > 1
    # gt_poses rotation is preserved bit-identically (non-destructive wrapper).
    assert torch.equal(batch.gt_poses.rotation().tensor(),
                       inner[0:len(inner)].gt_poses.rotation().tensor())
    expected = _analytic_local(0.0)
    for b in range(B):
        assert _geodesic_max(rot.tensor()[b], expected) < 1e-4
    # Frame 0 of every window is the identity GT anchor.
    f0 = rot.tensor()[:, 0]
    assert torch.allclose(f0, pp.identity_SO3(B, 1).tensor()[:, 0].to(f0), atol=1e-5)


def test_per_window_reanchors_to_varying_gt_not_global_or_drift():
    # GT attitude ramps (GT_RATE) while the gyro spins at a different rate (OMEGA_Z),
    # so integrated gyro diverges from GT. Each window re-anchors to its own
    # start-frame GT (yaw GT_RATE * t) then follows the gyro within the window
    # (slope OMEGA_Z).
    GT_RATE = 0.3      # rad/s, the true-attitude ramp (differs from the OMEGA_Z gyro rate)
    seq = _wrap(gt_yaw_rate=GT_RATE)
    rot = seq[0:len(seq)].input_rot                                # (B, WINDOW) SO3
    B   = rot.shape[0]
    assert B > 1

    local = OMEGA_Z * torch.arange(WINDOW, dtype=torch.float64) * DT_S  # in-window gyro spin
    for b in range(B):
        s   = b * STEP                                             # this window's absolute start frame
        ang = torch.zeros(WINDOW, 3, dtype=torch.float64)
        ang[:, 2] = GT_RATE * (s * DT_S) + local                  # window GT anchor + gyro within window
        assert _geodesic_max(rot.tensor()[b], pp.so3(ang).Exp()) < 1e-4

    # Each window's frame-0 yaw is its own GT anchor (GT_RATE * t_start), distinct per window.
    f0_yaw      = rot.Log().tensor()[:, 0, 2]                       # (B,) z-angle at window start
    starts      = torch.tensor([b * STEP for b in range(B)], dtype=f0_yaw.dtype)
    expected_f0 = GT_RATE * starts * DT_S
    assert torch.allclose(f0_yaw, expected_f0, atol=1e-4)
    assert (expected_f0[-1] - expected_f0[0]).abs() > 0.1          # sanity: anchors differ across windows


def test_multi_imu_not_rejected():
    # Multi-IMU (K=2): orientation integrates the main-body gyro (K index 0).
    seq = _wrap(k=2)
    rot = seq[0:len(seq)].input_rot
    assert _geodesic_max(rot.tensor()[0], _analytic_local(0.0)) < 1e-4


def test_integrates_corrected_gyro_stream():
    # Corrected gyro reads zero while raw spins, so rotation stays at the identity
    # GT anchor (confirms integration uses the corrected stream).
    seq = _wrap(corrected_gyro_z=0.0)
    rot = seq[0:len(seq)].input_rot
    assert rot.Log().norm(dim=-1).max().item() < 1e-5


def test_non_rotation_fields_pass_through():
    base = _SyntheticSequence()
    win  = SlidingWindow(OmegaConf.create({"window_size": WINDOW, "step_size": STEP}), base)
    seq  = WindowIntegratedOrientation(OmegaConf.create({}), win)

    got, raw = seq[0:len(seq)], win[0:len(win)]
    # Entire gt_poses (translation AND rotation) is preserved: the wrapper is
    # non-destructive and only sets input_rot.
    assert torch.equal(got.gt_poses.tensor(), raw.gt_poses.tensor())
    assert raw.input_rot is None and got.input_rot is not None
    assert torch.equal(got.gt_vel_w, raw.gt_vel_w)
    assert torch.equal(got.acc,  raw.acc)
    assert torch.equal(got.gyro, raw.gyro)
    assert torch.equal(got.time_ns, raw.time_ns)
    assert torch.equal(got.is_first_window, raw.is_first_window)


def test_works_over_random_window():
    # RandomWindow picks arbitrary starts per read. Per-window re-anchoring makes the
    # local rotation identical regardless of the drawn start.
    seq   = _wrap(window_cls=RandomWindow)
    rot   = seq[0:8].input_rot
    expected = _analytic_local(0.0)
    for b in range(rot.shape[0]):
        assert _geodesic_max(rot.tensor()[b], expected) < 1e-4
