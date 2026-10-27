"""The data / network contract, asserted on the objects the closed-loop run produced.

Every fixture here is real: ``evaluator.batch`` is the collated ``IMUData_w_GT`` that
``evaluate.run_sequence``'s own DataLoader hands the network, ``evaluator.network`` is
the checkpoint ``trained_run`` wrote, and the variant sequences come from the *stored*
eval node with one config path overridden (``evaluator.build_sequence``). Nothing is
built from a hand-written OmegaConf and no network is built from constructor literals,
so a config or loader change cannot leave a test passing against a stale copy of itself.

Each invariant is kept because it fails as a plausible wrong *number* rather than as a
crash: a dropped K axis, a float32'd nanosecond clock, two swapped sigmas, a cloned
"corrected" stream, windows that reflatten out of order, or predictions one anchor out of
phase with their targets all leave every shape and dtype intact.
"""
import dataclasses

import pypose as pp
import pytest
import torch

from Sequence.debias import STATUS_DISABLED
from omegaconf import OmegaConf

from .conftest import network_yaml

pytestmark = pytest.mark.closedloop

#: Per-IMU fields carry an explicit K axis (``Interface/data.py``) because K physical
#: IMUs sit at different mount points on one body; body-level GT describes the body and
#: must NOT. field -> trailing shape after (B, N, K).
_PER_IMU_TRAILING = {
    "time_ns":        (1,),
    "acc":            (3,),
    "gyro":           (3,),
    "corrected_acc":  (3,),
    "corrected_gyro": (3,),
    "acc_cov":        (3, 3),
    "gyro_cov":       (3, 3),
}
_BODY_LEVEL = {"gt_poses", "gt_vel_w"}
#: Body-level as well — no K axis — but populated ONLY by an explicit wrapper, so they are
#: None on a plain loader read. Each is one fact about the *body*, which is why a K axis
#: would be wrong rather than merely unused:
#:   ``is_first_window``  one bool per window (which window is anchored at frame 0).
#:   ``window_start``     one start frame per window, set by the window wrappers.
#:   ``seq_id``           one source-trajectory index per window, set by TagSequence.
#:   ``input_rot``        one body->world attitude. Both writers read the main-body IMU
#:       (K index 0) and publish a single rotation — ``WindowIntegratedOrientation``
#:       integrates ``corrected_gyro[:, :, 0]``, ``ExternalOrientation`` reads a per-
#:       sequence quaternion stream — and ``get_est_velocity`` conditions the network on
#:       that one rotation. K copies of a rigid body's attitude could only disagree.
#:   ``stationary``       one motion state. ``ZeroVelocity.detect_stationary`` runs on the
#:       K=0 raw stream over the full sequence; the body is either at rest or it is not, so
#:       a per-IMU mask would be K votes on a body-level fact, and the stationary
#:       regularization it feeds acts on the *body* velocity.
_OPTIONAL   = {"is_first_window", "window_start", "seq_id", "input_rot", "stationary"}

#: Float streams, i.e. everything ``config.dtype`` governs. ``gt_poses`` is a LieTensor
#: and is checked alongside them; ``time_ns`` is deliberately not in here.
_FLOAT_STREAMS = ("acc", "gyro", "corrected_acc", "corrected_gyro",
                  "acc_cov", "gyro_cov", "gt_vel_w")


#: ``corrected_acc`` / ``corrected_gyro`` are None when the dataset ran no correction
#: stage. The loops below skip them in that case but assert that nothing ELSE is None, so
#: a field going missing still fails.
_MAY_BE_NONE = {"corrected_acc", "corrected_gyro"}


def _streams(batch, names):
    for name in names:
        value = getattr(batch, name)
        if value is None:
            assert name in _MAY_BE_NONE, f"{name} must never be None"
            continue
        yield name, value


def test_eval_batch_fields_split_into_per_imu_and_body_level(evaluator):
    """The K axis, on the batch the eval run actually collated.

    The field-set equality is a tripwire: a new ``IMUData_w_GT`` field must be
    classified as per-IMU (gets a K axis) or body-level (must not), because that
    decision is the whole multi-IMU contract and a field that quietly lands on the
    wrong side broadcasts instead of failing.

    ``input_rot`` and ``stationary`` are body-level (see ``_OPTIONAL``): one attitude and
    one motion state for the whole rigid body, both derived from the main-body IMU at K
    index 0, so neither may grow a K axis. Both are also wrapper-populated, so this test
    asserts both halves of that: None on the plain eval node (a loader must never
    auto-fill them, and collate must not invent them), and K-free at the frame axis once
    the wrappers that own them are composed underneath the window.
    """
    batch = evaluator.batch
    names   = {f.name for f in dataclasses.fields(batch)}
    covered = set(_PER_IMU_TRAILING) | _BODY_LEVEL | _OPTIONAL
    assert names == covered, (
        f"IMUData_w_GT fields changed: {names ^ covered}. Classify the new field as "
        "per-IMU (gets a K axis) or body-level (must not) and extend this test."
    )

    windows = len(evaluator.sequence)
    frames  = int(evaluator.cfg.train.window)
    for name, value in _streams(batch, _PER_IMU_TRAILING):
        trailing = _PER_IMU_TRAILING[name]
        assert tuple(value.shape) == (windows, frames, 1, *trailing), (
            f"{name} must be (B={windows}, N={frames}, K=1, {trailing}); "
            f"got {tuple(value.shape)}"
        )

    assert tuple(batch.gt_vel_w.shape) == (windows, frames, 3)
    # SE3 stores 7 columns (xyz + xyzw); lshape is the Lie-tensor batch shape.
    assert tuple(batch.gt_poses.lshape) == (windows, frames)
    assert tuple(batch.gt_poses.shape)  == (windows, frames, 7)

    # Set by the window wrappers only — one value per window, never per frame.
    # ``train_pgd`` looks labels up by ``window_start``.
    assert batch.is_first_window.dtype is torch.bool
    assert tuple(batch.is_first_window.shape) == (windows,)
    assert batch.window_start.dtype is torch.int64
    assert tuple(batch.window_start.shape) == (windows,)
    assert evaluator.sequence.internal_seq[0:16].window_start is None
    assert evaluator.sequence.internal_seq[0:16].is_first_window is None, (
        "a bare loader read must leave is_first_window None; only the wrappers set it"
    )

    # The other two wrapper-owned fields: absent here, because the run's eval node is a
    # bare SlidingWindow over the loader. A loader that starts inferring either one from
    # data availability, or a collate that materialises a default, fails here.
    assert evaluator.sequence.internal_seq[0:16].input_rot  is None, (
        "a bare loader read must leave input_rot None; only ExternalOrientation / "
        "WindowIntegratedOrientation set it"
    )
    assert evaluator.sequence.internal_seq[0:16].stationary is None, (
        "a bare loader read must leave stationary None; only ZeroVelocity sets it"
    )
    assert batch.input_rot  is None, "collate invented an input_rot"
    assert batch.stationary is None, "collate invented a stationary mask"


def test_wrapper_populated_input_rot_and_stationary_stay_body_level(evaluator, smoke, tmp_path):
    """``input_rot`` / ``stationary``, once their wrappers are composed, carry no K axis.

    The claim in ``_OPTIONAL`` is only half asserted by the None checks above: a field
    that is always None cannot show a wrong axis. So the *stored* eval node is
    re-instantiated with the two wrappers that own these fields inserted between the
    window and the loader — the layer they must sit at, since both index frames — and the
    result is collated through the same DataLoader the eval path uses.

    Shapes are the whole point: ``stationary`` is (B, N) and ``input_rot`` is (B, N)
    SO3 / (B, N, 4). A K axis on either would broadcast silently against the per-IMU
    streams instead of failing, which is exactly the failure the field-set tripwire above
    exists to prevent. ``ExternalOrientation`` is used for ``input_rot`` rather than
    ``WindowIntegratedOrientation`` because the smoke loader ships ``debias: false``, and
    that wrapper refuses to integrate the as-measured gyro (it raises on
    ``corrected_gyro is None``) — the orientation source differs, the axis contract does not.

    Non-destructiveness is asserted with it: both wrappers document that ``gt_poses`` is
    left untouched, because it is the GT used for anchors, metrics and the pseudo-label
    frame. A wrapper that overwrote it would still produce a correctly-shaped batch.
    """
    import pickle

    from omegaconf import OmegaConf as OC

    # Identity attitude for every frame of the source clip: a value that is provably NOT
    # the GT rotation, so overwriting gt_poses with it could not go unnoticed below.
    quat = torch.zeros(smoke.frames, 4)
    quat[:, 3] = 1.0
    pickle_path = tmp_path / "external_orientation.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({smoke.seq_name: {"orientation": quat.numpy()}}, f)

    # The stored loader node, interpolations unresolved, re-parented under the two
    # wrappers. Below the window on purpose: both index the frame axis, and
    # ZeroVelocity's detector must see full-sequence context.
    loader_node = OC.to_container(evaluator.cfg.eval_dataset.sequences[0].seq, resolve=False)
    wrapped = evaluator.build_sequence({"eval_dataset.sequences.0.seq": {
        "_target_": "Sequence.ZeroVelocity.ZeroVelocity",
        "config": {},
        "seq": {
            "_target_": "Sequence.ExternalOrientation.ExternalOrientation",
            "config": {"path": str(pickle_path)},
            "seq": loader_node,
        },
    }})
    batch   = evaluator.load_batch(wrapped)
    windows = len(wrapped)
    frames  = int(evaluator.cfg.train.window)

    assert batch.stationary is not None, "ZeroVelocity did not reach the collated batch"
    assert batch.stationary.dtype is torch.bool
    assert tuple(batch.stationary.shape) == (windows, frames), (
        f"stationary must be (B={windows}, N={frames}) with no K axis; "
        f"got {tuple(batch.stationary.shape)}"
    )

    assert batch.input_rot is not None, "ExternalOrientation did not reach the collated batch"
    assert tuple(batch.input_rot.lshape) == (windows, frames), (
        f"input_rot must be (B={windows}, N={frames}) SO3 with no K axis; "
        f"got {tuple(batch.input_rot.lshape)}"
    )
    assert tuple(batch.input_rot.shape) == (windows, frames, 4)

    # gt_poses / gt_vel_w survive both wrappers byte for byte.
    assert torch.equal(batch.gt_poses.tensor(), evaluator.batch.gt_poses.tensor()), \
        "a wrapper overwrote gt_poses; anchors, metrics and the pseudo-label frame move with it"
    assert torch.equal(batch.gt_vel_w, evaluator.batch.gt_vel_w)
    assert not torch.equal(batch.input_rot.tensor(), batch.gt_poses.rotation().tensor()), (
        "the injected attitude equals the GT rotation, so this test could not tell an "
        "input_rot override from a gt_poses passthrough"
    )


def test_float_streams_honour_the_config_dtype_and_time_ns_stays_int64(evaluator):
    """``meta.dtype`` governs every float stream; ``time_ns`` stays int64.

    Nanosecond timestamps are integers on purpose (``EuRoC_Sequence`` rounds back to the
    raw file's ns) — casting them to the config dtype would lose ~µs resolution in fp32,
    which is what the ΔV horizon search and ``trend_mean_dt_sec`` walk on.

    Checked at the run's own dtype *and* at an fp64 override of the same node: with one
    dtype only, a stream hardcoded to float32 is indistinguishable from a compliant one.
    """
    run_dtype = evaluator.cfg.meta.dtype
    assert run_dtype is next(evaluator.network.parameters()).dtype, (
        "evaluate.py builds the network and the data from the same cfg.meta.dtype; a "
        "mismatch means one of the two stopped reading it"
    )

    fp64_batch = evaluator.load_batch(
        evaluator.build_sequence({"meta.dtype": "${as_dtype:'fp64'}"})
    )
    for batch, dtype in ((evaluator.batch, run_dtype), (fp64_batch, torch.float64)):
        for name, value in _streams(batch, _FLOAT_STREAMS):
            assert value.dtype is dtype, name
        assert batch.gt_poses.dtype is dtype
        assert batch.time_ns.dtype is torch.int64
    assert fp64_batch.acc.dtype is not run_dtype, "the fp64 override did not take"


def test_covariances_are_sigma_squared_identity_with_the_sigmas_not_swapped(evaluator):
    """acc_cov / gyro_cov are ``sigma**2 * I`` built from the config's two sigmas.

    The smoke groups happen to use one value for both sigmas, which cannot detect a
    swapped assignment — so a second instance of the *same* node is built with distinct
    ones. Both feed the preintegrator's noise model, so a swap is a silently mis-weighted
    covariance, never an error.
    """
    seq_config = evaluator.cfg.eval_dataset.sequences[0].seq.config
    batch      = evaluator.batch
    eye        = torch.eye(3, dtype=batch.acc_cov.dtype)

    assert torch.allclose(batch.acc_cov,  eye * float(seq_config.acc_sigma) ** 2)
    assert torch.allclose(batch.gyro_cov, eye * float(seq_config.gyro_sigma) ** 2)

    probe = evaluator.load_batch(evaluator.build_sequence({
        "eval_dataset.sequences.0.seq.config.acc_sigma":  0.03,
        "eval_dataset.sequences.0.seq.config.gyro_sigma": 0.07,
    }))
    assert torch.allclose(probe.acc_cov,  eye * 0.03 ** 2)
    assert torch.allclose(probe.gyro_cov, eye * 0.07 ** 2)


def test_corrected_streams_match_the_correction_chain(evaluator):
    """``corrected_*`` is the correction-chain output; ``acc``/``gyro`` stay as measured.

    The network and Δv target select their streams independently, so swapping either
    pair silently changes the training objective while preserving every shape and dtype.

    Both cases are asserted off the loader's own ``debias_report``, so this test states
    the invariant rather than the fixture's happenstance:
      - correction applied  -> the two pairs must differ
      - correction declined -> an empty chain is the identity, so they must be equal
    """
    loader   = evaluator.sequence.internal_seq
    applied  = loader.debias_report.applied

    if loader.debias_report.status == STATUS_DISABLED:
        # No stage ran: there is no corrected stream at all, and it must NOT be aliased
        # to the as-measured one.
        assert loader._corrected_acc  is None
        assert loader._corrected_gyro is None
        read = loader[0:len(loader)]
        assert read.corrected_acc  is None
        assert read.corrected_gyro is None
        assert evaluator.batch.corrected_acc is None, "collate invented a corrected stream"
        return

    if applied:
        assert not torch.equal(loader._corrected_acc,  loader._acc), \
            "debias applied but corrected_acc == acc: the calibrater output was discarded"
        assert not torch.equal(loader._corrected_gyro, loader._gyro)
    else:
        assert torch.equal(loader._corrected_acc,  loader._acc)
        assert torch.equal(loader._corrected_gyro, loader._gyro)

    # The split must survive __getitem__ and collate, which is where the two consumers
    # actually read it.
    read  = loader[0:len(loader)]
    assert torch.equal(read.corrected_acc [0], loader._corrected_acc)
    assert torch.equal(read.corrected_gyro[0], loader._corrected_gyro)

    batch = evaluator.batch
    if applied:
        assert not torch.equal(batch.corrected_acc, batch.acc), "collate lost the split"
    else:
        # Stage ran but declined on this clip: its output is its input.
        assert torch.equal(batch.corrected_acc, batch.acc)


def test_gt_attitude_is_unit_norm_and_every_stream_is_finite(evaluator):
    """A NaN or an off-sphere quaternion poisons preintegration without raising.

    ``run_sequence`` feeds ``gt_poses.rotation()`` to the preintegrator's ``init_state``
    and ``.Log()`` to the network's orientation input, so a SLERP result that drifted
    off the unit sphere is a wrong rotation, not an exception.
    """
    batch = evaluator.batch
    for name, value in _streams(batch, _FLOAT_STREAMS):
        assert torch.isfinite(value).all(), f"{name} is non-finite"
    assert torch.isfinite(batch.gt_poses.tensor()).all()

    quat_norm = batch.gt_poses.rotation().tensor().norm(dim=-1)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-6)


def test_sliding_windows_reflatten_to_the_source_in_time_order(evaluator, smoke):
    """``evaluate.py`` folds the window axis into time and treats it as one trajectory.

    ``evaluate.main`` (and ``evaluator.score``) reshape the (W, M, ·) chunks to (W*M, ·)
    and hand them to ``compute_metrics`` / ``integrate_trapezoid`` as a single
    time-ordered stream. That is only valid while the windows tile
    the source in order with no gap, overlap or reordering. The run's own eval group
    yields a single window at ``train.window=1024``, which cannot see a reordering, so
    the same node is re-instantiated at a window that tiles four times.
    """
    window   = 500
    sequence = evaluator.build_sequence({"train.window": window})
    batch    = evaluator.load_batch(sequence)

    assert len(sequence) == smoke.frames // window
    source = sequence.internal_seq[0:smoke.frames]
    for name in ("acc", "gyro", "time_ns", "gt_vel_w"):
        stacked = getattr(batch, name)
        assert stacked.shape[:2] == (len(sequence), window)
        flat = stacked.reshape(1, -1, *stacked.shape[2:])
        assert torch.equal(flat, getattr(source, name)), name

    # is_first_window is true exactly once, for the window anchored at frame 0.
    assert batch.is_first_window.tolist() == [True] + [False] * (len(sequence) - 1)

    # The consequence, through the shipped eval path: the flattened anchor clock is
    # strictly increasing, so the trapezoidal integration in score() is not integrating
    # backwards across a window boundary.
    _pred, _gt, _rot, t_ns_w, *_preint = evaluator.run(seq=sequence)
    t_ns = t_ns_w.reshape(-1)
    assert t_ns.numel() == len(sequence) * t_ns_w.shape[1]
    assert (t_ns.diff() > 0).all(), "flattened anchor timestamps are not time-ordered"


def test_forward_and_downsample_agree_on_the_anchor_count(evaluator):
    """M is ``anchor_count(N)`` on both sides, at two window lengths.

    ``forward`` trims the conv outputs that align past the end of the input; ``downsample``
    never invents them. If the two disagree, every loss and metric
    compares predictions against targets one anchor out of phase — a plausible wrong number
    rather than a crash. Two window lengths are needed because the phantom-tail gap is not a
    constant: ``conv_output_length - anchor_count`` is 2 at N=256 and 1 at N=1024, so a
    reintroduced fixed offset passes at one N and fails at the other.
    """
    network = evaluator.network
    gaps    = set()

    for window in (256, int(evaluator.cfg.train.window)):
        sequence = (evaluator.sequence if window == int(evaluator.cfg.train.window)
                    else evaluator.build_sequence({"train.window": window}))
        pred_w, gt_w, gt_rot_w, t_ns_w, *preint = evaluator.run(seq=sequence)

        anchors = network.anchor_count(window)
        assert pred_w.shape == (len(sequence), anchors, 3), (
            f"N={window}: forward returned {tuple(pred_w.shape)}, "
            f"expected {(len(sequence), anchors, 3)}"
        )
        # gt_w / gt_rot_w / t_ns_w / preint_* all come from network.downsample.
        assert gt_w.shape == gt_rot_w.shape == pred_w.shape
        assert t_ns_w.shape == (len(sequence), anchors)
        for value in preint:
            assert value.shape == pred_w.shape

        gap = network.conv_output_length(window) - anchors
        assert gap > 0, f"N={window}: forward is returning untrimmed conv outputs"
        gaps.add(gap)

    assert len(gaps) == 2, (
        f"the two window lengths no longer straddle the varying phantom tail (gaps={gaps}); "
        "pick another N so a constant offset cannot pass"
    )

    # Every published AirIO number comes from the acausal wiring that
    # Config/network/airio.yaml selects (``bidirectional: true``), i.e. an anchor's
    # prediction legitimately depends on later frames. The anchor grid asserted above is
    # silent about which wiring produced it, so pin the wiring the run actually built.
    assert network.gru1.bidirectional and network.gru2.bidirectional


def test_predicted_covariance_is_symmetric_positive_definite(evaluator):
    """``vel_cov = L @ Lᵀ`` with a softplus+1e-6 diagonal, so Cholesky succeeds.

    Forwarded exactly as ``evaluate.run_sequence`` and ``train_ssio`` do it —
    ``run_sequence`` itself discards ``vel_cov_b``, so this is the only closed-loop route
    to the covariance head. ``SupervisedVelocityLoss``'s NLL reads it directly, where a singular row is a NaN gradient rather than an exception.
    """
    from Model.Network import INetwork

    batch = evaluator.batch
    with torch.no_grad():
        output = evaluator.network(INetwork.Input(imu=batch, rot=batch.gt_poses.rotation().Log()))

    anchors = evaluator.network.anchor_count(int(evaluator.cfg.train.window))
    cov = output.vel_cov_b
    assert cov.shape == (len(evaluator.sequence), anchors, 3, 3)
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    torch.linalg.cholesky(cov)                      # raises if not SPD
    assert torch.isfinite(output.vel_b).all()


def test_imu_only_twin_shares_the_anchor_grid_and_ignores_rot(evaluator):
    """``network=airio_imu_only`` is a real ablation of the attitude *input*.

    Class-level on purpose: the trained run builds the default ``airio``, so the twin is
    unreachable through the entrypoints — but it is instantiated from the shipped
    ``Config/network/airio_imu_only.yaml`` and forwarded on the real eval batch, so no
    layout is copied here as a literal.

    Two claims, both of which would otherwise fail as a wrong conclusion rather than a
    wrong run: the twin sits on the same anchor grid (or its numbers are not comparable
    with airio's at all), and two forwards differing only in ``rot`` are bit-identical
    (or the "no attitude" arm still sees attitude and the ablation measures nothing).
    """
    from Interface     import instantiate
    from Model.Network import INetwork

    cfg   = evaluator.cfg
    twin  = instantiate(INetwork, OmegaConf.load(network_yaml("airio_imu_only")))
    twin  = twin.to(cfg.meta.device, cfg.meta.dtype)
    twin.eval()
    airio = evaluator.network

    assert (twin.stride, twin.start_index) == (airio.stride, airio.start_index)
    assert twin.anchor_count(int(cfg.train.window)) == airio.anchor_count(int(cfg.train.window))
    assert not any(k.startswith("ori_encoder.") for k in twin.state_dict()), (
        "the twin still carries the orientation encoder; checkpoints and parameter "
        "counts would silently match airio's"
    )

    batch    = evaluator.batch
    real_rot = batch.gt_poses.rotation().Log()          # what every call site passes
    zero_rot = pp.so3(torch.zeros_like(real_rot.tensor()))
    with torch.no_grad():
        with_gt   = twin(INetwork.Input(imu=batch, rot=real_rot)).vel_b
        with_zero = twin(INetwork.Input(imu=batch, rot=zero_rot)).vel_b
    assert torch.equal(with_gt, with_zero), (
        "airio_imu_only's output depends on Input.rot, so attitude leaks into the "
        "no-attitude arm and every airio-vs-imu_only comparison is confounded"
    )
