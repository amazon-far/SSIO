import dataclasses
import typing as tp

import torch
import pypose as pp
import jaxtyping as jt

from dataclasses import dataclass

if tp.TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass
class IMUData:
    """Multi-IMU stack contract: K physical IMUs at different mount points on
    the same rigid body. Body-level fields (``gt_poses``, ``gt_vel_w``) live on
    ``IMUData_w_GT`` without K — they describe the body trajectory, shared
    across IMUs. Per-IMU fields below carry an explicit ``K`` axis. Single-IMU
    loaders set ``K=1``.
    """
    time_ns       : jt.Int64[torch.Tensor, "B N K 1"]
    acc           : jt.Float[torch.Tensor, "B N K 3"]
    gyro          : jt.Float[torch.Tensor, "B N K 3"]
    acc_cov       : jt.Float[torch.Tensor, "B N K 3 3"]
    gyro_cov      : jt.Float[torch.Tensor, "B N K 3 3"]
    # As-measured and correction-chain output remain separate. None means no correction
    # stage ran; a stage that declined still returns its identity output so mixed pools
    # remain collatable.
    corrected_acc : tp.Optional[jt.Float[torch.Tensor, "B N K 3"]]
    corrected_gyro: tp.Optional[jt.Float[torch.Tensor, "B N K 3"]]
    # True only for the window at frame 0 of the sequence; set by the window
    # wrappers, None otherwise.
    is_first_window: tp.Optional[jt.Bool[torch.Tensor, "B"]] = None
    # Start frame of each window in the underlying sequence; set by the window
    # wrappers, None otherwise.
    window_start: tp.Optional[jt.Int64[torch.Tensor, "B"]] = None
    # Source sequence index; set by ``Sequence.TagSequence``, None otherwise.
    seq_id: tp.Optional[jt.Int64[torch.Tensor, "B"]] = None
    # Optional network-input attitude override (SO3, body->world). When present,
    # the velocity network is conditioned on THIS attitude instead of the GT
    # rotation in ``gt_poses`` (see ``get_est_velocity``), letting a chosen
    # orientation source (gyro-integrated, onboard AHRS, an external estimator)
    # be fed to the network while ``gt_poses`` stays the immutable GT used for
    # anchors/metrics and the pseudo-label frame. Populated ONLY by an explicit
    # wrapper (``ExternalOrientation`` / ``WindowIntegratedOrientation``) — never
    # auto-filled by a loader from data availability. None => network uses GT.
    input_rot: tp.Optional[jt.Float[pp.LieTensor, "B N 4"]] = None
    # Per-frame IMU-detected zero-velocity (stationary) mask. Populated ONLY by the
    # ``Sequence.ZeroVelocity`` wrapper (full-sequence stationary detection),
    # None otherwise. Consumed by the stationary regularization term.
    stationary: tp.Optional[jt.Bool[torch.Tensor, "B N"]] = None


@dataclass
class GTState:
    gt_poses   : jt.Float[pp.LieTensor, "B N 7"]
    gt_vel_w   : jt.Float[torch.Tensor, "B N 3"]


@dataclass
class IMUData_w_GT(IMUData, GTState):
    pass


@dataclass
class JointData:
    """Proprioceptive joint-state contract: J revolute/contact joints sampled
    on a single synchronized timeline (one joint topic), so ``joint_time_ns``
    carries no per-joint axis — unlike ``IMUData.time_ns`` (per-IMU, genuinely
    async). Each per-joint field carries an explicit ``J`` axis with a trailing
    scalar channel for parallelism with the IMU stack's ``3`` channel.

    ``effort`` is ``Optional`` because not every platform measures it (ANYmal,
    MIPO and the Unitree G1 all record joint torque, but a new platform may not).
    It is deliberately Optional **without a default** — a loader must
    pass ``effort=None`` explicitly rather than omitting it, so an absent torque
    channel is a stated fact and never a zero-filled tensor masquerading as a
    measurement. Which channels a network actually consumes is its own
    ``joint_channels`` config; requesting ``effort`` from a batch that carries
    None raises there.

    (A default is also structurally impossible: dataclass fields are collected in
    reverse-MRO order, so ``JointData``'s land before ``GTState``'s in
    ``IMUJointsData_w_GT`` and a default here would put a non-default ``gt_poses``
    after it.)
    """
    joint_time_ns: jt.Int64[torch.Tensor, "B N 1"]
    position:      jt.Float[torch.Tensor, "B N J 1"]
    velocity:      jt.Float[torch.Tensor, "B N J 1"]
    effort:        tp.Optional[jt.Float[torch.Tensor, "B N J 1"]]


@dataclass
class IMUJointsData_w_GT(IMUData_w_GT, JointData):
    """Multi-IMU stack + proprioceptive joints + body GT. An instance is-a
    ``IMUData``, ``GTState`` and ``JointData`` simultaneously, so it can be
    passed into separately-typed network-input slots without extraction."""
    pass


DataType = tp.TypeVar("DataType", bound="DataclassInstance")


def cat_field(values: list, dim: int = 0) -> tp.Any:
    """Concatenate one dataclass field, preserving LieTensor type."""
    first = values[0]
    if first is None and all(v is None for v in values):
        return None
    if isinstance(first, pp.LieTensor):
        raw = torch.cat([t.as_subclass(torch.Tensor) for t in values], dim=dim)
        return pp.LieTensor(raw, ltype=first.ltype)
    return torch.cat(values, dim=dim)


def move_dataclass(data: DataType, device: torch.device) -> DataType:
    """The dataclass with every tensor field on ``device`` (None fields kept).

    LieTensors are re-wrapped explicitly rather than trusting subclass
    propagation through ``.to`` (cf. ``cat_field``). Needed by the train loops
    once datasets stay CPU-resident for multi-worker loading; a no-op-per-field
    when the batch is already on ``device``.
    """
    kwargs: dict[str, tp.Any] = {}
    for f in dataclasses.fields(data):
        v = getattr(data, f.name)
        if isinstance(v, pp.LieTensor):
            kwargs[f.name] = pp.LieTensor(v.as_subclass(torch.Tensor).to(device), ltype=v.ltype)
        elif isinstance(v, torch.Tensor):
            kwargs[f.name] = v.to(device)
        else:
            kwargs[f.name] = v
    return type(data)(**kwargs)


def collate_dataclass(data: list[DataType]) -> DataType:
    cls = type(data[0])
    assert all(type(d) is cls for d in data)

    kwargs = {
        f.name: cat_field([getattr(d, f.name) for d in data])
        for f in dataclasses.fields(data[0])
    }
    return cls(**kwargs)
