"""The typed network contract.

Sole home for ``INetwork``. Other modules import it from here rather than defining
their own, so every ``isinstance`` guard in the repo checks the same class.

Depends on torch, pypose, jaxtyping and beartype only: no hydra, no omegaconf, no
config system. That keeps ``Model/`` importable without the configuration layer.
"""

import typing as tp
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import pypose as pp
import jaxtyping as jt
from beartype import beartype


#: Floor on the diagonal of L. softplus is positive in exact arithmetic but not in
#: floating point: it decays like exp(x), and ``L_ii ** 2`` squares that, so a cov
#: logit below about -53 already makes ``Sigma_ii`` underflow to exactly 0 in fp32
#: — while softplus itself only reaches 0 near -104. A zero on the diagonal makes
#: the covariance singular, and the Gaussian NLL that trains this head
#: (``Loss/ssio.py::SupervisedVelocityLoss``) then returns ``log(0) = -inf`` with
#: NaN gradients. That is not a corner case: the log-det term *rewards* shrinking
#: the covariance, so descent drives those logits toward -inf wherever the error
#: is small. 1e-6 sits ~6 orders above the fp32 cliff and ~6 below any physical
#: velocity uncertainty, so it cannot bias a converged covariance.
CHOLESKY_DIAG_FLOOR = 1e-6


def cholesky_cov(
    raw_L: jt.Float[torch.Tensor, "*B 6"],
) -> jt.Float[torch.Tensor, "*B 3 3"]:
    """Assemble ``vel_cov = L @ Lᵀ`` from a 6-channel covariance head output.

    Channel layout is ``[L_00, L_11, L_22, L_10, L_20, L_21]``. The diagonal goes
    through ``softplus`` plus :data:`CHOLESKY_DIAG_FLOOR` so it is *strictly*
    positive and the product is symmetric positive-**definite**, not merely
    semi-definite. The three off-diagonals stay unconstrained: any real lower
    triangular L with a positive diagonal yields an SPD product, and they have to
    be free to express correlation between the velocity axes.

    Lives here, next to the ``Output.vel_cov_b`` it fills, because it *is* that
    field's contract — every network in the repo that emits a full 3x3 velocity
    covariance calls this rather than re-deriving the assembly.
    """
    L = raw_L.new_zeros(*raw_L.shape[:-1], 3, 3)
    L[..., 0, 0] = F.softplus(raw_L[..., 0]) + CHOLESKY_DIAG_FLOOR
    L[..., 1, 1] = F.softplus(raw_L[..., 1]) + CHOLESKY_DIAG_FLOOR
    L[..., 2, 2] = F.softplus(raw_L[..., 2]) + CHOLESKY_DIAG_FLOOR
    L[..., 1, 0] = raw_L[..., 3]
    L[..., 2, 0] = raw_L[..., 4]
    L[..., 2, 1] = raw_L[..., 5]
    return L @ L.transpose(-1, -2)


@tp.runtime_checkable
class ImuStack(tp.Protocol):
    """The IMU fields a network reads, described structurally.

    Typed as a Protocol rather than a concrete dataclass so this module needs no
    import from any project's data layer. The trade-off is real: this checks that
    ``acc``/``gyro`` are *present*, not that they are tensors — see
    ``Input.__post_init__``, which restores the part that matters.

    K is the multi-IMU axis; single-IMU loaders set K=1.
    """

    acc:  jt.Float[torch.Tensor, "B N K 3"]
    gyro: jt.Float[torch.Tensor, "B N K 3"]


@tp.runtime_checkable
class JointStack(tp.Protocol):
    """The proprioceptive joint fields a network reads, described structurally.

    Same Protocol trade-off as ``ImuStack``: presence, not tensor-ness, is
    checked here — ``Input.__post_init__`` restores the part that matters.

    J is the joint axis; the trailing scalar channel parallels the IMU stack's 3.

    ``effort`` is Optional: platforms that do not measure joint torque (Unitree G1
    ``/lowstate`` carries only q and dq) set it to None. A network declares which
    channels it consumes via its own ``joint_channels``; asking for a channel the
    batch carries as None is an error there, not a silent zero-fill.
    """

    joint_time_ns: jt.Int64[torch.Tensor, "B N 1"]
    position:      jt.Float[torch.Tensor, "B N J 1"]
    velocity:      jt.Float[torch.Tensor, "B N J 1"]
    effort:        tp.Optional[jt.Float[torch.Tensor, "B N J 1"]]


class INetwork(torch.nn.Module):
    @jt.jaxtyped(typechecker=beartype)
    @dataclass
    class Input:
        """What every caller builds to forward a network: ``imu`` plus attitude.

        ``rot`` is **offered, not required**. Call sites fill it unconditionally
        from ``data.gt_poses.rotation().Log()`` — GT attitude on plain loaders,
        gyro-integrated attitude when the dataset is wrapped in
        ``Sequence.WindowIntegratedOrientation`` — but whether a network *reads* it
        is per-implementation: ``AirIONetwork`` conditions on it (the AirIO
        contract: attitude is a known input), while ``AirIONetworkIMUOnly``
        (``network=airio_imu_only``) has no ``ori_encoder`` and ignores the field
        entirely. So a filled ``rot`` is not evidence that a given model needs
        attitude; check the implementation.

        Which IMU pair a network reads is its own ``imu_input`` setting; the training
        recipe selects the preintegrated Δv stream independently.
        """
        imu: ImuStack
        rot: jt.Float[pp.LieTensor, "B N 3"]    # so3 rotation R_{w <- b}
        # Proprioceptive joint data, present only for some datasets/networks.
        # A batch carrying joints alongside IMU (e.g. IMUJointsData_w_GT) is
        # passed as the same object into both slots; plain IMU batches pass None.
        joints: tp.Optional[JointStack] = None

        def __post_init__(self) -> None:
            # ImuStack is structural, so it accepts any object carrying the two
            # attribute names regardless of what they hold; check they are tensors.
            if not (torch.is_tensor(self.imu.acc) and torch.is_tensor(self.imu.gyro)):
                raise TypeError(
                    f"INetwork.Input.imu must carry tensor acc/gyro, got "
                    f"acc={type(self.imu.acc).__name__}, gyro={type(self.imu.gyro).__name__}"
                )
            if self.joints is not None and not (
                torch.is_tensor(self.joints.position) and torch.is_tensor(self.joints.velocity)
            ):
                raise TypeError(
                    f"INetwork.Input.joints must carry tensor position/velocity, got "
                    f"position={type(self.joints.position).__name__}, "
                    f"velocity={type(self.joints.velocity).__name__}"
                )

    @jt.jaxtyped(typechecker=beartype)
    @dataclass
    class Output:
        """Network prediction, in the **body frame** — hence the ``_b`` suffix.

        The frame is in the names deliberately: GT velocity is ``gt_vel_w``
        (world), these are body, and every consumer must rotate before comparing
        (``R_{w<-b}`` from ``Input.rot``). Confusing the two does not raise — it
        silently yields a plausible, wrong number.
        """
        vel_b    : jt.Float[torch.Tensor, "B N 3"]
        vel_cov_b: jt.Float[torch.Tensor, "B N 3 3"]

    def forward(self, x: Input) -> Output: ...

    def anchor_count(self, n_frames: int) -> int:
        """Number of output anchors backed by a real input frame.

        ``forward`` returns exactly this many predictions, and ``downsample``
        returns exactly this many target rows. Derive it from the input length —
        never from the raw conv output length minus a constant.
        """
        ...

    def downsample(
        self,
        x: jt.Float[torch.Tensor, "*B N C"],
        *,
        repeat_tail: bool = False,
    ) -> jt.Float[torch.Tensor, "*B M C"]:
        """Slice ``x`` onto the output anchor grid, returning ``anchor_count(N)`` rows.

        Every returned row is a real sample of ``x``, so predictions and targets
        align with no call-site slicing. ``repeat_tail=True`` repeats the final row out to
        the raw conv output length; those extra rows are **fabricated, not
        sampled**, and must not reach a training target or a metric.
        """
        ...

    @property
    def stride(self) -> int:
        """Total temporal stride between consecutive output anchors in input frames."""
        ...

    @property
    def start_index(self) -> int:
        """Input-frame index of output anchor 0 (the end-of-segment + 1 anchor)."""
        ...
