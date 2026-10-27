import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp

import jaxtyping as jt
from typing import cast, Sequence

from .Encoder.CNN  import MotionEncoder
from .Network      import INetwork, cholesky_cov


#: Which IMU stream a network reads. ``IMUData`` splits the pair by CONTENT:
#: ``acc``/``gyro`` are as measured, ``corrected_acc``/``corrected_gyro`` are the
#: output of bias calibration (debias).
#:
#: Independent of the stream the preintegrated Δv reference integrates, which the
#: training recipe names separately (``train.dv_imu_input``).
IMU_STREAMS = ("raw", "corrected")


def select_imu(imu, stream: str):
    """Return the (acc, gyro) pair named by ``stream``."""
    if stream == "corrected":
        return (
            cast(torch.Tensor, imu.corrected_acc),
            cast(torch.Tensor, imu.corrected_gyro),
        )
    assert stream == "raw"
    return imu.acc, imu.gyro


class AirIONetwork(INetwork):
    """Strided CNN + GRU body-frame velocity network with a Cholesky covariance head.

    Inputs: body-frame acc/gyro (gravity not removed) and rot = GT R_{w<-b}.Log().
    Output vel is body-frame at each strided anchor (stride prod(s_list)); the
    caller rotates to world via R_wb. vel_cov = L @ Lᵀ (softplus diag, 1e-6 floor).
    Multi-IMU (K>1) is fused inside the conv stem's channel dim (in_channels =
    6*num_imus), not mean-fused at the output; see downsample() for anchor placement.
    """

    def __init__(
        self,
        k_list:               Sequence[int],
        s_list:               Sequence[int],
        p_list:               Sequence[int],
        imu_encoder_channels: Sequence[int],
        ori_encoder_channels: Sequence[int],
        fusion_dim:           int,
        gru1_hidden:          int,
        gru2_hidden:          int,
        decoder_hidden:       int,
        bidirectional:        bool,
        num_imus:             int,
        imu_input:            str,
    ) -> None:
        super().__init__()
        if imu_input not in IMU_STREAMS:
            raise ValueError(f"imu_input must be one of {IMU_STREAMS}, got {imu_input!r}")
        self.imu_input = str(imu_input)

        # Model/Encoder/CNN.py's MotionEncoder zip-truncates mismatched lists in
        # silence; this graph depends on the four staying in lockstep. The input
        # channel is computed (6 * num_imus), so the encoder lists carry only the
        # per-block output widths.
        if not (
            len(imu_encoder_channels)
            == len(ori_encoder_channels)
            == len(k_list)
            == len(s_list)
            == len(p_list)
        ):
            raise ValueError(
                "expected equal encoder-layout lengths, got "
                f"imu_encoder_channels={list(imu_encoder_channels)}, "
                f"ori_encoder_channels={list(ori_encoder_channels)}, "
                f"k_list={list(k_list)}, s_list={list(s_list)}, p_list={list(p_list)}"
            )
        if len(k_list) != 2:
            raise ValueError(
                f"AirIONetwork currently assumes a 2-layer CNN encoder; got k_list={list(k_list)}. "
                "The end-of-segment anchor formula in `downsample` is hand-derived for L=2."
            )
        if num_imus < 1:
            raise ValueError(f"num_imus must be >= 1, got {num_imus}")
        self.num_imus = num_imus

        imu_c_list = [6 * num_imus, *imu_encoder_channels] # acc + gyro
        ori_c_list = [3, *ori_encoder_channels] # so(3) tangent of GT R_{w<-b} (axis-angle)

        # Store CNN layout so `downsample` can compute the output-to-input anchor mapping.
        self.k_list = tuple(k_list)
        self.s_list = tuple(s_list)
        self.p_list = tuple(p_list)

        self.feature_encoder = MotionEncoder(
            c_list=imu_c_list, k_list=k_list, s_list=s_list, p_list=p_list,
        )
        self.ori_encoder = MotionEncoder(
            c_list=ori_c_list, k_list=k_list, s_list=s_list, p_list=p_list,
        )

        concat_dim = imu_encoder_channels[-1] + ori_encoder_channels[-1]
        self.fcn2       = nn.Linear(concat_dim, fusion_dim)
        self.batchnorm2 = nn.BatchNorm1d(fusion_dim)
        self.gelu       = nn.GELU()

        factor    = 2 if bidirectional else 1
        self.gru1 = nn.GRU(input_size=fusion_dim          , hidden_size=gru1_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.gru2 = nn.GRU(input_size=gru1_hidden * factor, hidden_size=gru2_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)

        self.veldecoder = nn.Sequential(
            nn.Linear(gru2_hidden * factor, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 3),
        )

        # Cholesky-factor head: 6 unique entries of a 3x3 lower-triangular L.
        # Layout of the output channels:
        #     [L_00, L_11, L_22, L_10, L_20, L_21]
        # The first three are mapped through softplus(+ 1e-6) to guarantee a
        # strictly positive diagonal of L; the remaining three are unconstrained.
        self.velcov_decoder = nn.Sequential(
            nn.Linear(gru2_hidden * factor, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 6),
        )

    @property
    def stride(self) -> int:
        """Total temporal stride between consecutive output anchors (= prod(s_list))."""
        return self.s_list[0] * self.s_list[1]

    @property
    def start_index(self) -> int:
        """Input-frame index of output anchor 0 (matches the slice in ``downsample``)."""
        return (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1])

    def conv_output_length(self, n_frames: int) -> int:
        """Raw number of values the conv stack emits for ``n_frames`` inputs.

        Exceeds ``anchor_count`` whenever padding lets the last kernel positions
        hang off the end of the input.
        """
        length = n_frames
        for k, s, p in zip(self.k_list, self.s_list, self.p_list):
            length = (length + 2 * p - k) // s + 1
        return length

    def anchor_count(self, n_frames: int) -> int:
        """Number of output anchors backed by a real input frame.

        Anchor ``i`` aligns to input frame ``start_index + i*stride``. The conv
        padding lets the stack emit further outputs that align *past* the end of
        the input (frame >= n_frames); those have no ground truth to compare
        against, so they are not anchors. ``forward`` trims them off and
        ``downsample`` never invents them.

        The count is derived from the input length, never from the conv output
        length minus a constant -- the gap is not always 1 (it is 2 at
        n_frames=256 with the default kernels).
        """
        s_idx = self.start_index
        if n_frames <= s_idx:
            raise ValueError(
                f"{type(self).__name__}: need more than start_index={s_idx} frames "
                f"to form an anchor; got n_frames={n_frames}."
            )
        return (n_frames - 1 - s_idx) // self.stride + 1

    def downsample(
        self,
        x: jt.Float[torch.Tensor, "*B N C"],
        *,
        repeat_tail: bool = False,
    ) -> jt.Float[torch.Tensor, "*B M C"]:
        """Slice (*B, N, C) onto the output anchors: x[..., s_idx::stride, :].

            s_idx  = (k_0 - p_0) + s_0 * (k_1 - 1 - p_1)     # end-of-segment + 1
            stride = s_0 * s_1

        Returns exactly ``anchor_count(N)`` rows, every one a real sample of
        ``x`` -- matching what ``forward`` returns, so predictions and targets
        line up without any call-site slicing.

        ``repeat_tail=True`` additionally repeats the final row out to
        ``conv_output_length(N)``. Those extra rows are **fabricated, not
        sampled**: pass it only to line up against an untrimmed conv output, and
        never to build a training target or a metric. Supports torch.Tensor and
        pp.LieTensor.
        """
        selected = x[..., self.start_index::self.stride, :]

        # Tripwire: the slice length is the anchor count by construction. If a
        # kernel/stride change ever breaks that, fail here rather than silently
        # shifting every target by a frame.
        expected = self.anchor_count(x.shape[-2])
        assert selected.shape[-2] == expected
        if not repeat_tail:
            return selected

        n_phantom = self.conv_output_length(x.shape[-2]) - selected.shape[-2]
        if n_phantom <= 0:
            return selected

        repeats = (1,) * (x.ndim - 2) + (n_phantom, 1)
        tail = x[..., -1:, :].repeat(*repeats)
        if isinstance(x, pp.LieTensor):
            # torch.cat may drop the LieTensor subclass (cf. collate_dataclass).
            raw = torch.cat(
                [selected.as_subclass(torch.Tensor), tail.as_subclass(torch.Tensor)],
                dim=-2,
            )
            return pp.LieTensor(raw, ltype=x.ltype)
        return torch.cat([selected, tail], dim=-2)

    def _encode_imu(self, x: "INetwork.Input") -> torch.Tensor:
        """Encode K IMUs to ``(B, T, imu_encoder_channels[-1])``.

        The ``K`` physical IMUs are flattened into the channel dim of a single
        Conv1d across all IMUs at once. K=1 collapses to the original 6-channel
        stem Conv1d behaviour.
        """
        acc, gyro = select_imu(x.imu, self.imu_input)
        imu_feat = torch.cat([acc, gyro], dim=-1)                # (B, N, K, 6)
        B, N, K, _ = imu_feat.shape
        if K != self.num_imus:
            raise ValueError(
                f"{type(self).__name__}: the conv stem was built for num_imus="
                f"{self.num_imus} (in_channels={6 * self.num_imus}), but the batch "
                f"carries K={K} IMUs. The K axis is folded into the channel dim, so "
                "the two must match exactly."
            )
        imu_feat = imu_feat.reshape(B, N, K * 6)                 # (B, N, 6K)
        # Conv1d wants (B, C, N); transpose back to (B, T, C) after.
        return self.feature_encoder(imu_feat.transpose(-1, -2)).transpose(-1, -2)

    def _encode_rot(self, x: "INetwork.Input") -> torch.Tensor:
        """Encode the so(3) attitude-tangent stream to ``(B, T, ori_encoder_channels[-1])``."""
        rot_feat = x.rot.tensor()                                # (B, N, 3)
        return self.ori_encoder(rot_feat.transpose(-1, -2)).transpose(-1, -2)

    def encode(self, x: "INetwork.Input") -> torch.Tensor:
        """Encode and fuse all input streams to hidden ``(B, T, fusion_dim)`` feature
        passed into the GRU trunk.

        This is *the* variation point of the family: every subclass changes only
        which streams it encodes and which module projects the concatenation to
        ``fusion_dim``.
        """

        f = torch.cat([self._encode_imu(x), self._encode_rot(x)], dim=-1)           # (B, T, concat_dim)
        f = self.fcn2(f)                                                            # (B, T, fusion_dim)
        f = self.batchnorm2(f.transpose(-1, -2)).transpose(-1, -2)
        return self.gelu(f)

    def forward(self, x: INetwork.Input) -> INetwork.Output:
        N = x.imu.acc.shape[1]
        return self.decode_and_trim(self.encode(x), N)

    def decode_and_trim(
        self,
        f: jt.Float[torch.Tensor, "B T C"],
        N: int,
    ) -> INetwork.Output:
        """GRU trunk → vel/cov decode → anchor trim.

        Everything downstream of ``encode``, so a variant that only changes how
        ``f`` is built (``AirIONetworkIMUOnly``, ``JointsAirIONetworkIMUOnly``) shares
        this and cannot drift from it.

        ``f`` is ``(B, T, fusion_dim)``, i.e. what the GRU trunk consumes: there is
        no folded IMU axis, because ``_encode_imu`` fuses the K physical IMUs inside
        the conv stem's channel dim (``6*num_imus``). ``N`` is the
        number of *input* frames, needed for the anchor trim.
        """
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        f, _ = self.gru1(f)                                                        # (B, T, gru1_hidden)
        f, _ = self.gru2(f)                                                        # (B, T, gru2_hidden)

        vel     = self.veldecoder(f)                                               # (B, T, 3)
        vel_cov = cholesky_cov(self.velcov_decoder(f.detach()))                    # (B, T, 3, 3) — stop-grad shared features

        # Drop the trailing conv outputs that align past the end of the input:
        # they are computed mostly from zero-padding and have no ground truth, so
        # supervising or scoring them compares a prediction against a fabricated
        # target. Trimming here keeps forward and downsample on one grid.
        M = self.anchor_count(N)
        return self.Output(vel_b=vel[:, :M], vel_cov_b=vel_cov[:, :M])


# The per-joint measurements a loader may carry. A network's `joint_channels` is a
# subset, taken verbatim: the config list *is* the conv-stem channel layout, so two
# configs naming the same channels in a different order build different (equally
# valid) stems whose checkpoints are not interchangeable. Referenced by name in
# `_encode_joint_stack`, so a typo there surfaces as a clear error on the first
# forward rather than needing a separate construction-time validator.
JOINT_CHANNELS: tuple[str, ...] = ("position", "velocity", "effort")


def _encode_joint_stack(
    encoder: MotionEncoder,
    x: "INetwork.Input",
    owner: str,
    channels: tuple[str, ...],
) -> torch.Tensor:
    """Encode the joint stack to ``(B, T, joint_encoder_channels[-1])``.

    The ``channels`` fields are concatenated to C = len(channels) channels per
    joint and the J joints flattened into the channel dim
    ((B, N, J, C) -> (B, N, C*J)).

    Which channels exist is a property of the robot, not of this family, so the set
    is configured per network rather than hardcoded — it also doubles as the lever
    for a leave-one-channel-out ablation. A channel the batch carries as None is an
    error, never a zero-fill, which would feed the conv stem a dead constant that
    reads as a real measurement.

    Module-level rather than a method so every joints network in this family shares
    one flatten convention. Channel order *is* the checkpoint layout: joint j
    occupies channels C*j..C*j+C-1, so two networks that flattened differently
    could not exchange a ``joint_encoder`` state dict, and a loader emitting joints
    in a different order would silently train against scrambled channels.
    """
    assert x.joints is not None, f"{owner} requires INetwork.Input.joints; got None"
    parts = []
    for name in channels:
        # getattr default folds "not a JointData field at all" (a typo in the YAML)
        # into the same error as "field exists but this platform does not measure it".
        field = getattr(x.joints, name, None)
        if field is None:
            raise ValueError(
                f"{owner}: joint_channels requests {name!r}, which this batch does not "
                f"provide (unknown field, or carried as None). Valid channels are "
                f"{list(JOINT_CHANNELS)}; drop it from joint_channels, or use a dataset "
                f"that measures it."
            )
        parts.append(field)
    joint_feat = torch.cat(parts, dim=-1)                                # (B, N, J, C)
    B, N, J, C = joint_feat.shape
    joint_feat = joint_feat.reshape(B, N, J * C)                         # (B, N, C*J)
    # Conv1d wants (B, C, N); transpose back to (B, T, C) after.
    return encoder(joint_feat.transpose(-1, -2)).transpose(-1, -2)


class AirIONetworkIMUOnly(AirIONetwork):
    """Body-frame AirIO network without orientation input.

    Anchor geometry and multi-IMU handling match ``AirIONetwork``. ``Input.rot``
    is accepted and ignored. Checkpoints are not interchangeable because this
    class has no ``ori_encoder`` and a different ``fcn2`` input width.
    """

    def __init__(
        self,
        k_list:               Sequence[int],
        s_list:               Sequence[int],
        p_list:               Sequence[int],
        imu_encoder_channels: Sequence[int],
        fusion_dim:           int,
        gru1_hidden:          int,
        gru2_hidden:          int,
        decoder_hidden:       int,
        bidirectional:        bool,
        num_imus:             int,
        imu_input:            str,
    ) -> None:
        # Skip AirIONetwork.__init__ (it builds the ori branch); inherit only its
        # geometry methods, which read k_list/s_list/p_list set below.
        INetwork.__init__(self)
        if imu_input not in IMU_STREAMS:
            raise ValueError(f"imu_input must be one of {IMU_STREAMS}, got {imu_input!r}")
        self.imu_input = str(imu_input)

        if not (len(imu_encoder_channels) == len(k_list) == len(s_list) == len(p_list)):
            raise ValueError(
                f"expected len(imu_encoder_channels) == len(k_list) == len(s_list) == len(p_list), got "
                f"imu_encoder_channels={list(imu_encoder_channels)}, k_list={list(k_list)}, "
                f"s_list={list(s_list)}, p_list={list(p_list)}"
            )
        if len(k_list) != 2:
            raise ValueError(
                f"AirIONetworkIMUOnly assumes a 2-layer CNN encoder; got k_list={list(k_list)}. "
                "The end-of-segment anchor formula in `downsample` is hand-derived for L=2."
            )
        if num_imus < 1:
            raise ValueError(f"num_imus must be >= 1, got {num_imus}")
        self.num_imus = num_imus

        self.k_list = tuple(k_list)
        self.s_list = tuple(s_list)
        self.p_list = tuple(p_list)

        self.feature_encoder = MotionEncoder(
            c_list=[6 * num_imus, *imu_encoder_channels],  # acc + gyro per IMU
            k_list=k_list, s_list=s_list, p_list=p_list,
        )

        self.fcn2       = nn.Linear(imu_encoder_channels[-1], fusion_dim)
        self.batchnorm2 = nn.BatchNorm1d(fusion_dim)
        self.gelu       = nn.GELU()

        factor    = 2 if bidirectional else 1
        self.gru1 = nn.GRU(input_size=fusion_dim          , hidden_size=gru1_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.gru2 = nn.GRU(input_size=gru1_hidden * factor, hidden_size=gru2_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)

        self.veldecoder = nn.Sequential(
            nn.Linear(gru2_hidden * factor, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 3),
        )
        self.velcov_decoder = nn.Sequential(
            nn.Linear(gru2_hidden * factor, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 6),
        )

    def encode(self, x: "INetwork.Input") -> torch.Tensor:
        """IMU stream only — ``x.rot`` is ignored by design (class docstring).

        The single overridden method: ``forward`` and ``decode_and_trim`` are
        inherited verbatim, so the twin cannot drift from ``AirIONetwork``
        downstream of the encoder.
        """
        f = self._encode_imu(x)                                                     # (B, T, imu_encoder_channels[-1])
        f = self.fcn2(f)                                                            # (B, T, fusion_dim)
        f = self.batchnorm2(f.transpose(-1, -2)).transpose(-1, -2)
        return self.gelu(f)


class JointsAirIONetworkIMUOnly(AirIONetworkIMUOnly):
    """Fuse body-frame IMU and joint streams without orientation input.

    ``Input.rot`` is accepted and ignored.
    """

    def __init__(
        self,
        k_list:                 Sequence[int],
        s_list:                 Sequence[int],
        p_list:                 Sequence[int],
        imu_encoder_channels:   Sequence[int],
        joint_encoder_channels: Sequence[int],
        joint_channels:         Sequence[str],
        fusion_dim:             int,
        fusion_hidden:          int,
        gru1_hidden:            int,
        gru2_hidden:            int,
        decoder_hidden:         int,
        bidirectional:          bool,
        num_imus:               int,
        num_joints:             int,
        imu_input:              str,
    ) -> None:
        # Builds the IMU stem, fcn2, the GRU trunk and both heads, and validates
        # the k/s/p list lengths against imu_encoder_channels.
        super().__init__(
            k_list=k_list, s_list=s_list, p_list=p_list,
            imu_encoder_channels=imu_encoder_channels, fusion_dim=fusion_dim,
            gru1_hidden=gru1_hidden, gru2_hidden=gru2_hidden,
            decoder_hidden=decoder_hidden, bidirectional=bidirectional,
            num_imus=num_imus, imu_input=imu_input,
        )

        if num_joints < 1:
            raise ValueError(f"num_joints must be >= 1, got {num_joints}")
        self.num_joints = num_joints
        self.joint_channels = tuple(joint_channels)

        if len(joint_encoder_channels) != len(k_list):
            raise ValueError(
                f"expected len(joint_encoder_channels) == len(k_list), got "
                f"joint_encoder_channels={list(joint_encoder_channels)}, "
                f"k_list={list(k_list)}"
            )
        # Same k/s/p as the IMU encoder, so both streams land on one time grid and
        # the concatenation in `encode` lines up frame-for-frame.
        self.joint_encoder = MotionEncoder(
            c_list=[len(self.joint_channels) * num_joints, *joint_encoder_channels],
            k_list=k_list, s_list=s_list, p_list=p_list,
        )

        # 3-layer MLP, not the base twin's single `fcn2` Linear: kinematic joint<->IMU coupling (contact-dependent
        # stance/swing, forward kinematics) is more nonlinear than IMU<->attitude.
        concat_dim = imu_encoder_channels[-1] + joint_encoder_channels[-1]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim,    fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, fusion_dim),
        )

    def encode(self, x: "INetwork.Input") -> torch.Tensor:
        """Fuse IMU + joints to (B, T, fusion_dim) — ``x.rot`` is ignored by design.

        The single overridden method: ``forward`` and ``decode_and_trim`` are
        inherited verbatim, so this twin cannot drift from ``AirIONetwork``
        downstream of the encoder.
        """
        f = torch.cat(
            [
                self._encode_imu(x),
                _encode_joint_stack(
                    self.joint_encoder, x, type(self).__name__, self.joint_channels,
                ),
            ],
            dim=-1,
        )                                                                           # (B, T, concat_dim)
        f = self.fusion_mlp(f)                                                      # (B, T, fusion_dim)
        f = self.batchnorm2(f.transpose(-1, -2)).transpose(-1, -2)
        return self.gelu(f)
