"""Constant IMU bias estimators.

Estimators remove constant offsets without reorienting axes. acc/gyro must
share the body frame that rot_wb maps to z-up world, where
PyPose subtracts g_world = [0, 0, +gravity].
"""

import abc
import dataclasses
import typing as tp

import pypose as pp
import torch
from omegaconf import DictConfig

from Interface.stationary import detect_stationary, median_dt_s

# Status records estimator action, not measured correction quality.
STATUS_DISABLED          = "disabled"           # turned off by config
STATUS_STATIC            = "static"             # estimated on genuinely static frames
STATUS_SKIPPED_SHORT     = "skipped_short"      # sequence shorter than the window
STATUS_SKIPPED_NO_STATIC = "skipped_no_static"  # no qualifying static run; IMU untouched


@dataclasses.dataclass(frozen=True)
class DebiasReport:
    """Per-sequence estimator report exposed as loader.debias_report.

    static_spans are frames judged static; used_spans are frames averaged.
    Both contain ascending, disjoint half-open [start, end) pairs, empty when
    nothing was estimated. Span estimators can average intervening motion;
    window_mean_speed is GT speed over used frames and exposes that difference.
    """

    status           : str
    applied          : bool
    n_total          : int
    n_used           : int                        # frames the estimate averaged over
    acc_bias         : list[float]                # subtracted from acc ([] when not applied)
    gyro_bias        : list[float]
    static_spans     : list[tuple[int, int]]      # frames this estimator judged static
    used_spans       : list[tuple[int, int]]      # frames it averaged the bias over
    window_mean_speed: float | None               # m/s over the frames used
    estimator        : str = ""                   # class name, so a report is self-describing

    @property
    def coverage(self) -> float:
        """Fraction of the sequence the estimate averaged over."""
        return self.n_used / self.n_total if self.n_total else 0.0


def _not_applied(
    status   : str,
    n_total  : int,
    estimator: str,
    static   : list[tuple[int, int]] | None = None,
) -> DebiasReport:
    """Report a declined estimate while retaining any frames that qualified as static."""
    return DebiasReport(
        status=status, applied=False, n_total=n_total, n_used=0,
        acc_bias=[], gyro_bias=[], static_spans=static or [], used_spans=[],
        window_mean_speed=None, estimator=estimator,
    )


def _spans(idx: torch.Tensor) -> list[tuple[int, int]]:
    """Ascending frame indices -> half-open ``[(start, end)]`` contiguous runs."""
    if idx.numel() == 0:
        return []
    i     = idx.flatten().tolist()
    spans = [[i[0], i[0] + 1]]
    for j in i[1:]:
        if j == spans[-1][1]:
            spans[-1][1] = j + 1
        else:
            spans.append([j, j + 1])
    return [(a, b) for a, b in spans]


class IMUDebias(abc.ABC):
    """Removes a constant acc/gyro bias from one sequence and reports on it."""

    def __init__(self, gravity: float):
        self.gravity = float(gravity)

    @property
    def name(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def __call__(
        self,
        acc     : torch.Tensor,                   # (N, 3) body frame
        gyro    : torch.Tensor,                   # (N, 3) body frame
        rot_wb  : pp.LieTensor,                   # SO3 (N,) R_w_from_body
        gt_vel_w: torch.Tensor,                   # (N, 3) world frame
        *,
        time_ns : torch.Tensor | None = None,     # (N,) int64; only ``imu`` needs it
    ) -> tuple[torch.Tensor, torch.Tensor, DebiasReport]:
        """``(acc, gyro, report)`` — debiased streams and what was done."""

class SelectionDebias(IMUDebias):
    """Estimates bias from frames selected as static by subclasses."""

    def __init__(self, gravity: float, *, debias_gyro: bool = True):
        super().__init__(gravity)
        # False: estimate and subtract the acc bias only; the gyro passes
        # through untouched (for IMUs whose gyro is already bias-compensated).
        self.debias_gyro = bool(debias_gyro)

    @abc.abstractmethod
    def _select(
        self,
        acc     : torch.Tensor,
        gyro    : torch.Tensor,
        speed   : torch.Tensor,
        *,
        time_ns : torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str, torch.Tensor]:
        """Return (used_index, status, static_index) as ascending frame indices.

        Used and static sets differ when a selector widens to a span. Empty used_index
        with skipped_* declines the estimate; static_index may still record a near miss.
        """

    def __call__(
        self,
        acc     : torch.Tensor,                   # (N, 3) body frame
        gyro    : torch.Tensor,                   # (N, 3) body frame
        rot_wb  : pp.LieTensor,                   # SO3 (N,) R_w_from_body
        gt_vel_w: torch.Tensor,                   # (N, 3) world frame
        *,
        time_ns : torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, DebiasReport]:
        n     = int(acc.shape[0])
        speed = gt_vel_w.norm(dim=-1)
        idx, status, static_idx = self._select(acc, gyro, speed, time_ns=time_ns)
        if idx.numel() == 0:
            return acc, gyro, _not_applied(status, n, self.name, _spans(static_idx))

        # Subtract gravity in each selected frame's body coordinates before
        # averaging; selected static frames may have different attitudes.
        g_w       = acc.new_tensor([0.0, 0.0, self.gravity])
        g_in_body = pp.SO3(rot_wb[idx]).Inv() @ g_w                     # (S, 3)
        acc_bias  = (acc[idx] - g_in_body).mean(dim=0)                   # (3,)
        if self.debias_gyro:
            gyro_bias = gyro[idx].mean(dim=0)                                # (3,) zero rate at rest
        else:
            gyro_bias = torch.zeros_like(acc_bias)

        report = DebiasReport(
            status            = status,
            applied           = True,
            n_total           = n,
            n_used            = int(idx.numel()),
            acc_bias          = acc_bias .detach().flatten().tolist(),
            gyro_bias         = gyro_bias.detach().flatten().tolist(),
            static_spans      = _spans(static_idx),
            used_spans        = _spans(idx),
            window_mean_speed = float(speed[idx].mean()),
            estimator         = self.name,
        )
        return acc - acc_bias, gyro - gyro_bias, report


class StaticSpanDebias(SelectionDebias):
    """Averages the span between the first and last GT-static frame.

    The span may include motion; see ``Doc/debias.md`` before selecting this mode.
    """

    #: GT speed threshold in m/s, specific to this span estimator.
    static_thresh: float = 0.001
    min_win: int = 10

    def _select(
        self, acc: torch.Tensor, gyro: torch.Tensor, speed: torch.Tensor,
        *, time_ns: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str, torch.Tensor]:
        del time_ns
        n       = int(acc.shape[0])
        min_win = min(self.min_win, n)

        static_idx = (speed < self.static_thresh).nonzero(as_tuple=False).squeeze(-1)
        if static_idx.numel() < min_win:
            return (torch.empty(0, dtype=torch.long, device=acc.device),
                    STATUS_SKIPPED_NO_STATIC, static_idx)

        win_start = int(static_idx[0].item())
        win_end   = min(n, int(static_idx[-1].item()) + 1)
        # The span between the first and last static frame is at least as long as
        # the static frames it brackets, so min_win is already satisfied here.
        idx = torch.arange(win_start, win_end, device=acc.device)
        return idx, STATUS_STATIC, static_idx


class StaticSetDebias(SelectionDebias):
    """Averages frames covered by a run passing GT-speed and gyro gates."""

    def __init__(
        self,
        gravity  : float,
        *,
        window     : int   = 90,     # frames (~0.45 s @ 200 Hz)
        speed_max  : float = 0.02,   # m/s,   max GT speed inside the run
        gyro_max   : float = 0.05,   # rad/s, max |gyro| inside the run
        debias_gyro: bool  = True,
    ):
        super().__init__(gravity, debias_gyro=debias_gyro)
        self.window    = int(window)
        self.speed_max = float(speed_max)
        self.gyro_max  = float(gyro_max)

    def _select(
        self, acc: torch.Tensor, gyro: torch.Tensor, speed: torch.Tensor,
        *, time_ns: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str, torch.Tensor]:
        del time_ns
        n     = int(acc.shape[0])
        w     = self.window
        empty = torch.empty(0, dtype=torch.long, device=acc.device)
        if n < w:
            return empty, STATUS_SKIPPED_SHORT, empty

        speed_w = speed              .unfold(0, w, 1)                   # (n-w+1, w)
        gyro_w  = gyro.norm(dim=-1)  .unfold(0, w, 1)
        ok = ((speed_w.max(dim=1).values < self.speed_max)
              & (gyro_w .max(dim=1).values < self.gyro_max))
        if not bool(ok.any()):
            # The gate qualifies runs; isolated slow frames do not qualify.
            return empty, STATUS_SKIPPED_NO_STATIC, empty

        # Dilate run-qualification to per-frame membership: frame j is static iff
        # some qualifying run covers it, i.e. ok[i] for some i in [j-w+1, j].
        # conv1d with a ones kernel and padding w-1 is exactly that coverage count.
        cover = torch.nn.functional.conv1d(
            ok.to(acc.dtype).view(1, 1, -1),
            torch.ones(1, 1, w, dtype=acc.dtype, device=acc.device),
            padding=w - 1,
        ).view(-1)[:n]
        idx = (cover > 0).nonzero(as_tuple=False).squeeze(-1)
        return idx, STATUS_STATIC, idx


#: Allowed detector overrides; reject unknown config keys at construction.
_DETECTOR_KWARGS = frozenset({"a_std_max", "gyro_mean_max", "gyro_std_max", "grav_dev_max", "win_s"})


class StaticIMUDebias(SelectionDebias):
    """Select static frames from raw IMU using the shared ZUPT detector.

    Defaults match the loader stationary mask; detector overrides can change the
    selected set. No GT speed is read, but accelerometer bias still uses rot_wb
    to express gravity in body coordinates. Bias and tilt are inseparable at one
    static attitude, so only the gyro estimate is fully label-free.

    time_ns sets the detector's seconds-based window. min_static_s is an optional
    quorum; below it the estimator declines while reporting the static frames.
    """

    def __init__(
        self,
        gravity     : float,
        min_static_s: float | None = None,
        *,
        debias_gyro : bool = True,
        **detector_kwargs: float,
    ):
        super().__init__(gravity, debias_gyro=debias_gyro)
        if min_static_s is not None and min_static_s <= 0.0:
            raise ValueError(f"{self.name}: min_static_s must be > 0 s (or omitted), got {min_static_s}")
        self.min_static_s = min_static_s
        bad = set(detector_kwargs) - _DETECTOR_KWARGS
        if bad:
            raise ValueError(
                f"{self.name}: unknown detector kwargs {sorted(bad)}; "
                f"allowed: {sorted(_DETECTOR_KWARGS)}"
            )
        self.detector_kwargs = {k: float(v) for k, v in detector_kwargs.items()}

    def _select(
        self, acc: torch.Tensor, gyro: torch.Tensor, speed: torch.Tensor,
        *, time_ns: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str, torch.Tensor]:
        del speed
        if time_ns is None:
            raise ValueError(
                f"{self.name} needs time_ns: the stationary window is in seconds. "
                "Pass the loader's int64 clock to the calibrater call."
            )
        n     = int(acc.shape[0])
        empty = torch.empty(0, dtype=torch.long, device=acc.device)
        dt    = median_dt_s(time_ns)
        win_s = self.detector_kwargs.get("win_s", 1.0)
        if n < round(win_s / dt):
            return empty, STATUS_SKIPPED_SHORT, empty

        mask = detect_stationary(
            acc.unsqueeze(0), gyro.unsqueeze(0), dt,
            gravity=self.gravity, **self.detector_kwargs,
        )[0]
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        quorum = round(self.min_static_s / dt) if self.min_static_s is not None else 1
        if idx.numel() < quorum:
            # Decline, but hand back the near-miss frames as static_spans for the audit.
            return empty, STATUS_SKIPPED_NO_STATIC, idx
        return idx, STATUS_STATIC, idx


#: Use kind rather than _target_: outer Hydra recursion would instantiate
#: before the loader can supply gravity.
CALIBRATERS: dict[str, type[IMUDebias]] = {
    "span": StaticSpanDebias,
    "set" : StaticSetDebias,
    "imu" : StaticIMUDebias,
}


def is_calibrater_node(value: tp.Any) -> bool:
    """Validate null or a known kind; build_calibrater's constructor checks remaining keys."""
    return value is None or (
        isinstance(value, DictConfig) and str(value.get("kind")) in CALIBRATERS
    )


def disabled_report(n_total: int) -> DebiasReport:
    """Publish a report even when disabled; AirIMU and evaluators read it unconditionally."""
    return _not_applied(STATUS_DISABLED, n_total, "none")


def run_calibrater(
    config  : DictConfig,
    gravity : float,
    tag     : str,
    acc     : torch.Tensor,
    gyro    : torch.Tensor,
    rot_wb  : pp.LieTensor,
    gt_vel_w: torch.Tensor,
    *,
    time_ns : torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, DebiasReport] | tuple[None, None, DebiasReport]:
    """Debias single-IMU (N, 3) body-frame streams before adding the K axis.

    Returns (corrected_acc, corrected_gyro, report), with both streams in the body frame.
    rot_wb maps body to world; gt_vel_w is world-frame. Disabled correction
    returns (None, None, disabled_report), keeping absence distinct from raw IMU.
    """
    if bool(config.debias):
        result = build_calibrater(config, gravity)(
            acc, gyro, rot_wb, gt_vel_w, time_ns=time_ns,
        )
    else:
        result = (None, None, disabled_report(int(acc.shape[0])))
    report = result[2]
    print(f"[{tag}] debias {report.status} " + (
        f"over {report.n_used}/{report.n_total} static frames | "
        f"gyro bias {report.gyro_bias} | acc bias {report.acc_bias}"
        if report.applied else f"({report.n_total} frames); IMU left untouched"), flush=True)
    return result


def build_calibrater(config: DictConfig, gravity: float) -> IMUDebias:
    """Build the explicitly named estimator; null is invalid when debiasing.

    Remaining node keys are constructor kwargs. The loader supplies gravity so
    calibration and downstream preintegration subtract the same value.
    """
    node = config.calibrater
    if node is None:
        raise ValueError(
            "debias: true needs a `calibrater: {kind: ...}` node naming the estimator "
            f"(one of {sorted(CALIBRATERS)}); there is no default."
        )
    params = {str(k): v for k, v in node.items()}
    kind   = params.pop("kind", None)
    if kind not in CALIBRATERS:
        raise ValueError(f"calibrater.kind must be one of {sorted(CALIBRATERS)}, got {kind!r}.")
    return CALIBRATERS[kind](gravity, **params)
