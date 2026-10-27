"""Shared eval figures for the two training stages.

Both stages log the same two per-sequence views, so they live here rather than in
one stage importing plots from the other. Every function returns a Figure and
logs nothing: the caller owns the wandb key and the plt.close().
"""

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot, as in the entrypoints
import matplotlib.pyplot as plt
import torch

from Utils.trainer import integrate_trapezoid



def plot_velocity_series(
    t_plot_s,            # (T,)    seconds, NaN row per window boundary
    gt_vel_w_plot,       # (T, 3)  world-frame GT velocity
    pred_vel_w_plot,     # (T, 3)  world-frame prediction
    preint_vel_w_plot,   # (T, 3)  world-frame preintegrated reference, or None (PGD has none)
    std_vel_w_plot,      # (T, 3)  per-axis σ (already √diag of the rotated cov)
    window_edges,        # (W,)    window start times, drawn as guides
    title: str,
):
    """3-panel world-frame velocity vs time (x/y/z), pred ±1σ shaded.

    A time series, not a spatial path. Inputs are the NaN-broken ``*_plot``
    copies, so matplotlib lifts the pen between windows instead of drawing a
    jump. Returns the figure; the caller logs and closes it.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ax, axis_name, ch in zip(axes, ("x", "y", "z"), range(3)):
        ax.plot(t_plot_s, gt_vel_w_plot    [:, ch], label="GT",     linewidth=1.0)
        ax.plot(t_plot_s, pred_vel_w_plot  [:, ch], label="pred",   linewidth=1.0, alpha=0.8)
        if preint_vel_w_plot is not None:
            ax.plot(t_plot_s, preint_vel_w_plot[:, ch], label="preint", linewidth=1.0, alpha=0.8)
        ax.fill_between(
            t_plot_s,
            pred_vel_w_plot[:, ch] - std_vel_w_plot[:, ch],
            pred_vel_w_plot[:, ch] + std_vel_w_plot[:, ch],
            color="C1", alpha=0.2, linewidth=0, label="pred ±1σ",
        )
        ax.set_ylabel(f"v_{axis_name} (m/s)")
        ax.grid(alpha=0.3)
        # Compare against None explicitly: avoid ValueError if window_edges is an ndarray.
        for edge in (window_edges if window_edges is not None else ()):
            ax.axvline(edge, color="0.7", linewidth=0.5, alpha=0.5, zorder=0)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_trajectory_2d(
    pred_vel_w: torch.Tensor,   # (W, M, 3) world-frame prediction, NO NaN breaks
    gt_vel_w:   torch.Tensor,   # (W, M, 3) world-frame GT
    t_2d_ns:    torch.Tensor,   # (W, M)    int64 ns
    title: str,
):
    """2D trajectory, pred vs GT, side by side in the XY and XZ planes.

    Positions are the origin-anchored trapezoidal integral of world-frame
    velocity — the same construction ``evaluate.py`` uses, so these shapes and
    its post-hoc ATE are built identically.

    XZ carries information XY cannot: vertical drift from an accelerometer-bias
    or gravity-magnitude error is invisible from directly above.

    Takes the UNBROKEN (W, M, 3) series, never the ``*_plot`` copies: those carry
    NaN separator rows and a cumulative sum over a NaN poisons every later
    sample, which would blank the path after the first window boundary.
    Returns the figure; the caller logs and closes it.
    """
    pred_pos = integrate_trapezoid(pred_vel_w.reshape(-1, 3), t_2d_ns.reshape(-1))
    gt_pos   = integrate_trapezoid(gt_vel_w.reshape(-1, 3),   t_2d_ns.reshape(-1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (h, v, h_name, v_name) in zip(axes, ((0, 1, "X", "Y"), (0, 2, "X", "Z"))):
        ax.plot(gt_pos  [:, h], gt_pos  [:, v], label="GT",   linewidth=1.5, linestyle="--")
        ax.plot(pred_pos[:, h], pred_pos[:, v], label="pred", linewidth=1.5)
        ax.scatter([0], [0], c="k", s=25, zorder=3, label="start")
        ax.set_xlabel(f"{h_name} (m)")
        ax.set_ylabel(f"{v_name} (m)")
        ax.set_title(f"{h_name}{v_name} plane")
        # Equal aspect per panel so a metre reads the same on both axes; each
        # panel keeps its own limits, since XZ is far flatter than XY.
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
