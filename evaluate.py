"""Standalone post-training evaluator for SSIO checkpoints.

Loads a run's saved Hydra config + checkpoint (positional ``hydra_output_dir``,
optional ``--checkpoint``; defaults to the highest-step weight), runs the eval
sequences, and writes metrics + plots to ``<hydra_output_dir>/eval/<stem>/``
(result.json, summary.json, plus per-sequence trajectory / velocity /
translation / rotation PNGs).

Per-sequence world-frame metrics (positions are origin-anchored trapezoidal
integrals — no SE3/Umeyama alignment, so ATE is cumulative velocity drift):
    * AVE       — mean ‖pred_v − gt_v‖.
    * ATE       — RMSE of ‖pred_pos − gt_pos‖; ATE_mean is the mean companion.
    * ΔV(T)     — non-overlapping Δv error at T ∈ {1, 2, 10} s.
    * RVTE(h)   — overlapping Δv RMSE at horizon h, with scale-free nRVTE(h).
                  Invariant to constant bias.
"""
import argparse
import glob
import json
import os
import re
import typing as tp

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import torch
from omegaconf            import OmegaConf
from torch.utils.data     import DataLoader

from Interface            import instantiate                # registers OmegaConf resolvers (host_env / as_device / as_dtype)
from Interface.data       import IMUData_w_GT, collate_dataclass
from Interface.model      import INetwork, get_est_velocity
from Interface.sequence   import ISequence
from Utils.modality       import apply_modality_mask
from Utils.trainer        import integrate_trapezoid


_CKPT_RE         = re.compile(r"checkpoint_(\d+)\.pt$")
_DV_WINDOWS_S    = (1.0, 2.0, 10.0)
_RVTE_HORIZONS_S = (1.0, 2.0, 5.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "hydra_output_dir", type=str,
        help="Hydra run directory (must contain .hydra/config.yaml and checkpoint_*.pt).",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Optional checkpoint filename inside the hydra dir (or absolute path); "
             "defaults to the highest-step checkpoint_*.pt.",
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="Optional path to a dataset YAML (e.g. ./Config/dataset/TLIO_Test.yaml) that "
             "overrides cfg.eval_dataset. Interpolations in the override (e.g. ${train.window}, "
             "${meta.device}) resolve against the stored config's root.",
    )
    return p.parse_args()


def apply_data_override(cfg, data_path: str | None) -> None:
    """If ``data_path`` is given, swap ``cfg.eval_dataset`` for the loaded YAML.

    Assigning the loaded node into the stored ``cfg`` tree lets interpolations
    inside the override (``${train.window}``, ``${meta.device}``,
    ``${host_env:DATA_ROOT}``…) resolve against the stored hyperparameters,
    so the override only needs to specify the sequences themselves.
    """
    if data_path is None:
        return
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"--data file not found: {data_path!r}")
    cfg.eval_dataset = OmegaConf.load(data_path)
    print(f"[evaluate] data override: {data_path}")


def pick_checkpoint(hydra_dir: str, override: str | None) -> str:
    if override is not None:
        path = override if os.path.isabs(override) else os.path.join(hydra_dir, override)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--checkpoint not found: {path!r}")
        return path

    candidates = glob.glob(os.path.join(hydra_dir, "checkpoint_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_*.pt found under {hydra_dir!r}")

    def _step(path: str) -> int:
        m = _CKPT_RE.search(os.path.basename(path))
        if m is None:
            raise ValueError(f"Unexpected checkpoint name {path!r}; expected checkpoint_<step>.pt")
        return int(m.group(1))

    return max(candidates, key=_step)


@torch.no_grad()
def run_sequence(
    network: INetwork,
    seq: ISequence[IMUData_w_GT],
    *,
    use_aux_imu: bool,
    use_joints: bool,
    gravity: float = 9.81007,
    imu_input: str = "corrected",
) -> tuple[
    torch.Tensor,  # pred_world      (W, M, 3)  network velocity, lifted body→world
    torch.Tensor,  # gt_world        (W, M, 3)  GT velocity, world frame
    torch.Tensor,  # gt_rot_log      (W, M, 3)  GT rotation so3-log at anchors
    torch.Tensor,  # t_ns            (W, M)     int64 ns timestamps at anchors
    torch.Tensor,  # preint_vel      (W, M, 3)  preint world velocity, per-window v-reset
    torch.Tensor,  # preint_pos      (W, M, 3)  preint world position (sequence-start-anchored)
    torch.Tensor,  # preint_rot_log  (W, M, 3)  preint world rotation so3-log
]:
    """Run the network + an open-loop preintegrator over one sequence.

    Returns (W, M, …) per-window tensors so the caller can flatten for metrics
    or keep the window axis for NaN-gapped plots. Preintegration uses the selected
    stream from IMU 0, with GT pose and network velocity as the per-window initial
    state. Positions are origin-anchored to match the reported trajectories.
    """
    device = next(network.parameters()).device
    dtype  = next(network.parameters()).dtype
    # reset=True: each batch element is its own window (init_state used verbatim).
    integrator = pp.module.IMUPreintegrator(prop_cov=False, reset=True, gravity=gravity).to(device, dtype)

    loader = DataLoader[IMUData_w_GT](
        seq, batch_size=256, shuffle=False, collate_fn=collate_dataclass,
    )

    pred_chunks      : list[torch.Tensor] = []
    gt_chunks        : list[torch.Tensor] = []
    gt_rot_chunks    : list[torch.Tensor] = []
    t_chunks         : list[torch.Tensor] = []
    preint_vel_chunks: list[torch.Tensor] = []
    preint_pos_chunks: list[torch.Tensor] = []
    preint_rot_chunks: list[torch.Tensor] = []

    for data in loader:
        data        = apply_modality_mask(data, use_aux_imu=use_aux_imu, use_joints=use_joints)
        pred        = get_est_velocity(network, data)
        gt_rot_log  = network.downsample(data.gt_poses.rotation().Log())                # (B, M, 3)
        rot_anchor  = pp.so3(gt_rot_log).Exp()                                          # (B, M)
        pred_world  = rot_anchor @ pred.vel_b                                           # (B, M, 3)
        gt_world    = network.downsample(data.gt_vel_w)                                 # (B, M, 3)
        # time_ns is (B, N, K, 1); IMU 0 is canonical for the eval time axis.
        t_ns        = network.downsample(data.time_ns[:, :, 0])                         # (B, M, 1) int64

        # ------- Preintegrator on IMU 0, per-window with v-reset ---------------
        # dt from int64 ns → fp(dtype) seconds; gyro/acc of length N-1 drive
        # the N-1 integration steps (cf. ``get_ref_delta_v``).
        if imu_input == "corrected":
            acc = tp.cast(torch.Tensor, data.corrected_acc)[:, :, 0]
            gyro = tp.cast(torch.Tensor, data.corrected_gyro)[:, :, 0]
        else:
            assert imu_input == "raw"
            acc, gyro = data.acc[:, :, 0], data.gyro[:, :, 0]                           # (B, N, 3)

        t0_ns_BN = data.time_ns[:, :, 0, 0]                                             # (B, N) int64
        dt       = (t0_ns_BN[:, 1:] - t0_ns_BN[:, :-1]).to(dtype).unsqueeze(-1) * 1e-9  # (B, N-1, 1) sec

        init_pos = data.gt_poses.translation()[:, :1]                                   # (B, 1, 3) @ frame 0
        init_rot = data.gt_poses.rotation()   [:, :1]                                   # (B, 1) SO3 @ frame 0
        init_vel = pred_world                 [:, :1]                                   # (B, 1, 3) network @ anchor 0

        out = integrator(
            dt=dt, gyro=gyro[:, :-1], acc=acc[:, :-1],
            init_state={"pos": init_pos, "rot": init_rot, "vel": init_vel},
        )
        # out["pos"] / ["vel"]: (B, N-1, 3); out["rot"]: SO3 with tensor (B, N-1, 4).
        # Prepend init state at frame 0 so the time axis lines up with the N
        # input frames before ``network.downsample`` selects the M anchors.
        preint_pos_full = torch.cat([init_pos, out["pos"]], dim=1)                      # (B, N, 3)
        preint_vel_full = torch.cat([init_vel, out["vel"]], dim=1)                      # (B, N, 3)
        preint_rot_full = pp.SO3(torch.cat(
            [init_rot.tensor(), out["rot"].tensor()], dim=1,
        ))                                                                              # (B, N) SO3

        preint_pos_M     = network.downsample(preint_pos_full)                          # (B, M, 3)
        preint_vel_M     = network.downsample(preint_vel_full)                          # (B, M, 3)
        preint_rot_log_M = network.downsample(preint_rot_full.Log())                    # (B, M, 3)

        pred_chunks      .append(pred_world      .cpu())
        gt_chunks        .append(gt_world        .cpu())
        gt_rot_chunks    .append(gt_rot_log      .cpu())
        t_chunks         .append(t_ns            .cpu())
        preint_vel_chunks.append(preint_vel_M    .cpu())
        preint_pos_chunks.append(preint_pos_M    .cpu())
        preint_rot_chunks.append(preint_rot_log_M.cpu())

    pred       = torch.cat(pred_chunks,       dim=0)                                    # (W, M, 3)
    gt         = torch.cat(gt_chunks,         dim=0)
    gt_rot     = torch.cat(gt_rot_chunks,     dim=0)
    t_ns       = torch.cat(t_chunks,          dim=0).squeeze(-1)                        # (W, M)
    preint_vel = torch.cat(preint_vel_chunks, dim=0)
    preint_pos = torch.cat(preint_pos_chunks, dim=0)
    preint_rot = torch.cat(preint_rot_chunks, dim=0)
    # Origin-anchor: subtract sequence-start absolute position so preint_pos
    # lives in the same frame as the trapezoidal pred_pos / gt_pos (both 0 at
    # t=0). Subsequent windows reset to GT — they stay in the same frame.
    preint_pos = preint_pos - preint_pos[0:1, 0:1, :]
    return pred, gt, gt_rot, t_ns, preint_vel, preint_pos, preint_rot


def compute_relative_velocity_trend(
    pred_vel_world: torch.Tensor,
    gt_vel_world:   torch.Tensor,
    timestamps:     torch.Tensor,
    horizons_sec:   list[float] | tuple[float, ...],
) -> dict[str, float]:
    """Relative Velocity Trend Error over multiple horizons (offset-invariant).

    Per horizon h on dv(t,h) = vel[t+h] - vel[t]:
        RVTE_<h>      = sqrt(mean ‖dv_pred - dv_gt‖²)
        nRVTE_<h>     = RVTE / sqrt(mean ‖dv_gt‖²)        (scale-free)

    Plus aggregate mean_nRVTE over fitting horizons. Horizons longer than the
    sequence are skipped.
    """
    N = pred_vel_world.shape[0]
    pred = pred_vel_world.detach().cpu().double()
    gt   = gt_vel_world  .detach().cpu().double()
    ts   = timestamps    .detach().cpu().double()

    mean_dt = float((ts[-1] - ts[0]) / (N - 1))

    eps = 1e-8
    metrics: dict[str, float] = {"trend_mean_dt_sec": mean_dt}
    nrvte_list: list[float] = []

    for h_sec in horizons_sec:
        h_frames = round(h_sec / mean_dt)
        label = (
            f"{int(h_sec)}s"
            if float(h_sec).is_integer()
            else f"{h_sec:g}s".replace(".", "p")
        )
        if h_frames >= N:
            print(
                f"[evaluate] WARNING: RVTE horizon {h_sec}s ({h_frames} frames) "
                f"exceeds sequence length N={N}; reporting NaN",
                flush=True,
            )
            metrics[f"RVTE_{label}"] = float("nan")
            metrics[f"nRVTE_{label}"] = float("nan")
            continue

        dv_pred  = pred[h_frames:] - pred[:-h_frames]   # (N - h, 3)
        dv_gt    = gt  [h_frames:] - gt  [:-h_frames]
        residual = dv_pred - dv_gt

        rvte   = float(torch.sqrt((residual ** 2).mean()))
        gt_rms = float(torch.sqrt((dv_gt    ** 2).mean()))
        nrvte  = rvte / (gt_rms + eps)

        metrics[f"RVTE_{label}"]      = rvte
        metrics[f"nRVTE_{label}"]     = nrvte

        nrvte_list.append(nrvte)

    if nrvte_list:
        metrics["mean_nRVTE"]     = float(np.mean(nrvte_list))

    return metrics


def compute_metrics(
    pred:     torch.Tensor,    # (T, 3) world-frame predicted velocity
    gt:       torch.Tensor,    # (T, 3) world-frame GT velocity
    pred_pos: torch.Tensor,    # (T, 3) integrated predicted position, origin-anchored
    gt_pos:   torch.Tensor,    # (T, 3) integrated GT position,        origin-anchored
    t_ns:     torch.Tensor,    # (T,)   int64 nanosecond timestamps
) -> dict[str, float]:
    """AVE + ATE + non-overlapping ΔV(T) + RVTE.

    ΔV windowing is driven by timestamp (anchors aren't strictly uniform: tail
    padding + cross-window gaps), walking the cursor forward by ≥T seconds.
    ATE is on origin-anchored positions (no SE3/Umeyama) → cumulative drift.
    """
    T = pred.shape[0]
    assert pred.ndim == 2 and pred.shape[1] == 3
    assert gt.shape == pred.shape == pred_pos.shape == gt_pos.shape
    assert t_ns.ndim == 1 and len(t_ns) == T and T >= 2
    assert bool(torch.all(t_ns[1:] > t_ns[:-1]))

    ave = (pred - gt).norm(dim=-1).mean().item()

    pos_err = (pred_pos - gt_pos).norm(dim=-1)
    ate      = float(pos_err.pow(2).mean().sqrt().item())   # RMSE of position-error norms
    ate_mean = float(pos_err.mean().item())                 # mean L2 position error

    # fp64 in seconds, anchored at sequence start — int64 ns directly to fp64
    # avoids the precision loss of an fp32 cast on ~1e14-magnitude TAI ns values.
    t_s = (t_ns - t_ns[0]).to(torch.float64) * 1.0e-9
    duration = t_s[-1].item()

    out: dict[str, float] = {"AVE": ave, "ATE": ate, "ATE_mean": ate_mean}
    for window_s in _DV_WINDOWS_S:
        if duration < window_s:
            # Short clips can't support the longer ΔV horizons (e.g. a single
            # 5 s window vs ΔV(10 s)); report NaN instead of aborting the whole
            # eval — NaNs are dropped from the cross-sequence aggregation.
            print(f"[evaluate] WARN: sequence too short for ΔV at {window_s} s "
                  f"(duration {duration:.2f} s); reporting NaN.")
            out[f"dv_err_{window_s:g}s"] = float("nan")
            continue

        idx_start: list[int] = []
        idx_end:   list[int] = []
        cursor = 0
        while cursor < T:
            target = t_s[cursor].item() + window_s
            # Smallest index strictly past cursor whose timestamp ≥ target.
            # ``searchsorted`` with side='left' returns that boundary directly.
            j = int(torch.searchsorted(t_s, torch.tensor(target, dtype=t_s.dtype)).item())
            if j >= T:
                break  # tail does not fit a full window → drop
            idx_start.append(cursor)
            idx_end.append(j)
            cursor = j  # non-overlapping: next window starts where this one ended

        s = torch.tensor(idx_start, dtype=torch.long)
        e = torch.tensor(idx_end,   dtype=torch.long)
        dv_pred = pred[e] - pred[s]
        dv_gt   = gt  [e] - gt  [s]
        out[f"dv_err_{int(window_s)}s"] = (dv_pred - dv_gt).norm(dim=-1).mean().item()

    # RVTE: overlapping, GT-normalized Δv trend error (offset-/PGO-invariant).
    out.update(compute_relative_velocity_trend(
        pred_vel_world=pred,
        gt_vel_world  =gt,
        timestamps    =t_s,
        horizons_sec  =_RVTE_HORIZONS_S,
    ))

    return out




def flatten_seq_name_for_filename(seq_name: str) -> str:
    """Make ``seq_name`` safe as a single path component (e.g. BlackBird ``a/b/c``).

    Replaces ``/`` with ``_`` so ``savefig`` does not interpret hierarchy inside
    the filename stem.
    """
    return seq_name.replace("/", "_")


def plot_per_axis(
    seq_name: str,
    quantity: str,                    # e.g. "velocity" / "translation" / "rotation"
    unit: str,                        # e.g. "m/s" / "m" / "rad"
    gt_w:     torch.Tensor,           # (W, M, 3)
    pred_w:   torch.Tensor | None,    # (W, M, 3) or None when there is no network prediction
    preint_w: torch.Tensor | None,    # (W, M, 3) or None when there is no preint curve
    t_w_s:    torch.Tensor,           # (W, M) fp64 seconds, anchored at sequence start
    save_path: str,
) -> None:
    """Stack of three time-series subplots (x / y / z) with per-window NaN gaps.

    A NaN sentinel is appended to each window so matplotlib breaks the line
    between successive windows instead of connecting the last anchor of
    window ``w`` to the first anchor of window ``w+1`` — matches the eval
    plot style in ``train_ssio.evaluate``. Window starts are also
    marked with light vertical lines for orientation.

    ``pred_w=None`` plots GT (and preint, if present) only — used for the
    rotation plot, where the SSIO network consumes GT rotation as input
    rather than predicting it.
    """
    W = gt_w.shape[0]

    def _with_break(arr: torch.Tensor) -> np.ndarray:                # (W, M, C) -> ((M+1)·W, C)
        pad = torch.full_like(arr[:, :1], float("nan"))
        return torch.cat([arr, pad], dim=1).reshape(-1, arr.shape[-1]).numpy()

    t_pad  = torch.full((W, 1), float("nan"), dtype=t_w_s.dtype)
    t_flat = torch.cat([t_w_s, t_pad], dim=1).reshape(-1).numpy()
    gt_flat     = _with_break(gt_w)
    pred_flat   = _with_break(pred_w)   if pred_w   is not None else None
    preint_flat = _with_break(preint_w) if preint_w is not None else None
    window_edges = t_w_s[:, 0].numpy()                              # one tick per window start

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i, axis in enumerate(("x", "y", "z")):
        axes[i].plot(t_flat, gt_flat[:, i], label="GT", linewidth=1.0)
        if pred_flat is not None:
            axes[i].plot(t_flat, pred_flat[:, i],   label="pred",   linewidth=1.0, alpha=0.85)
        if preint_flat is not None:
            axes[i].plot(t_flat, preint_flat[:, i], label="preint", linewidth=1.0, alpha=0.8)
        for edge in window_edges:
            axes[i].axvline(edge, color="0.7", linewidth=0.5, alpha=0.5, zorder=0)
        axes[i].set_ylabel(f"{axis} ({unit})")
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc="best")
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"Sequence {seq_name} — per-axis {quantity}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def summarize(per_seq: list[dict]) -> dict:
    metric_keys = sorted({
        k
        for row in per_seq
        for k, value in row.items()
        if k != "name" and isinstance(value, (int, float))
    })
    out: dict = {"num_sequences": len(per_seq)}
    for k in metric_keys:
        vals = np.array([
            row.get(k, float("nan"))
            if isinstance(row.get(k), (int, float))
            else float("nan")
            for row in per_seq
        ], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            vals = np.array([float("nan")])
        out[k] = {
            "mean":   float(vals.mean()),
            "median": float(np.median(vals)),
            "std":    float(vals.std(ddof=0)),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
        }
    return out


def main() -> None:
    args = parse_args()
    hydra_dir = os.path.abspath(args.hydra_output_dir)
    if not os.path.isdir(hydra_dir):
        raise NotADirectoryError(f"Not a directory: {hydra_dir!r}")

    cfg_path = os.path.join(hydra_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f".hydra/config.yaml not found under {hydra_dir!r}")

    # Don't eager-resolve: the as_device/as_dtype resolvers return torch
    # device/dtype objects OmegaConf can't store. On-demand access resolves fine.
    cfg = OmegaConf.load(cfg_path)
    apply_data_override(cfg, args.data)

    ckpt_path = pick_checkpoint(hydra_dir, args.checkpoint)
    print(f"[evaluate] hydra dir : {hydra_dir}")
    print(f"[evaluate] checkpoint: {os.path.basename(ckpt_path)}")

    network = instantiate(INetwork, cfg.network).to(cfg.meta.device, cfg.meta.dtype)
    network.load_state_dict(torch.load(ckpt_path, map_location=cfg.meta.device))
    network.eval()

    # `select` with a default, not attribute access: evaluate.py scores run dirs saved
    # before the modality block existed, whose stored config has no `modality` node.
    use_aux_imu = bool(OmegaConf.select(cfg, "modality.use_aux_imu", default=True))
    use_joints  = bool(OmegaConf.select(cfg, "modality.use_joints",  default=True))

    eval_sequences = [
        instantiate(ISequence[IMUData_w_GT], seq_cfg)
        for seq_cfg in cfg.eval_dataset.sequences
    ]

    out_dir = os.path.join(hydra_dir, "eval", os.path.splitext(os.path.basename(ckpt_path))[0])
    os.makedirs(out_dir, exist_ok=True)

    results: list[dict] = []
    for idx, seq in enumerate(eval_sequences):
        # ISequence.seq_name — loaders return their configured name, wrappers
        # forward it, so any wrapper nesting (e.g. WindowIntegratedOrientation) works.
        seq_name = seq.seq_name
        print(f"[evaluate] [{idx + 1:>3d}/{len(eval_sequences)}] {seq_name} …", flush=True)

        pred_w, gt_w, gt_rot_w, t_ns_w, preint_vel_w, preint_pos_w, preint_rot_w = run_sequence(
            network, seq, use_aux_imu=use_aux_imu, use_joints=use_joints,
            gravity=float(OmegaConf.select(cfg, "train.gravity", default=9.81007)),
            imu_input=str(OmegaConf.select(cfg, "train.dv_imu_input", default="corrected")),
        )
        W, M, _   = pred_w.shape
        pred      = pred_w  .reshape(-1, 3)
        gt        = gt_w    .reshape(-1, 3)
        t_ns      = t_ns_w  .reshape(-1)
        pred_pos  = integrate_trapezoid(pred, t_ns)                                 # (T, 3) origin-anchored
        gt_pos    = integrate_trapezoid(gt,   t_ns)
        metrics = compute_metrics(pred, gt, pred_pos, gt_pos, t_ns)
        metrics["name"] = seq_name
        results.append(metrics)

        # Reshape continuous trajectories back to (W, M, 3) so the per-axis
        # plots can apply the same NaN-gap treatment to every curve.
        pred_pos_w = pred_pos.reshape(W, M, 3)
        gt_pos_w   = gt_pos  .reshape(W, M, 3)
        t_s_w      = (t_ns_w - t_ns_w[0, 0]).to(torch.float64) * 1.0e-9

        stem = flatten_seq_name_for_filename(seq_name)

        # Trajectory overlay (GT vs pred, both origin-anchored), XY and XZ. XZ is
        # where vertical drift shows; a top-down view alone hides it.
        gt_np, pred_np = gt_pos.numpy(), pred_pos.numpy()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, (h, v, h_name, v_name) in zip(axes, ((0, 1, "x", "y"), (0, 2, "x", "z"))):
            ax.plot(gt_np  [:, h], gt_np  [:, v], label="GT",   linewidth=1.0)
            ax.plot(pred_np[:, h], pred_np[:, v], label="pred", linewidth=1.0, alpha=0.85)
            ax.scatter([0], [0], c="k", marker="o", s=20, zorder=3, label="start")
            ax.set_xlabel(f"{h_name} (m)")
            ax.set_ylabel(f"{v_name} (m)")
            ax.set_title(f"{h_name}{v_name} plane")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3)
        axes[0].legend(loc="best")
        fig.suptitle(f"Sequence {seq_name} — trajectory")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"seq_{stem}_traj.png"), dpi=120)
        plt.close(fig)

        plot_per_axis(seq_name, "velocity",    "m/s", gt_w,     pred_w,     preint_vel_w, t_s_w,
                      os.path.join(out_dir, f"seq_{stem}_vel.png"))
        plot_per_axis(seq_name, "translation", "m",   gt_pos_w, pred_pos_w, preint_pos_w, t_s_w,
                      os.path.join(out_dir, f"seq_{stem}_trans.png"))
        plot_per_axis(seq_name, "rotation",    "rad", gt_rot_w, None,       preint_rot_w, t_s_w,
                      os.path.join(out_dir, f"seq_{stem}_rot.png"))

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(results, f, indent=4)
    summary = summarize(results)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[evaluate] wrote {out_dir} (result.json, summary.json, "
          f"{len(results)} trajectory plots)")
    for k in [k for k in results[0] if k != "name"]:
        s = summary[k]
        print(f"           {k:<14} mean={s['mean']:.5f}  median={s['median']:.5f}  "
              f"std={s['std']:.5f}  min={s['min']:.5f}  max={s['max']:.5f}")


if __name__ == "__main__":
    main()
