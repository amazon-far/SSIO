import typing as tp

import wandb
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import pypose as pp
import jaxtyping as jt
import matplotlib
matplotlib.use("Agg")  # headless backend — avoids Tcl_AsyncDelete race with wandb's upload thread
import matplotlib.pyplot as plt
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from Interface             import instantiate, validate_hydra_config
from Interface.sequence    import ISequence
from Interface.data        import IMUData_w_GT, collate_dataclass, move_dataclass
from Interface.model       import INetwork, get_est_velocity
from Interface.loss        import I_SSIO_Loss
from Loss.regularization             import RegularizationLoss
from Sequence.ZeroVelocity  import attach_stationary_mask
from Utils.checkpoint      import load_resume_checkpoint, save_best_model, save_checkpoint_artifact
from Utils.config          import SSIO_CONFIG_SPEC
from Utils.lr_schedule     import scheduler_step_epoch, regularization_schedule_weight
from Utils.modality        import apply_modality_mask
from Utils.plots           import plot_trajectory_2d, plot_velocity_series
from Utils.trainer         import metric_ATE, metric_AVE, metric_vel_dc_bias, optimize_step


@main(version_base=None, config_path="./Config", config_name="train_ssio")
@validate_hydra_config(SSIO_CONFIG_SPEC)
def main_func(cfg):
    wandb.init(
        project = cfg.wandb.project,
        mode    = cfg.wandb.mode,
        group   = "ssio",
        name    = cfg.wandb.name,
        config  = OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        dir     = HydraConfig.get().runtime.output_dir,       # → <hydra_dir>/wandb/run-…
    )

    use_aux_imu = cfg.modality.use_aux_imu
    use_joints  = cfg.modality.use_joints

    regularization_weight = cfg.train.regularization_weight
    regularization = (
        RegularizationLoss(cfg.train.regularization_gate, cfg.train.regularization_speed_max)
        if regularization_weight > 0.0 else None
    )

    # ----- Instantiate sequences -----------------------------------------------
    train_sequences = [
        instantiate(ISequence[IMUData_w_GT], seq_cfg)
        for seq_cfg in cfg.train_dataset.sequences
    ]
    eval_sequences = [
        instantiate(ISequence[IMUData_w_GT], seq_cfg)
        for seq_cfg in cfg.eval_dataset.sequences
    ]

    if regularization is not None and regularization.gate == "imu":
        train_sequences = [attach_stationary_mask(seq) for seq in train_sequences]
        print(f"[regularization] ZeroVelocity mask attached to {len(train_sequences)} train sequences", flush=True)

    # ----- Loaders -------------------------------------------------------------
    loss_fn        = instantiate(I_SSIO_Loss, cfg.loss)
    # >0 needs CPU-resident train sequences (DataLoader
    # workers fork/spawn and must not touch device tensors); batches are moved
    # to meta.device in the epoch loop, which is a per-field no-op at 0 workers
    # with device-resident datasets.
    num_workers = cfg.train.num_workers
    worker_kwargs: dict = dict(num_workers=num_workers, persistent_workers=num_workers > 0)

    # Optional train-pool subsample (seed=0, nested across ratios); eval pool unchanged.
    train_pool: Dataset = ConcatDataset(train_sequences)
    data_ratio = cfg.train.data_ratio
    if data_ratio < 1.0:
        total  = len(train_pool)  # type: ignore[arg-type]
        n_keep = max(1, round(data_ratio * total))
        gen    = torch.Generator().manual_seed(0)
        kept   = sorted(torch.randperm(total, generator=gen).tolist()[:n_keep])
        train_pool = Subset(train_pool, kept)
        print(f"[data_ratio={data_ratio}] keeping {n_keep}/{total} train windows", flush=True)

    train_dataset = DataLoader[IMUData_w_GT](
        train_pool,
        batch_size=cfg.train.batch, shuffle=True, pin_memory=True, collate_fn=collate_dataclass,
        **worker_kwargs,
    )

    network    = instantiate(INetwork, cfg.network).train().to(cfg.meta.device, cfg.meta.dtype)
    resume_from = cfg.train.resume_from
    if resume_from:
        load_resume_checkpoint(resume_from, network, cfg.meta.device)
    optimizer  = instantiate(torch.optim.Optimizer, cfg.optimizer, params=network.parameters())
    scheduler  = instantiate(torch.optim.lr_scheduler.LRScheduler, cfg.scheduler, optimizer=optimizer)
    integrator = pp.module.IMUPreintegrator(prop_cov=False, reset=True, gravity=cfg.train.gravity).to(cfg.meta.device, cfg.meta.dtype)

    # EMA shadow; evaluated/checkpointed when on. decay=0.8 is deliberate (~5-step
    # average); get_ema_multi_avg_fn's own default is 0.999.
    ema_decay = 0.8
    ema_model: AveragedModel | None = (
        AveragedModel(network, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
        if cfg.train.ema else None
    )
    eval_model = ema_model.module if ema_model is not None else network

    checkpoint_every_epochs = cfg.train.checkpoint_every_epochs

    # Δv/preintegration stream, independent of the network's own imu_input.
    dv_imu_input = cfg.train.dv_imu_input

    # Figure cadence, independent of the metric cadence. 1 = every epoch; metrics
    # are unaffected either way.
    eval_plot_every_epochs = cfg.train.eval_plot_every_epochs
    max_train_steps = cfg.train.max_train_steps

    try:
        step = 0
        stop = False

        init_metrics = evaluate(cfg, step, loss_fn, eval_model, eval_sequences)
        best_ave  = init_metrics["eval/AVE"]
        best_loss = init_metrics["eval/loss"]
        save_best_model("best_ave_model.pt", eval_model, "best_eval_AVE",  best_ave)
        save_best_model("best_model.pt",     eval_model, "best_eval_loss", best_loss)

        for epoch in range(cfg.train.epoch):
            regularization_s = regularization_schedule_weight(
                epoch,
                cfg.train.epoch,
                start=cfg.train.regularization_sched_start,
                end=cfg.train.regularization_sched_end,
                enable=cfg.train.regularization_sched_enable,
            )
            regularization_weight_eff = regularization_weight * regularization_s
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(f"{epoch:03d}", total=len(train_dataset))
                for data in train_dataset:
                    step += 1
                    data = move_dataclass(data, cfg.meta.device)
                    data = apply_modality_mask(data, use_aux_imu=use_aux_imu, use_joints=use_joints)
                    loss, metrics = train_step_single(
                        network, integrator, loss_fn, data=data,
                        regularization=regularization, regularization_weight=regularization_weight_eff,
                        dv_imu_input=dv_imu_input,
                    )
                    loss.backward()
                    optimize_step(optimizer, ema_model, network)
                    if step % 10 == 0: wandb.log(metrics, step=step)
                    progress.update(task, advance=1)
                    if max_train_steps is not None and step >= max_train_steps:
                        stop = True
                        break

            # Metrics every epoch (the scheduler needs eval/loss); figures only
            # every eval_plot_every_epochs, and always on the final epoch.
            epoch_next = epoch + 1
            want_plots = (
                epoch_next % eval_plot_every_epochs == 0
                or epoch_next == cfg.train.epoch
                or stop
            )
            eval_metrics = evaluate(
                cfg, step, loss_fn, eval_model, eval_sequences, plots=want_plots,
            )
            val_ave = eval_metrics["eval/AVE"]
            if val_ave < best_ave:
                best_ave = val_ave
                save_best_model("best_ave_model.pt", eval_model, "best_eval_AVE", val_ave)
            val_loss = eval_metrics["eval/loss"]
            if val_loss < best_loss:
                best_loss = val_loss
                save_best_model("best_model.pt", eval_model, "best_eval_loss", val_loss)
            scheduler_step_epoch(scheduler, val_loss)
            wandb.log({"train/lr": optimizer.param_groups[0]["lr"]}, step=step)
            if regularization is not None:
                wandb.log({"train/regularization_sched_s": regularization_s, "train/regularization_weight_eff": regularization_weight_eff}, step=step)
            epoch_done = epoch + 1
            if epoch_done % checkpoint_every_epochs == 0 or epoch_done == cfg.train.epoch or stop:
                save_checkpoint_artifact(step, eval_model)
            if stop:
                break
    finally:
        wandb.finish()


def get_ref_delta_v(
    model: pp.module.IMUPreintegrator,
    data : IMUData_w_GT,
    *,
    start_index: int,
    imu_input  : str = "corrected",
) -> tuple[
    jt.Float[torch.Tensor, "B N 3"],
    jt.Float[torch.Tensor, "B N 3 3"],
]:
    """Cumulative Δv and its covariance per anchor, in the anchor-0 body frame;
    index 0 zero-padded.

    GT rotation (fed as ``rot=``) makes pypose remove gravity per sample, so Dv
    is body-frame and gravity-free. Cov is propagated analytically (pypose does
    not), treating GT rotation as deterministic so only ``acc_cov`` propagates:

        Σ_{i,k} = Σ_{j<k} R_{ij} C_a[j] R_{ij}^T Δt_j²,   R_{ij} = R_wb[i]^T R_wb[j]

    Preintegration starts at input frame 0, but the losses rotate Δv with the
    rotation at *output anchor 0* = input frame ``start_index``. Δv is therefore
    re-expressed in that frame here; skipping it leaves a residual term
    (R_0 - R_start) @ Δv that no network output can fit and that does not cancel
    under differencing (measured on EuRoC: 0.9-3.4% of |Δv|, linear in the
    angular rate).

    Main-body IMU (K index 0) only, as in ``evaluate.py``'s preint baseline and
    ``Sequence.WindowIntegratedOrientation``. K>1 is legal and is **indexed, not
    fused**: IMUs at different mount points measure different specific force
    (a_k = a_ref + α×r_k + ω×(ω×r_k)), so a mean over K estimates no physical point's
    Δv and cov/K claims independence co-located sensors do not have. Aux IMUs are
    therefore excluded from the Δv reference even when the network consumes them
    (``modality.use_aux_imu``).
    """
    # The Δv reference has its OWN stream key, deliberately not the network's
    # `imu_input`: one flag governing both would move the network input and the
    # self-supervision label together.

    if imu_input == "corrected":
        acc = tp.cast(torch.Tensor, data.corrected_acc)[:, :, 0]
        gyro = tp.cast(torch.Tensor, data.corrected_gyro)[:, :, 0]
    else:
        assert imu_input == "raw"
        acc, gyro = data.acc[:, :, 0], data.gyro[:, :, 0]                             # (B, N, 3)

    time_ns = data.time_ns       [:, :, 0]                                            # (B, N, 1) int64
    acc_cov = data.acc_cov       [:, :, 0]                                            # (B, N, 3, 3)
    rot_wb  = data.gt_poses.rotation()                                                # SO3 [B, N], body-level (no K)

    dt     = (time_ns[:, 1:] - time_ns[:, :-1]).to(acc.dtype) * 1e-9
    preint = model.integrate(
        dt   = dt,
        gyro = gyro  [:, :-1],
        acc  = acc   [:, :-1],
        rot  = rot_wb[:, :-1],
    )
    Dv = torch.cat([torch.zeros_like(preint["Dv"][:, :1]), preint["Dv"]], dim=1)

    # Per-step Δv-cov in body_i frame, then cumulative sum (index 0 zero-padded).
    R_ij     = (rot_wb[:, :1].Inv() @ rot_wb[:, :-1]).matrix()
    step_cov = R_ij @ acc_cov[:, :-1] @ R_ij.transpose(-1, -2) * dt.pow(2).unsqueeze(-1)
    cum_cov  = torch.cumsum(step_cov, dim=1)
    Dv_cov   = torch.cat([torch.zeros_like(cum_cov[:, :1]), cum_cov], dim=1)

    # Re-express from the frame-0 body frame into the anchor-0 body frame, so
    # ``R[:, 0:1] @ ref_delta_v`` in the losses is exact rather than a proxy.
    R_align = rot_wb[:, start_index:start_index + 1].Inv() @ rot_wb[:, 0:1]            # SO3 [B, 1]
    A       = R_align.matrix()
    Dv      = R_align @ Dv
    Dv_cov  = A @ Dv_cov @ A.transpose(-1, -2)


    return Dv, Dv_cov


# ====== Per-step training helpers ======


def train_step_single(
    network    : INetwork,
    integrator : pp.module.IMUPreintegrator,
    loss_fn    : I_SSIO_Loss,
    *,
    data         : IMUData_w_GT,
    regularization         : RegularizationLoss | None = None,
    regularization_weight  : float = 0.0,
    dv_imu_input : str = "corrected",
) -> tuple[torch.Tensor, dict[str, float]]:
    """One training step: the loss against the integration teacher (or, for the
    supervised baseline, against GT velocity), plus the optional stationary term.
    """
    network.train()  # sole owner of the mode flag — evaluate() leaves the net in eval
    ref_delta_v, ref_delta_v_cov = get_ref_delta_v(
        integrator, data, start_index=network.start_index, imu_input=dv_imu_input)
    ref_delta_v     = network.downsample(ref_delta_v)
    # downsample takes (*B, N, C); flatten the trailing 3x3 block to C=9 to
    # downsample, then unflatten back to (*B, M, 3, 3).
    B_, N_, _, _    = ref_delta_v_cov.shape
    ref_delta_v_cov = network.downsample(ref_delta_v_cov.reshape(B_, N_, 9)).reshape(B_, -1, 3, 3)

    ref_rot_Twb = pp.so3(network.downsample(data.gt_poses.rotation().Log()))
    pred        = get_est_velocity(network, data)

    gt_vel_w_anchor = network.downsample(data.gt_vel_w)
    gt_vel_b_anchor = ref_rot_Twb.Exp().Inv() @ gt_vel_w_anchor

    loss, metrics = loss_fn(
        ref_rot_Twb, ref_delta_v, ref_delta_v_cov,
        pred.vel_b, pred.vel_cov_b, gt_vel_b_anchor,
    )
    metrics["train/AVE"] = metric_AVE(
        ref_rot_Twb, pred.vel_b, gt_vel_w_anchor,
    )
    metrics["train/vel_dc_bias"] = metric_vel_dc_bias(
        ref_rot_Twb, pred.vel_b, gt_vel_w_anchor,
    )
    if regularization is not None and regularization_weight > 0.0:
        regularization_loss, regularization_metrics = regularization(
            pred.vel_b, regularization.mask(network, data, gt_vel_b_anchor), regularization_weight,
        )
        loss = loss + regularization_loss
        metrics.update(regularization_metrics)
        metrics["train/loss"] = loss.detach().item()   # keep train/loss == what is optimized
    return loss, metrics


@torch.no_grad()
def evaluate(
    cfg,
    step: int,
    loss: I_SSIO_Loss,
    model: INetwork,
    eval_sequences: list[ISequence[IMUData_w_GT]],
    plots: bool = True,
) -> dict[str, float]:
    """Eval all sequences; log aggregate eval/* and per-sequence eval_seq/* metrics,
    plus (when ``plots``) per-sequence velocity-series and 2D-trajectory (XY + XZ)
    figures.

    Assumes non-overlapping windows (step_size == window_size), so windows concat
    directly.

    ``plots`` is separate from the metric path on purpose: metrics are needed every
    epoch (the LR scheduler consumes ``eval/loss``), whereas figures are 2 images ×
    N sequences per call, which dominates upload time at long schedules.
    """
    dv_imu_input = cfg.train.dv_imu_input
    device     = next(model.parameters()).device
    dtype      = next(model.parameters()).dtype
    integrator = pp.module.IMUPreintegrator(prop_cov=False, reset=True, gravity=cfg.train.gravity).to(device, dtype)
    use_aux_imu = cfg.modality.use_aux_imu
    use_joints  = cfg.modality.use_joints

    model.eval()
    # Keys the loss *may* emit, aggregated only when present (see the `if k not in
    # metrics` skip below and the `if v` guards at the two aggregation sites):
    #   - train/loss, train/AVE, train/vel_dc_bias — always there (AVE and
    #     vel_dc_bias are computed here, not by the loss).
    #   - train/L2norm_vel_dv_centered / train/L2norm_vel_dv — norms of the residual
    #     against the *preintegration Δv* reference, mean-centred per window and not.
    #     Both are absent on the supervised baseline (SupervisedVelocityLoss), which has no
    #     Δv reference at all — hence the `if v` guards at the aggregation sites.
    # Neither L2norm is GT-referenced: only train/AVE and train/vel_dc_bias are.
    agg: dict[str, list[float]] = {
        "train/loss"                   : [],
        "train/L2norm_vel_dv_centered" : [],
        "train/L2norm_vel_dv"          : [],
        "train/AVE"                    : [],
        "train/vel_dc_bias"            : [],
    }
    # ATE is one number per *sequence* (a cumulative integral over the whole
    # timeline), not per batch, so it cannot live in `agg` / `seq_agg`; collected
    # here and averaged over sequences at the end.
    ate_agg     : list[float] = []
    ate_mean_agg: list[float] = []

    for idx, seq in enumerate(eval_sequences):
        loader = DataLoader[IMUData_w_GT](
            seq, batch_size=256, shuffle=False, collate_fn=collate_dataclass,
        )

        # Per-sequence copy of the same metrics: the global `agg` means over every
        # batch of every sequence, which hides which sequence is dragging the
        # average. Both are kept — aggregate for the headline, per-sequence for
        # attribution.
        seq_agg: dict[str, list[float]] = {k: [] for k in agg}

        pred_chunks  : list[torch.Tensor] = []
        gt_chunks    : list[torch.Tensor] = []
        t_chunks     : list[torch.Tensor] = []
        std_chunks   : list[torch.Tensor] = []
        preint_chunks: list[torch.Tensor] = []

        for data in loader:
            data = apply_modality_mask(data, use_aux_imu=use_aux_imu, use_joints=use_joints)
            ref_delta_v, ref_delta_v_cov = get_ref_delta_v(
                integrator, data, start_index=model.start_index, imu_input=dv_imu_input)
            ref_delta_v     = model.downsample(ref_delta_v)
            B_, N_, _, _    = ref_delta_v_cov.shape
            ref_delta_v_cov = model.downsample(ref_delta_v_cov.reshape(B_, N_, 9)).reshape(B_, -1, 3, 3)

            ref_rot_Twb = pp.so3(model.downsample(data.gt_poses.rotation().Log()))
            pred        = get_est_velocity(model, data)

            gt_vel_w_M      = model.downsample(data.gt_vel_w)
            gt_vel_b_anchor = ref_rot_Twb.Exp().Inv() @ gt_vel_w_M

            _, metrics = loss(
                ref_rot_Twb, ref_delta_v, ref_delta_v_cov,
                pred.vel_b, pred.vel_cov_b, gt_vel_b_anchor,
            )
            metrics["train/AVE"] = metric_AVE(
                ref_rot_Twb, pred.vel_b, gt_vel_w_M,
            )
            metrics["train/vel_dc_bias"] = metric_vel_dc_bias(
                ref_rot_Twb, pred.vel_b, gt_vel_w_M,
            )
            for k in agg:
                if k not in metrics:
                    continue          # e.g. the supervised path emits no Δv residual
                agg[k].append(metrics[k])
                seq_agg[k].append(metrics[k])

            # World-frame velocities (prediction lifted via per-anchor GT R_wb;
            # raw, not mean-aligned).
            R          = ref_rot_Twb.Exp()                                 # SO3 [B, M]
            pred_world = R @ pred.vel_b                                       # (B, M, 3)
            t_anchor   = model.downsample(data.time_ns[:, :, 0])            # IMU 0; int64 — do not cast to fp32

            # Preint velocity: v0 from pred @ anchor 0, then v_w = v0 + R0 @ Δv.
            R0           = R[:, :1]
            v0_world     = pred_world[:, :1]
            preint_world = v0_world + (R0 @ ref_delta_v)

            # Uncertainty shade: rotate body-frame cov to world (R Σ Rᵀ), take √diag.
            R_mat   = R.matrix()
            cov_w   = R_mat @ pred.vel_cov_b @ R_mat.transpose(-1, -2)
            std_w   = torch.diagonal(cov_w, dim1=-2, dim2=-1).clamp_min(0).sqrt()

            pred_chunks  .append(pred_world.cpu())
            gt_chunks    .append(gt_vel_w_M.cpu())
            t_chunks     .append(t_anchor.cpu())
            std_chunks   .append(std_w.cpu())
            preint_chunks.append(preint_world.cpu())

        # World-frame velocity per anchor — a time series, not a spatial path; the
        # *_plot copies below carry a NaN row per window so matplotlib breaks the
        # line between windows rather than drawing a jump.
        pred_vel_w   = torch.cat(pred_chunks,   dim=0)                            # (W, M, 3)
        gt_vel_w     = torch.cat(gt_chunks,     dim=0)
        std_vel_w    = torch.cat(std_chunks,    dim=0)
        preint_vel_w = torch.cat(preint_chunks, dim=0)
        W, window_M, _ = pred_vel_w.shape
        t_2d_ns = torch.cat(t_chunks, dim=0).reshape(W, window_M)              # (W, M) int64
        t0_ns   = t_2d_ns[0, 0]

        def _with_break(arr):  # (W, M, C) -> (W*(M+1), C) with NaN row per window
            pad = torch.full_like(arr[:, :1], float("nan"))
            return torch.cat([arr, pad], dim=1).reshape(-1, arr.shape[-1])

        pred_vel_w_plot   = _with_break(pred_vel_w  ).numpy()
        gt_vel_w_plot     = _with_break(gt_vel_w    ).numpy()
        std_vel_w_plot    = _with_break(std_vel_w   ).numpy()
        preint_vel_w_plot = _with_break(preint_vel_w).numpy()

        # Subtract t0 in int64, then fp64→seconds (cf. evaluate.py / train_pgd).
        # Casting TAI-scale ns to fp32 before subtract collapses the time axis.
        t_pad    = torch.full((W, 1), float("nan"), dtype=torch.float64)
        t_plot_s = (
            (torch.cat([t_2d_ns.to(torch.float64), t_pad], dim=1).reshape(-1) - t0_ns.to(torch.float64))
            * 1e-9
        ).numpy()
        window_edges = ((t_2d_ns[:, 0] - t0_ns).to(torch.float64) * 1e-9).numpy()

        # ----- Per-sequence scalars — every eval, unconditionally --------------
        seq_label = seq.seq_name
        # Unbroken (W, M, ·) series flattened to (T, ·) — never the *_plot copies,
        # whose NaN separators would poison the cumulative integral.
        ate, ate_mean = metric_ATE(
            pred_vel_w.reshape(-1, 3), gt_vel_w.reshape(-1, 3), t_2d_ns.reshape(-1),
        )
        ate_agg     .append(ate)
        ate_mean_agg.append(ate_mean)

        seq_scalars = {f"eval_seq/{k.split('/', 1)[1]}/{seq_label}": sum(v) / len(v)
                       for k, v in seq_agg.items() if v}
        seq_scalars[f"eval_seq/ATE/{seq_label}"]      = ate
        seq_scalars[f"eval_seq/ATE_mean/{seq_label}"] = ate_mean
        wandb.log(seq_scalars, step=step)

        # ----- Figures — gated; see `plots` -----------------------------------
        if not plots:
            continue

        fig = plot_velocity_series(
            t_plot_s, gt_vel_w_plot, pred_vel_w_plot, preint_vel_w_plot,
            std_vel_w_plot, window_edges,
            title=f"Eval sequence {seq_label} @ step {step}",
        )
        wandb.log({f"evaluate_plot/{idx}": wandb.Image(fig)}, step=step)
        plt.close(fig)

        ave_seq = sum(seq_agg["train/AVE"]) / len(seq_agg["train/AVE"]) if seq_agg["train/AVE"] else float("nan")
        fig2 = plot_trajectory_2d(
            pred_vel_w, gt_vel_w, t_2d_ns,
            title=f"{seq_label} @ step {step} — AVE {ave_seq:.3f} m/s, ATE {ate:.3f} m",
        )
        wandb.log({f"eval_traj_2d/{seq_label}": wandb.Image(fig2)}, step=step)
        plt.close(fig2)

    eval_metrics = {f"eval/{k.split('/', 1)[1]}": sum(v) / len(v) for k, v in agg.items() if v}
    eval_metrics["eval/ATE"]      = sum(ate_agg)      / len(ate_agg)
    eval_metrics["eval/ATE_mean"] = sum(ate_mean_agg) / len(ate_mean_agg)
    wandb.log(eval_metrics, step=step)
    return eval_metrics


if __name__ == "__main__":
    main_func()
