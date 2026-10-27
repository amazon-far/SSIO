"""Pose graph distillation (PGD) trainer: trajectory-level PGO pseudo-labels.

For every training trajectory, the network predicts velocity over stitched
windows, and one pose graph per trajectory chains those predictions with
main-IMU preintegration factors. The graph's posterior velocities are cached per
anchor, and training windows are fit to them. By default the graph is solved
once before training (``train.label_refresh_every_steps: null``); a positive
value re-solves it with the current network every that many steps.

Shares knob names, EMA and checkpoint policy with ``train_ssio.py``.
"""
import typing as tp
from dataclasses import dataclass, fields

import wandb
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import pypose as pp
import jaxtyping as jt
import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot
import matplotlib.pyplot as plt

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from hydra import main
from hydra.core.hydra_config import HydraConfig
from beartype import beartype
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from Interface          import instantiate, validate_hydra_config
from Interface.sequence import ISequence
from Interface.data     import IMUData_w_GT, IMUJointsData_w_GT, collate_dataclass
from Interface.model    import INetwork, get_est_velocity
from Interface.graph    import I_IMUFactorGraph
from Interface.loss     import I_Adapt_Loss
from Sequence.SlidingWindow import SlidingWindow
from Sequence.TagSequence   import TagSequence
from Utils.checkpoint   import load_resume_checkpoint, save_best_model, save_checkpoint_artifact
from Utils.config       import ADAPT_CONFIG_SPEC
from Utils.lr_schedule  import scheduler_step_epoch
from Utils.modality     import apply_modality_mask
from Utils.plots        import plot_trajectory_2d, plot_velocity_series
from Utils.trainer      import metric_ATE, optimize_step


# The trainer does its own windowing on the stride grid, so dataset-config window
# wrappers are stripped; value-modifying wrappers are kept.
WINDOW_WRAPPERS = ("SlidingWindow", "RandomWindow")


def unwrap_windows(seq: ISequence) -> ISequence:
    while type(seq).__name__ in WINDOW_WRAPPERS and hasattr(seq, "internal_seq"):
        seq = seq.internal_seq
    return seq


@main(version_base=None, config_path="./Config", config_name="train_pgd")
@validate_hydra_config(ADAPT_CONFIG_SPEC)
def main_func(cfg):
    wandb.init(
        project = cfg.wandb.project,
        mode    = cfg.wandb.mode,
        group   = "pgd",
        name    = cfg.wandb.name,
        config  = OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        dir     = HydraConfig.get().runtime.output_dir,       # → <hydra_dir>/wandb/run-…
    )

    use_aux_imu = cfg.modality.use_aux_imu
    use_joints  = cfg.modality.use_joints

    # Sequences may return IMUData_w_GT (IMU-only) or IMUJointsData_w_GT (with joints);
    # apply_modality_mask normalises the type after loading.
    train_raw = [
        unwrap_windows(instantiate(ISequence[IMUJointsData_w_GT], seq_cfg))
        for seq_cfg in cfg.train_dataset.sequences
    ]
    eval_raw = [
        unwrap_windows(instantiate(ISequence[IMUJointsData_w_GT], seq_cfg))
        for seq_cfg in cfg.eval_dataset.sequences
    ]

    loss_fn   = instantiate(I_Adapt_Loss, cfg.loss)
    window    = cfg.train.window
    step_size = cfg.train.step_size
    refresh_every = cfg.train.label_refresh_every_steps    # None: solve once, before training
    batch_windows = cfg.train.label_batch_windows
    # Network factors get one fixed isotropic std: the self-supervised network's
    # covariance head is never trained, so its prediction is not a usable weight.
    obs_sigma     = cfg.train.network_factor_sigma
    dv_imu_input  = cfg.train.dv_imu_input
    checkpoint_every_epochs = cfg.train.checkpoint_every_epochs
    eval_plot_every_epochs  = cfg.train.eval_plot_every_epochs
    max_train_steps = cfg.train.max_train_steps

    network    = instantiate(INetwork, cfg.network).train().to(cfg.meta.device, cfg.meta.dtype)
    if cfg.train.resume_from:
        load_resume_checkpoint(cfg.train.resume_from, network, cfg.meta.device)
    optimizer  = instantiate(torch.optim.Optimizer, cfg.optimizer, params=network.parameters())
    scheduler  = instantiate(torch.optim.lr_scheduler.LRScheduler, cfg.scheduler, optimizer=optimizer)
    integrator = pp.module.IMUPreintegrator(prop_cov=True, reset=True, gravity=cfg.train.gravity).to(cfg.meta.device, cfg.meta.dtype)

    stride = network.stride
    if step_size % stride != 0:
        raise ValueError(
            f"train.step_size={step_size} must be a multiple of the network stride "
            f"({stride}) so that window anchors land on the trajectory anchor grid."
        )

    # Student windows, tagged with their trajectory so labels can be looked up.
    student_sequences = [
        TagSequence(
            OmegaConf.create({"seq_id": i}),
            SlidingWindow(OmegaConf.create({"window_size": window, "step_size": step_size}), raw),
        )
        for i, raw in enumerate(train_raw)
    ]
    train_loader = DataLoader[IMUData_w_GT](
        ConcatDataset(student_sequences),
        batch_size=cfg.train.batch, shuffle=True, pin_memory=True, collate_fn=collate_dataclass,
    )

    # Weight average (EMA), an implementation detail: it predicts for the label
    # solves and is what gets evaluated and checkpointed. See train_ssio.py for the decay.
    ema_decay = 0.8
    ema_model: AveragedModel | None = (
        AveragedModel(network, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
        if cfg.train.ema else None
    )
    eval_model = tp.cast(INetwork, ema_model.module if ema_model is not None else network)
    bank = TrajectoryLabelBank(train_raw)

    try:
        step = 0
        stop = False

        init_metrics = evaluate(step, loss_fn, eval_model, integrator, eval_raw, cfg.graph, window,
                                batch_windows=batch_windows, obs_sigma=obs_sigma, dv_imu_input=dv_imu_input,
                                use_aux_imu=use_aux_imu, use_joints=use_joints, plots=False)
        best_ave  = init_metrics["eval/AVE"]
        best_loss = init_metrics["eval/loss"]
        save_best_model("best_ave_model.pt", eval_model, "best_eval_AVE",  best_ave)
        save_best_model("best_model.pt",     eval_model, "best_eval_loss", best_loss)

        for epoch in range(cfg.train.epoch):
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(f"{epoch:03d}", total=len(train_loader))
                for data in train_loader:
                    if step == 0 or (refresh_every is not None and step % refresh_every == 0):
                        bank.refresh(eval_model, integrator, cfg.graph, window, step=step,
                                     batch_windows=batch_windows, obs_sigma=obs_sigma, dv_imu_input=dv_imu_input,
                                     use_aux_imu=use_aux_imu, use_joints=use_joints)
                    step += 1
                    data = apply_modality_mask(data, use_aux_imu=use_aux_imu, use_joints=use_joints)
                    loss, metrics = train_step(network, loss_fn, bank, data)
                    loss.backward()
                    optimize_step(optimizer, ema_model, network)
                    if step % 10 == 0: wandb.log(metrics, step=step)
                    progress.update(task, advance=1)
                    if max_train_steps is not None and step >= max_train_steps:
                        stop = True
                        break

            # Metrics every epoch (the scheduler needs eval/loss); figures only every
            # eval_plot_every_epochs, and always on the final epoch.
            epoch_next = epoch + 1
            want_plots = (
                epoch_next % eval_plot_every_epochs == 0
                or epoch_next == cfg.train.epoch
                or stop
            )
            eval_metrics = evaluate(step, loss_fn, eval_model, integrator, eval_raw, cfg.graph, window,
                                    batch_windows=batch_windows, obs_sigma=obs_sigma, dv_imu_input=dv_imu_input,
                                    use_aux_imu=use_aux_imu, use_joints=use_joints, plots=want_plots)
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
            # Periodic, plus always the last epoch — else a final partial interval
            # (epoch=100, every=30) is discarded. Matches train_ssio.py.
            if epoch_next % checkpoint_every_epochs == 0 or epoch_next == cfg.train.epoch or stop:
                save_checkpoint_artifact(step, eval_model)
            if stop:
                break
    finally:
        wandb.finish()


@jt.jaxtyped(typechecker=beartype)
@dataclass
class IMUPreintegratorResult:
    dv     : jt.Float[torch.Tensor, "B N 3"]
    dr     : jt.Float[torch.Tensor, "B N 3"]
    dt     : jt.Float[torch.Tensor, "B N 1"] # ms
    dvr_cov: jt.Float[torch.Tensor, "B N 6 6"]


def run_preintegrator(
    network  : INetwork,
    model    : pp.module.IMUPreintegrator,
    imu      : IMUData_w_GT,
    imu_input: str = "corrected",
) -> IMUPreintegratorResult:
    """Per-anchor-segment preint in segment-start body frame; returns
    (dv, dr, dt, dvr_cov) shaped (B, M, ...) with slot M-1 zero-padded.

    Anchors are at frames ``s_idx + k*stride``; slot k describes segment
    anchor_k -> anchor_{k+1}. dr is the relative rotation from ``gt_poses``, i.e.
    always GT — ``WindowIntegratedOrientation`` writes only ``input_rot`` and
    leaves ``gt_poses`` untouched, so it does NOT make this frame GT-free.
    dvr_cov is the [rot, vel] block of pypose's 9x9. Main-body IMU (K index 0)
    only, as in ``get_ref_delta_v`` and ``evaluate.py``; K>1 is legal and is indexed,
    not fused, so aux IMUs stay out of the PGO factors even when the network
    consumes them.
    """

    if imu_input == "corrected":
        acc = tp.cast(torch.Tensor, imu.corrected_acc)[:, :, 0]
        gyro = tp.cast(torch.Tensor, imu.corrected_gyro)[:, :, 0]
    else:
        assert imu_input == "raw"
        acc, gyro = imu.acc[:, :, 0], imu.gyro[:, :, 0]                               # (B, N, 3)

    s_idx  = network.start_index
    stride = network.stride

    time_ns  = imu.time_ns[:, :, 0]                                                  # (B, N, 1) int64
    R_wb     = imu.gt_poses.rotation()                                               # (B, N) SO3, body-level (no K)

    B, N   = time_ns.shape[0], time_ns.shape[1]
    dtype, device = acc.dtype, acc.device

    dt_per_step = (time_ns[:, 1:] - time_ns[:, :-1]).to(dtype) * 1.0e-9

    # Need M-1 segments of length stride starting at frame s_idx.
    n_intervals_total = N - 1
    M_minus_1 = (n_intervals_total - s_idx) // stride
    assert M_minus_1 >= 1

    # Slice the per-step axis: keep [s_idx, s_idx + (M-1)*stride).
    seg_slice = slice(s_idx, s_idx + M_minus_1 * stride)
    dt_seg   = dt_per_step[:, seg_slice]                                              # (B, (M-1)*stride, 1)
    acc_seg  = acc        [:, seg_slice]                                              # (B, (M-1)*stride, 3)
    gyro_seg = gyro       [:, seg_slice]                                              # (B, (M-1)*stride, 3)
    R_wb_seg = R_wb       [:, seg_slice]                                              # (B, (M-1)*stride) SO3

    dt_chunk   = dt_seg.reshape(B * M_minus_1, stride, 1)
    acc_chunk  = acc_seg.reshape(B * M_minus_1, stride, 3)
    gyro_chunk = gyro_seg.reshape(B * M_minus_1, stride, 3)
    R_chunk_raw = pp.SO3(R_wb_seg).tensor().reshape(B * M_minus_1, stride, 4)
    R_chunk     = pp.SO3(R_chunk_raw)

    # init_state per (batch, segment): R_wb(anchor_k), zero v, zero p.
    R_anchor_raw = R_chunk_raw[:, 0:1]
    R_anchor     = pp.SO3(R_anchor_raw)
    BM = B * M_minus_1
    init_state = {
        "rot": R_anchor,
        "vel": torch.zeros(BM, 1, 3, dtype=dtype, device=device),
        "pos": torch.zeros(BM, 1, 3, dtype=dtype, device=device),
    }

    preint_chunk = model(
        dt=dt_chunk, gyro=gyro_chunk, acc=acc_chunk,
        rot=R_chunk, init_state=init_state,
    )

    # Final-step state; Dv into anchor-k body frame.
    Dv_world = preint_chunk["vel"][:, -1, :]
    cov_end  = preint_chunk["cov"]                                                    # 9x9 ordered [rot, vel, pos]
    R_anchor_flat = pp.SO3(R_anchor_raw.squeeze(1))
    Dv_body_k = R_anchor_flat.Inv() @ Dv_world
    Dt_total  = dt_chunk.sum(dim=1, keepdim=False)

    # [rot, vel] block; rotate the vel sub-block into anchor-k body frame (rot block is frame-invariant).
    dvr_cov_6 = cov_end[:, :6, :6]
    R_anc_mat_T = R_anchor_flat.matrix().transpose(-1, -2)
    T_rotvel = torch.zeros_like(dvr_cov_6)
    T_rotvel[:, 0:3, 0:3] = torch.eye(3, dtype=dtype, device=device)
    T_rotvel[:, 3:6, 3:6] = R_anc_mat_T
    dvr_cov_6 = T_rotvel @ dvr_cov_6 @ T_rotvel.transpose(-1, -2)

    # Unfold the segment axis. Dr is the exact relative rotation between consecutive
    # anchors from gt_poses, so it needs no preintegration at all.
    Dv_seg = Dv_body_k.reshape(B, M_minus_1, 3)
    anchor_idx = torch.arange(s_idx, s_idx + M_minus_1 * stride + 1, stride, device=device)
    R_anchors  = pp.SO3(R_wb.tensor()[:, anchor_idx])                                 # (B, M) body->world
    Dr_seg     = (R_anchors[:, :-1].Inv() @ R_anchors[:, 1:]).Log().tensor()
    Dt_seg  = Dt_total .reshape(B, M_minus_1, 1)    * 1.0e3                            # s -> ms
    Cov_seg = dvr_cov_6.reshape(B, M_minus_1, 6, 6)

    eye6 = torch.eye(6, dtype=dtype, device=device).expand(B, 1, 6, 6)
    return IMUPreintegratorResult(
        dv      = torch.cat([Dv_seg, torch.zeros_like(Dv_seg[:, :1])], dim=1),
        dr      = torch.cat([Dr_seg, torch.zeros_like(Dr_seg[:, :1])], dim=1),
        dt      = torch.cat([Dt_seg, torch.zeros_like(Dt_seg[:, :1])], dim=1),
        dvr_cov = torch.cat([Cov_seg, eye6],                    dim=1),
    )


# ====== Trajectory labels ======


@dataclass
class TrajectoryLabels:
    """Per-anchor series of one trajectory; anchor j is frame start_index + j*stride."""
    seq_name    : str
    time_ns     : torch.Tensor        # (M,)   int64, IMU 0
    rot_so3     : torch.Tensor        # (M, 3) so3 tangent of GT R_{w<-b} at the anchors
    gt_vel_w    : torch.Tensor        # (M, 3)
    gt_vel_b    : torch.Tensor        # (M, 3)
    net_vel_b   : torch.Tensor        # (M, 3)    teacher prediction (body)
    net_cov_b   : torch.Tensor        # (M, 3, 3) teacher covariance (body)
    target_vel_b: torch.Tensor        # (M, 3)    PGO posterior mean (body)
    info        : dict                # solver info


def stitched_window_starts(n_frames: int, window: int, network: INetwork) -> list[int]:
    """Window starts whose anchors tile the trajectory grid without gaps.

    Consecutive starts are ``stride * anchor_count(window)`` apart, so window
    ``w+1``'s first anchor is one stride after window ``w``'s last. The final
    start is pulled back onto the stride grid so the tail is covered by an
    overlapping window; overlapping anchors take the later window's prediction.
    """
    stride = network.stride
    if n_frames < window:
        raise ValueError(f"trajectory has {n_frames} frames < window={window}")
    gen_step = stride * network.anchor_count(window)
    starts = list(range(0, n_frames - window + 1, gen_step))
    last = ((n_frames - window) // stride) * stride
    if last > starts[-1]:
        starts.append(last)
    return starts


def _gather_frames(data: IMUData_w_GT, idx: torch.Tensor) -> IMUData_w_GT:
    """Windows ``idx`` (W, window) of a (1, N, ...) trajectory; per-batch tags are dropped."""
    kwargs = {}
    for f in fields(data):
        v = getattr(data, f.name)
        if isinstance(v, pp.LieTensor):
            kwargs[f.name] = pp.LieTensor(v.as_subclass(torch.Tensor)[0][idx], ltype=v.ltype)
        elif isinstance(v, torch.Tensor):
            kwargs[f.name] = v[0][idx] if v.dim() >= 2 else None
        else:
            kwargs[f.name] = v
    return type(data)(**kwargs)


def _slice_frames(data: IMUData_w_GT, n: int) -> IMUData_w_GT:
    """The first ``n`` frames of a (1, N, ...) trajectory."""
    kwargs = {}
    for f in fields(data):
        v = getattr(data, f.name)
        if isinstance(v, pp.LieTensor):
            kwargs[f.name] = pp.LieTensor(v.as_subclass(torch.Tensor)[:, :n], ltype=v.ltype)
        elif isinstance(v, torch.Tensor) and v.dim() >= 2:
            kwargs[f.name] = v[:, :n]
        else:
            kwargs[f.name] = v
    return type(data)(**kwargs)


@torch.no_grad()
def stitched_forward(
    network: INetwork,
    data   : IMUData_w_GT,          # (1, N, ...) full trajectory
    window : int,
    *,
    batch_windows: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Teacher ``(vel_b, cov_b, M)`` on the covered trajectory anchor grid."""
    N = data.acc.shape[1]
    stride = network.stride
    starts = stitched_window_starts(N, window, network)
    M_win  = network.anchor_count(window)
    M      = starts[-1] // stride + M_win            # anchors covered

    vel = data.acc.new_zeros(M, 3)
    cov = data.acc.new_zeros(M, 3, 3)
    filled = torch.zeros(M, dtype=torch.bool, device=data.acc.device)
    for b0 in range(0, len(starts), batch_windows):
        chunk = starts[b0:b0 + batch_windows]
        idx = torch.stack([torch.arange(s, s + window, device=data.acc.device) for s in chunk])  # (W, window)
        out = get_est_velocity(network, _gather_frames(data, idx))
        for w, s in enumerate(chunk):
            j0 = s // stride
            vel[j0:j0 + M_win] = out.vel_b[w]
            cov[j0:j0 + M_win] = out.vel_cov_b[w]
            filled[j0:j0 + M_win] = True
    assert bool(filled.all()), "stitched forward left anchors uncovered — window tiling bug"
    return vel, cov, M


@torch.no_grad()
def solve_trajectory(
    network   : INetwork,
    integrator: pp.module.IMUPreintegrator,
    graph_cfg,
    raw_seq   : ISequence[IMUData_w_GT],
    window    : int,
    *,
    batch_windows: int,
    obs_sigma    : float,
    dv_imu_input : str,
    use_aux_imu  : bool,
    use_joints   : bool,
) -> TrajectoryLabels:
    """Solve one pose graph over a whole trajectory from stitched teacher predictions.

    The graph is anchored on the first anchor's GT rotation and velocity. Every
    network velocity factor uses the isotropic std ``obs_sigma``.
    """
    network.eval()
    data = apply_modality_mask(raw_seq[0:len(raw_seq)], use_aux_imu=use_aux_imu, use_joints=use_joints)
    stride, s_idx = network.stride, network.start_index

    net_vel, net_cov, M = stitched_forward(network, data, window, batch_windows=batch_windows)

    # Trim to exactly M anchors so preintegration and downsampling agree with the forward.
    data_t = _slice_frames(data, s_idx + (M - 1) * stride + 1)
    preint = run_preintegrator(network, integrator, data_t, imu_input=dv_imu_input)

    rot_so3  = network.downsample(data_t.gt_poses.rotation().Log())[0]          # (M, 3)
    gt_vel_w = network.downsample(data_t.gt_vel_w)[0]                            # (M, 3)
    gt_vel_b = pp.so3(rot_so3).Exp().Inv() @ gt_vel_w
    time_ns  = network.downsample(data_t.time_ns[:, :, 0])[0, :, 0]              # (M,) int64

    obs_cov = (obs_sigma ** 2) * torch.eye(3, dtype=net_cov.dtype, device=net_cov.device).expand_as(net_cov)
    graph = instantiate(
        I_IMUFactorGraph, graph_cfg,
        net_vel_b     = net_vel[None],
        net_vel_cov_b = obs_cov[None],
        imu_dv_b      = preint.dv,
        imu_dr_b      = preint.dr,
        imu_dt_ms     = preint.dt,
        imu_dvr_cov   = preint.dvr_cov,
        rot_wb        = pp.so3(rot_so3[None, 0:1]),
        vel_prior     = gt_vel_b[None, 0:1],
    )
    info = graph.solve()
    vel  = graph.get_state[0]

    # A failed or non-finite solve falls back to the teacher for the whole
    # trajectory, so it reads as zero residual rather than reaching the optimizer
    # as NaN. An unconverged but finite solve is kept and reported.
    bad = (not bool(torch.isfinite(vel).all())) or bool(info["failed"].any())
    if info.get("rejected"):
        print(f"[PGO:{raw_seq.seq_name}] chi2 gate rejected {info['rejected']}/{M} velocity observations", flush=True)
    if bad:
        print(f"[PGO:{raw_seq.seq_name}] solve non-finite/failed — falling back to the teacher prediction", flush=True)
        vel = net_vel.clone()
    if not info["converged"]:
        print(f"[PGO:{raw_seq.seq_name}] not converged: iters={info['iters']} loss={info['final_loss']:.4g}", flush=True)

    return TrajectoryLabels(
        seq_name     = raw_seq.seq_name,
        time_ns      = time_ns,
        rot_so3      = rot_so3,
        gt_vel_w     = gt_vel_w,
        gt_vel_b     = gt_vel_b,
        net_vel_b    = net_vel,
        net_cov_b    = net_cov,
        target_vel_b = vel,
        info         = {
            "iters": int(info["iters"]), "converged": bool(info["converged"]),
            "rejected": int(info.get("rejected", 0)), "fallback": bool(bad),
        },
    )


class TrajectoryLabelBank:
    """Cached trajectory labels, looked up by ``(seq_id, window_start)``."""

    def __init__(self, sequences: list[ISequence[IMUData_w_GT]]):
        self.sequences = sequences
        self.labels: list[TrajectoryLabels] = []

    def refresh(
        self,
        teacher   : INetwork,
        integrator: pp.module.IMUPreintegrator,
        graph_cfg,
        window    : int,
        *,
        step         : int,
        batch_windows: int,
        obs_sigma    : float,
        dv_imu_input : str,
        use_aux_imu  : bool,
        use_joints   : bool,
    ) -> None:
        self.labels = [
            solve_trajectory(teacher, integrator, graph_cfg, seq, window,
                             batch_windows=batch_windows, obs_sigma=obs_sigma, dv_imu_input=dv_imu_input,
                             use_aux_imu=use_aux_imu, use_joints=use_joints)
            for seq in self.sequences
        ]
        def mean(xs): return sum(xs) / len(xs)
        d = [lab.net_vel_b - lab.target_vel_b for lab in self.labels]
        # Residuals are teacher vs PGO, not errors; only the AVE_* entries use GT.
        wandb.log({
            "label/pgo_vel_residual":    mean([x.norm(dim=-1).mean().item() for x in d]),
            "label/pgo_vel_residual_dc": mean([x.mean(dim=0).norm().item() for x in d]),
            "label/AVE_target":  mean([(l.target_vel_b - l.gt_vel_b).norm(dim=-1).mean().item() for l in self.labels]),
            "label/AVE_teacher": mean([(l.net_vel_b    - l.gt_vel_b).norm(dim=-1).mean().item() for l in self.labels]),
            "label/pgo_iters":     mean([l.info["iters"] for l in self.labels]),
            "label/n_rejected":    float(sum(l.info["rejected"] for l in self.labels)),
            "label/n_fallback":    float(sum(l.info["fallback"] for l in self.labels)),
            "label/n_unconverged": float(sum(not l.info["converged"] for l in self.labels)),
        }, step=step)

    def lookup(self, data: IMUData_w_GT, network: INetwork) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached body-frame (target velocity, GT velocity) for each student window."""
        assert data.seq_id is not None and data.window_start is not None, \
            "student batches must carry seq_id / window_start (TagSequence + SlidingWindow)"
        stride = network.stride
        M_win  = network.anchor_count(data.acc.shape[1])
        B = data.acc.shape[0]
        tv = data.acc.new_empty(B, M_win, 3)
        gv = data.acc.new_empty(B, M_win, 3)
        for b in range(B):
            s = int(data.window_start[b])
            if s % stride != 0:
                raise AssertionError(f"window_start={s} is off the stride grid ({stride})")
            lab = self.labels[int(data.seq_id[b])]
            j0 = s // stride
            if j0 + M_win > lab.target_vel_b.shape[0]:
                raise AssertionError(
                    f"window at {s} of {lab.seq_name} reaches past the cached labels "
                    f"({j0 + M_win} > {lab.target_vel_b.shape[0]})"
                )
            tv[b] = lab.target_vel_b[j0:j0 + M_win]
            gv[b] = lab.gt_vel_b    [j0:j0 + M_win]
        return tv, gv


def train_step(
    network: INetwork,
    loss_fn: I_Adapt_Loss,
    bank   : TrajectoryLabelBank,
    data   : IMUData_w_GT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit one batch of student windows to the cached trajectory labels."""
    network.train()  # sole owner of the mode flag — evaluate() leaves the net in eval
    out = get_est_velocity(network, data)
    target_vel, gt_vel_b = bank.lookup(data, network)
    if out.vel_b.shape[1] != target_vel.shape[1]:
        raise AssertionError(f"anchor-count disagreement: net {out.vel_b.shape[1]} vs labels {target_vel.shape[1]}")
    loss, metrics = loss_fn(out.vel_b, target_vel, gt_vel_b)
    metrics["train/AVE"] = (out.vel_b.detach() - gt_vel_b).norm(dim=-1).mean().item()
    return loss, metrics


@torch.no_grad()
def evaluate(
    step: int,
    loss: I_Adapt_Loss,
    model: INetwork,
    integrator: pp.module.IMUPreintegrator,
    eval_raw: list[ISequence[IMUData_w_GT]],
    graph_cfg,
    window: int,
    *,
    batch_windows: int,
    obs_sigma    : float,
    dv_imu_input : str,
    use_aux_imu  : bool,
    use_joints   : bool,
    plots        : bool = True,
) -> dict[str, float]:
    """Solve each eval trajectory as in training and score the teacher.

    ``eval/loss`` is the teacher-vs-PGO fit (label-free, used for model selection
    and the scheduler); ``eval/AVE`` and ``eval/ATE`` measure the teacher against GT.
    """
    model.eval()
    agg: dict[str, list[float]] = {"train/loss": [], "train/AVE": [], "train/pgo_vel_residual": []}
    ate_agg     : list[float] = []
    ate_mean_agg: list[float] = []

    for idx, seq in enumerate(eval_raw):
        lab = solve_trajectory(model, integrator, graph_cfg, seq, window,
                               batch_windows=batch_windows, obs_sigma=obs_sigma, dv_imu_input=dv_imu_input,
                               use_aux_imu=use_aux_imu, use_joints=use_joints)
        _, metrics = loss(lab.net_vel_b[None], lab.target_vel_b[None], lab.gt_vel_b[None])
        metrics["train/AVE"] = (lab.net_vel_b - lab.gt_vel_b).norm(dim=-1).mean().item()
        for k in agg:
            agg[k].append(metrics[k])

        R = pp.so3(lab.rot_so3).Exp()
        pred_vel_w = (R @ lab.net_vel_b).cpu()
        gt_vel_w   = lab.gt_vel_w.cpu()
        t_ns       = lab.time_ns.cpu()
        ate, ate_mean = metric_ATE(pred_vel_w, gt_vel_w, t_ns)
        ate_agg.append(ate)
        ate_mean_agg.append(ate_mean)
        wandb.log({f"eval_seq/AVE/{lab.seq_name}": metrics["train/AVE"],
                   f"eval_seq/ATE/{lab.seq_name}": ate}, step=step)

        if not plots:
            continue
        Rm = R.matrix()
        std_w = torch.diagonal(Rm @ lab.net_cov_b @ Rm.transpose(-1, -2), dim1=-2, dim2=-1).clamp_min(0).sqrt()
        t_s = ((t_ns - t_ns[0]).to(torch.float64) * 1e-9).numpy()
        fig = plot_velocity_series(
            t_s, gt_vel_w.numpy(), pred_vel_w.numpy(), None, std_w.cpu().numpy(), None,
            title=f"Eval sequence {lab.seq_name} @ step {step}",
        )
        wandb.log({f"evaluate_plot/{idx}": wandb.Image(fig)}, step=step)
        plt.close(fig)
        fig2 = plot_trajectory_2d(pred_vel_w, gt_vel_w, t_ns,
                                  title=f"{lab.seq_name} @ step {step} — ATE {ate:.3f} m")
        wandb.log({f"eval_traj_2d/{lab.seq_name}": wandb.Image(fig2)}, step=step)
        plt.close(fig2)

    eval_metrics = {f"eval/{k.split('/', 1)[1]}": sum(v) / len(v) for k, v in agg.items() if v}
    eval_metrics["eval/ATE"]      = sum(ate_agg)      / len(ate_agg)
    eval_metrics["eval/ATE_mean"] = sum(ate_mean_agg) / len(ate_mean_agg)
    wandb.log(eval_metrics, step=step)
    return eval_metrics


if __name__ == "__main__":
    main_func()
