"""LR scheduler helpers for SSIO trainers.

Schedulers are hydra-instantiated from ``Config/scheduler/*.yaml`` directly in
the trainers (``instantiate(LRScheduler, cfg.scheduler, optimizer=optimizer)``,
same idiom as the optimizer). The default ``constant`` choice is a
``ConstantLR(factor=1.0)`` no-op, so a real scheduler object always exists —
no None branching. Schedulers are stepped once per epoch after validation.

Checkpoints in this pipeline are weights-only (``save_checkpoint_artifact``
does not persist optimizer or scheduler state). On resume, the scheduler
restarts from its initial config — not from the pre-interruption step count.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


def scheduler_step_epoch(
    scheduler: LRScheduler,
    eval_loss: float | None,
) -> None:
    """Advance the scheduler at epoch boundary.

    ``ReduceLROnPlateau`` requires ``eval_loss`` (typically ``eval/loss``).
    All other epoch schedulers (e.g. ``CosineAnnealingLR``) call ``step()`` with no metric.
    """
    if isinstance(scheduler, ReduceLROnPlateau):
        assert eval_loss is not None
        scheduler.step(eval_loss)
    else:
        scheduler.step()


def regularization_schedule_weight(epoch, total_epochs, *, start, end, enable=True):
    """Global stationary-regularization schedule s(epoch) in [0,1], once per epoch (0-indexed epoch).
    full-ON (1) for epoch<start; LINEAR 1->0 over [start,end]; OFF (0) for epoch>=end.
    start/end: value in [0,1] = FRACTION of total_epochs; value>1 = ABSOLUTE epoch index.
    Disabled schedules return constant weight."""
    if not enable:
        return 1.0
    if start is None or end is None:
        raise ValueError("enabled regularization schedule requires both start and end")
    start_e = start*total_epochs if start <= 1.0 else float(start)
    end_e   = end*total_epochs   if end   <= 1.0 else float(end)
    if not (0.0 <= start_e <= end_e):
        raise ValueError(f"regularization schedule needs 0<=start<=end; got {start_e},{end_e}")
    if epoch < start_e: return 1.0
    if epoch >= end_e:  return 0.0
    return 1.0 - (epoch - start_e)/(end_e - start_e)
