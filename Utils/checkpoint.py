"""Shared checkpoint loading and saving for SSIO training scripts.

One ``train.resume_from`` URI warm-starts either training stage, with three
schemes: ``file:<path>`` (bare ``state_dict`` from ``save_checkpoint_artifact``),
``wandb:<entity>/<project>/checkpoint_<run>:<alias>`` (the artifact must hold
exactly one ``.pt``, per that same one-file-per-version convention), and
``airio:<path>`` (an AirIO checkpoint dict keyed ``model_state_dict``). All
three load with ``strict=True`` — the architecture must match exactly.
"""

import glob
import os

import torch
import wandb
from hydra.core.hydra_config import HydraConfig


def load_resume_checkpoint(spec: str, model: torch.nn.Module, device) -> None:
    """Load weights into ``model`` from the URI-style ``spec`` (see module docstring)."""
    if ":" not in spec:
        raise ValueError(
            f"train.resume_from must be 'file:<path>', 'wandb:<artifact>', or "
            f"'airio:<path>', got {spec!r}"
        )
    scheme, _, target = spec.partition(":")

    if scheme == "file":
        ckpt_path = os.path.expanduser(target)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"resume_from file not found: {ckpt_path}")
        _load_bare_state_dict(ckpt_path, model, device, label=scheme, target=target)

    elif scheme == "wandb":
        artifact  = wandb.Api().artifact(target)
        local_dir = artifact.download()
        pt_files  = glob.glob(os.path.join(local_dir, "*.pt"))
        if len(pt_files) != 1:
            raise RuntimeError(
                f"wandb artifact {target!r} must contain exactly one .pt file; "
                f"found {len(pt_files)}: {pt_files}"
            )
        _load_bare_state_dict(pt_files[0], model, device, label=scheme, target=target)

    elif scheme == "airio":
        ckpt_path = os.path.expanduser(target)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"resume_from file not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"airio:{target}: expected a checkpoint dict with 'model_state_dict'; "
                f"got {type(checkpoint).__name__}"
                + (f" with keys {sorted(checkpoint.keys())}" if isinstance(checkpoint, dict) else "")
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[resume] loaded airio:{target} from {ckpt_path}")

    else:
        raise ValueError(
            f"Unknown resume_from scheme {scheme!r}; expected 'file', 'wandb', or 'airio'."
        )


def _load_bare_state_dict(
    ckpt_path: str, model: torch.nn.Module, device, *, label: str, target: str,
) -> None:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        raise RuntimeError(
            f"{label}:{target}: checkpoint is a training bundle with "
            f"'model_state_dict', not a bare state_dict. Use "
            f"train.resume_from='airio:{target}' instead."
        )
    model.load_state_dict(state_dict, strict=True)
    print(f"[resume] loaded {label}:{target} ({len(state_dict)} tensors) from {ckpt_path}")


def save_checkpoint_artifact(step: int, model: torch.nn.Module) -> None:
    """Save a local weights snapshot and a versioned WandB artifact."""
    assert wandb.run is not None, "wandb must be initialised before save_checkpoint_artifact()"

    state_dict = model.state_dict()
    hydra_dir = HydraConfig.get().runtime.output_dir
    torch.save(state_dict, os.path.join(hydra_dir, f"checkpoint_{step:05d}.pt"))

    wandb_path = os.path.join(wandb.run.dir, f"{step:05d}.pt")
    os.makedirs(os.path.dirname(wandb_path), exist_ok=True)
    torch.save(state_dict, wandb_path)

    artifact = wandb.Artifact(name=f"checkpoint_{wandb.run.id}", type="model")
    artifact.add_file(wandb_path)
    wandb.log_artifact(artifact)


def save_best_model(
    filename: str,
    model: torch.nn.Module,
    summary_key: str,
    value: float,
) -> None:
    """Overwrite a best-of checkpoint and record its metric."""
    hydra_dir = HydraConfig.get().runtime.output_dir
    torch.save(model.state_dict(), os.path.join(hydra_dir, filename))
    if wandb.run is not None:
        wandb.run.summary[summary_key] = float(value)
