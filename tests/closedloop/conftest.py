"""Session fixtures for the closed-loop suite: ONE 2-step train run, reused by every test.

Closed loop means the tests drive the shipped entrypoints — they import
``train_ssio`` and ``evaluate`` and call their functions, so a defect in
either script surfaces here rather than in a re-implementation of it.

The whole suite costs ~5 s because the expensive halves happen once:

    trained_run   composes ``Config/train_ssio.yaml`` the way the CLI does, drives
                  ``train_ssio.main_func.__wrapped__(cfg)`` in-process for
                  ``TRAIN_STEPS`` steps, and leaves behind a directory that is a
                  legitimate ``evaluate.py`` target (``.hydra/config.yaml`` +
                  ``checkpoint_00002.pt``).
    evaluator     loads that checkpoint into a network built from the stored
                  ``cfg.network`` and exposes ``score()``, which is
                  ``evaluate.main``'s per-sequence body with the plotting removed:
                  ``run_sequence`` → ``integrate_trapezoid`` → ``compute_metrics``.

Why ``main_func.__wrapped__`` and not a subprocess: ``@hydra.main`` would parse
pytest's own ``sys.argv``. Reaching the plain ``(cfg) -> None`` underneath lets the
run share the interpreter (no torch re-import per test) and lets a traceback point
at the real defect. The two things ``@hydra.main`` normally supplies are supplied
here by hand: ``HydraConfig`` (``save_checkpoint_artifact`` and ``wandb.init`` read
``runtime.output_dir`` from it) and the ``.hydra/config.yaml`` dump.

Deliberate overrides, each load-bearing:

    hydra.searchpath                        adds ``tests/fixtures/config`` so the two
        smoke dataset groups resolve.
    dataset@train_dataset / @eval_dataset   the two smoke groups, i.e. the run is a
        pure config override of the production recipe — no test-only code path is
        exercised that a real run would not take.
    train.max_train_steps=2                 stops mid-epoch (the epoch is 10 windows
        at batch 4 = 3 batches) and forces a checkpoint save on the way out.
    meta.device=${as_device:'cpu'}          CPU keeps the run reproducible and
        independent of whether the host has a GPU; ``as_device`` would already
        demote ``cuda:0`` on a CPU-only host, so this only removes the *variation*.
    wandb.mode=disabled                     no network, no run directory outside tmp.

Everything else — network, optimizer, scheduler, loss, EMA — is left at the shipped
default on purpose, so the checkpoint under test is the checkpoint the default recipe
writes (EMA weights, via ``eval_model = ema_model.module``).
"""
import os
import types

import pytest

_HERE       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_HERE))
_TRAIN_CONFIG_DIR = os.path.join(_REPO_ROOT, "Config")            # holds train_ssio.yaml
# The smoke dataset groups are test-only, so they live under tests/fixtures/config/
# rather than in the Config/dataset/ that real experiments select from.
# _FIXTURE_CONFIG_DIR goes on Hydra's searchpath (see _SEARCHPATH) so that
# ``dataset@train_dataset=EuRoC_Smoke_Train`` resolves as an ordinary group.
_FIXTURE_CONFIG_DIR = os.path.join(os.path.dirname(_HERE), "fixtures", "config")
_DATASET_DIR        = os.path.join(_FIXTURE_CONFIG_DIR, "dataset")
_SEARCHPATH = f"hydra.searchpath=[file://{_FIXTURE_CONFIG_DIR}]"

#: Steps the smoke run takes. The checkpoint is named from the step counter, so this
#: is also the ``checkpoint_{step:05d}.pt`` the tests look for.
TRAIN_STEPS = 2
#: Batch size — small enough that the 10-window epoch is several batches, so the
#: ``max_train_steps`` cap really does interrupt an epoch rather than coincide with
#: its end.
TRAIN_BATCH = 4


def dataset_yaml(name: str) -> str:
    """Absolute path of a dataset group in ``tests/fixtures/config/dataset/``.

    ``evaluate.py --data`` takes a filesystem path rather than a Hydra group name,
    so the closed-loop tests need the path form as well as the group form.
    """
    path = os.path.join(_DATASET_DIR, f"{name}.yaml")
    if not os.path.isfile(path):
        pytest.fail(f"dataset config not found: {path}")
    return path


def network_yaml(name: str) -> str:
    """Absolute path of a *shipped* network group in ``Config/network/``.

    The same file the run itself composes from. A test that needs a network the trained run did not build — the
    ``airio_imu_only`` ablation twin — instantiates this YAML instead of passing
    constructor kwargs, so there is no hand-copied layout to drift from the config
    the experiments actually select.
    """
    path = os.path.join(_REPO_ROOT, "Config", "network", f"{name}.yaml")
    if not os.path.isfile(path):
        pytest.fail(f"network config not found: {path}")
    return path


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory, smoke) -> types.SimpleNamespace:
    """Run integration self-supervision training in-process for ``TRAIN_STEPS`` steps.

    Attributes:
        ``run_dir``    the stand-in Hydra run dir (a tmp dir), containing
                       ``.hydra/config.yaml`` and ``checkpoint_00002.pt``.
        ``cfg_path``   ``<run_dir>/.hydra/config.yaml`` — unresolved, exactly what
                       Hydra dumps, so ``evaluate`` can reload and re-resolve it.
        ``cfg``        the composed config the run actually consumed.
        ``steps``      ``TRAIN_STEPS``.
        ``epoch_windows``  length of one training epoch in windows, for asserting
                       that the step cap truncated it.
    """
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf, open_dict

    import train_ssio

    # train_ssio.py does not seed, and RandomWindow draws its window starts
    # from the GLOBAL torch RNG — so without this the checkpoint under test depends
    # on which other test files ran first (several call torch.manual_seed). Seed here
    # and the run is a function of the recipe alone, whatever the test selection.
    torch.manual_seed(20260824)

    run_dir = str(tmp_path_factory.mktemp("ssio_smoke_run"))

    with initialize_config_dir(config_dir=_TRAIN_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="train_ssio",
            overrides=[
                _SEARCHPATH,
                f"dataset@train_dataset={smoke.train_dataset}",
                f"dataset@eval_dataset={smoke.valid_dataset}",
                "train.epoch=1",
                f"train.batch={TRAIN_BATCH}",
                f"train.max_train_steps={TRAIN_STEPS}",
                # The fixture runs `debias: false`, so there is no corrected stream and
                # the default `corrected` would raise.
                "train.dv_imu_input=raw",
                "meta.device=${as_device:'cpu'}",
                "wandb.mode=disabled",
            ],
            return_hydra_config=True,
        )
        # @hydra.main would set this from hydra.run.dir; the script reads it in
        # save_checkpoint_artifact() and wandb.init(dir=...).
        OmegaConf.update(cfg, "hydra.runtime.output_dir", run_dir)
        hydra_singleton = HydraConfig.instance()
        previous = hydra_singleton.cfg
        hydra_singleton.set_config(cfg)
        try:
            # @hydra.main strips the hydra node before calling the task function,
            # and the script's wandb.init() does OmegaConf.to_container(resolve=True)
            # on whatever it is handed — keep the shape identical.
            with open_dict(cfg):
                del cfg["hydra"]
            os.makedirs(os.path.join(run_dir, ".hydra"), exist_ok=True)
            OmegaConf.save(cfg, os.path.join(run_dir, ".hydra", "config.yaml"))
            train_ssio.main_func.__wrapped__(cfg)
        finally:
            # Do not leak a configured Hydra singleton into the rest of the session.
            hydra_singleton.cfg = previous

    return types.SimpleNamespace(
        run_dir=run_dir,
        cfg_path=os.path.join(run_dir, ".hydra", "config.yaml"),
        cfg=cfg,
        steps=TRAIN_STEPS,
        epoch_windows=len(smoke.load(smoke.train_dataset, window=int(cfg.train.window))[0]),
    )


@pytest.fixture(scope="session")
def evaluator(trained_run, smoke) -> types.SimpleNamespace:
    """``evaluate.py``'s scoring path, wired to the checkpoint ``trained_run`` wrote.

    Attributes:
        ``ev``          the imported ``evaluate`` module (so tests reach ``summarize``,
                        ``parse_args`` … without re-importing).
        ``cfg``         the reloaded run config, with ``eval_dataset`` swapped through
                        ``evaluate.apply_data_override`` — the same call ``--data`` makes.
        ``ckpt_path``   what ``evaluate.pick_checkpoint`` selects with no override.
        ``network``     that checkpoint, loaded strictly into a fresh ``cfg.network``.
        ``sequence``    the single eval ``ISequence`` (reused; ``run_sequence`` is
                        read-only and provably repeatable on one instance).
        ``batch``       the collated ``IMUData_w_GT`` that ``run_sequence``'s own
                        DataLoader hands the network — the data contract under test.
        ``build_sequence(overrides=None)``  a fresh instance, for tests that must rule
                        out per-instance caching as the source of a repeatable answer;
                        ``overrides`` makes it a *variant* of the same stored node.
        ``load_batch(seq=None)``  ``batch`` for any sequence.
        ``use_aux_imu`` / ``use_joints``  the run's own ``modality`` flags, read the way
                        ``evaluate.main`` reads them and forwarded to ``run_sequence``.
        ``build_network()``   a second network loaded from the same checkpoint.
        ``run(seq=None, net=None)``    ``evaluate.run_sequence``'s 7-tuple.
        ``score(seq=None, net=None)``  the metric dict for one sequence, i.e.
                        ``run_sequence`` → ``integrate_trapezoid`` → ``compute_metrics``,
                        exactly as ``evaluate.main`` composes them.
    """
    import copy

    import torch
    from omegaconf        import OmegaConf
    from torch.utils.data import DataLoader

    import evaluate as ev
    from Interface          import instantiate
    from Interface.data     import IMUData_w_GT, collate_dataclass
    from Interface.sequence import ISequence
    from Model.Network      import INetwork
    from Utils.modality      import apply_modality_mask
    from Utils.trainer      import integrate_trapezoid

    # No OmegaConf.resolve(): ${meta.device} resolves to a torch.device, which is not
    # an OmegaConf primitive. evaluate.main() loads it the same lazy way.
    cfg = OmegaConf.load(trained_run.cfg_path)
    ev.apply_data_override(cfg, dataset_yaml(smoke.valid_dataset))
    ckpt_path = ev.pick_checkpoint(trained_run.run_dir, None)
    gravity = float(OmegaConf.select(cfg, "train.gravity", default=9.81007))
    # Off the run's own config, as evaluate.main() does — never hard-coded here.
    dv_imu_input = str(cfg.train.dv_imu_input)
    # ``run_sequence`` takes the two modality flags as REQUIRED keyword args and calls
    # ``apply_modality_mask`` with them; evaluate.main() reads them with ``select`` +
    # default so a run dir saved by an older config without the ``modality`` node
    # still scores. Mirrored here rather than passed as literals, so an arm that
    # flips one (``modality.use_aux_imu=false``) is exercised the way the script sees it.
    use_aux_imu = bool(OmegaConf.select(cfg, "modality.use_aux_imu", default=True))
    use_joints  = bool(OmegaConf.select(cfg, "modality.use_joints",  default=True))

    def build_network():
        net = instantiate(INetwork, cfg.network).to(cfg.meta.device, cfg.meta.dtype)
        net.load_state_dict(torch.load(ckpt_path, map_location=cfg.meta.device))
        net.eval()
        return net

    def build_sequence(overrides: dict | None = None):
        """The stored eval node, instantiated again — optionally as a variant.

        ``overrides`` maps dotted config paths to values and is applied to a *copy* of
        the run's own config before instantiation. That keeps a variant (another
        ``train.window``, another ``meta.dtype``, distinct per-sequence sigmas) a read
        of the shipped node resolving its own interpolations, rather than a hand-built
        OmegaConf that could disagree with what a real run composes.
        """
        node_cfg = cfg
        if overrides:
            node_cfg = copy.deepcopy(cfg)
            for path, value in overrides.items():
                OmegaConf.update(node_cfg, path, value, merge=False)
        return instantiate(ISequence[IMUData_w_GT], node_cfg.eval_dataset.sequences[0])

    def load_batch(seq=None):
        """The collated batch ``run_sequence`` feeds the network, via the same loader.

        Identical DataLoader kwargs to ``evaluate.run_sequence``, so tests assert on the object the shipped eval path really builds — including
        the ``collate_dataclass`` step, which is where a per-field concat could
        misalign the batch axis. Every eval sequence here has far fewer than 256
        windows, so the first batch is the whole sequence.

        ``apply_modality_mask`` is applied on the way out, because ``run_sequence``
        applies it to every batch before the network sees it. With the
        shipped ``modality`` defaults (both flags true) it returns the batch unchanged, so
        this only keeps the fixture honest if an arm ever flips a flag — where dropping it
        would leave the tests asserting a K axis the network never receives.
        """
        seq = seq if seq is not None else sequence
        assert len(seq) <= 256, (
            f"{len(seq)} windows: no longer one DataLoader batch, so load_batch() "
            "would silently return a prefix of the sequence"
        )
        return apply_modality_mask(
            next(iter(DataLoader[IMUData_w_GT](
                seq, batch_size=256, shuffle=False, collate_fn=collate_dataclass,
            ))),
            use_aux_imu=use_aux_imu, use_joints=use_joints,
        )

    network  = build_network()
    sequence = build_sequence()

    def run(seq=None, net=None):
        return ev.run_sequence(
            net if net is not None else network,
            seq if seq is not None else sequence,
            use_aux_imu=use_aux_imu,
            use_joints=use_joints,
            gravity=gravity,
            imu_input=dv_imu_input,
        )

    def score(seq=None, net=None):
        pred_w, gt_w, _gt_rot_w, t_ns_w, *_preint = run(seq=seq, net=net)
        pred = pred_w.reshape(-1, 3)
        gt   = gt_w  .reshape(-1, 3)
        t_ns = t_ns_w.reshape(-1)
        return ev.compute_metrics(
            pred, gt, integrate_trapezoid(pred, t_ns), integrate_trapezoid(gt, t_ns), t_ns,
        )

    return types.SimpleNamespace(
        ev=ev, cfg=cfg, ckpt_path=ckpt_path, network=network, sequence=sequence,
        batch=load_batch(), gravity=gravity,
        use_aux_imu=use_aux_imu, use_joints=use_joints,
        build_network=build_network, build_sequence=build_sequence, load_batch=load_batch,
        run=run, score=score,
    )
