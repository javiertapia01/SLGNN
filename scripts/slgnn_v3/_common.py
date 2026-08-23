"""Utilidades compartidas por los scripts de v3: configuración, modelo, escena."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gns_baseline import GNSConfig, GNSControlled  # noqa: E402
from slgnn_experiments.checkpointing import (  # noqa: E402
    RunDirectory, RunManifest, git_state, index_hash, make_run_id,
)
from slgnn_experiments.nondimensionalization import default_scales  # noqa: E402
from slgnn_experiments.runner import Dataset, TrainConfig  # noqa: E402
from slgnn_experiments.sampling import DEFAULT_QUOTAS, StratifiedSampler  # noqa: E402
from slgnn_experiments.scene import build_scene  # noqa: E402
from slgnn_experiments.splits import load_split, split_from_config  # noqa: E402
from slgnn_v3 import SLGNNv3, V3Config  # noqa: E402

MODEL_KEYS = ("slgnn_v3", "gns_controlled")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_model(model_key: str, model_cfg: dict, dtype=torch.float64):
    """Construye el modelo y devuelve `(modelo, config_resuelta, perfil)`."""
    if model_key == "slgnn_v3":
        cfg = V3Config.from_dict(model_cfg)
        return SLGNNv3(cfg).to(dtype), cfg.to_dict(), cfg.router.profile.value
    if model_key == "gns_controlled":
        cfg = GNSConfig.from_dict(model_cfg)
        return GNSControlled(cfg).to(dtype), cfg.to_dict(), "gns-controlled"
    raise ValueError(f"Modelo desconocido {model_key!r}. Conocidos: {MODEL_KEYS}")


def prepare(experiment: dict, seed: int, smoke: bool = False):
    """Carga datos, construye escena, dataset, sampler y configuración de train."""
    scales = default_scales()
    split = split_from_config(experiment["data"])
    max_steps = experiment["data"].get("max_steps")
    if smoke and experiment["data"].get("frame_stop") is None:
        max_steps = min(max_steps or 60, 60)
    frame_start = int(experiment["data"].get("frame_start", 0))
    frame_stop = experiment["data"].get("frame_stop")
    if smoke and frame_stop is not None:
        frame_stop = min(frame_stop, frame_start + 60)
    loaded = load_split(split, REPO_ROOT, scales, max_steps=max_steps,
                        frame_start=frame_start, frame_stop=frame_stop)
    scene = build_scene(split.dataset, scales)

    train_ds = Dataset.build(loaded.train, scene)
    val_ds = Dataset.build([loaded.val], scene) if loaded.val is not None else None
    test_ds = Dataset.build(loaded.test, scene) if loaded.test else None

    quotas = dict(experiment.get("sampler", {}).get("quotas", DEFAULT_QUOTAS))
    sampler = StratifiedSampler(train_ds.index, quotas, seed=seed)

    tcfg_raw = dict(experiment.get("train", {}))
    if smoke:
        tcfg_raw.update(updates=4, eval_every=2, log_every=1,
                        batch_size=min(int(tcfg_raw.get("batch_size", 4)), 2))
    tcfg = TrainConfig(seed=seed, **tcfg_raw)
    return scales, split, loaded, scene, train_ds, val_ds, test_ds, sampler, tcfg


def make_manifest(*, run_id, model_key, profile, seed, split, train_ds, sampler,
                  tcfg, scales, model, experiment_name, notes="") -> RunManifest:
    sha, dirty = git_state(REPO_ROOT)
    total, trainable = model.n_parameters()
    solver = getattr(getattr(model, "cfg", None), "solver", None)
    return RunManifest(
        run_id=run_id, model=model_key, profile=profile, seed=seed,
        dataset=split.dataset, train_cases=list(split.train_cases),
        val_case=split.val_case, test_cases=list(split.test_cases),
        n_transitions_available=len(train_ds.index),
        index_hash=index_hash(train_ds.index.items),
        scales=scales.as_dict(), n_parameters=total, n_trainable=trainable,
        budget_updates=tcfg.updates,
        budget_examples=tcfg.updates * tcfg.batch_size,
        sampler_quotas=dict(sampler.effective_quotas),
        sampler_composition=sampler.epoch_composition(tcfg.batch_size, tcfg.updates),
        solver_backend=("fista-projected+closed-form-scalar" if solver else "n/a"),
        solver_tolerances=(
            {"tol": solver.tol, "max_iters": solver.max_iters,
             "eval_max_iters": solver.eval_max_iters, "beta": solver.beta}
            if solver else {}
        ),
        git_sha=sha, git_dirty=dirty, notes=notes,
    )
