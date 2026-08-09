"""Bucle común de entrenamiento y evaluación.

El mismo runner conduce SLGNN-v3 y los baselines GNS. Que sea literalmente el
mismo código es lo que hace comparable el resultado: mismo sampler, mismos
targets, mismo presupuesto de actualizaciones, mismo criterio de selección de
checkpoint (validación, nunca test) y misma medición de tiempo.

Un modelo compatible expone:

    step(state, dt, surfaces, gravity=None, eval_mode=False, create_graph=None)
        -> objeto con `delta_p`, `delta_L`, `next_state`, `diagnostics`
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .contact_labels import RegimeLabels, label_transitions
from .data import Trajectory
from .metrics import (
    ContactMetrics, RegimeAccumulator, RolloutCurve, reachable_decomposition,
)
from .sampling import StratifiedSampler, TransitionIndex, build_index
from .scene import Scene, active_contact_keys, active_contact_normals
from .targets import TransitionTargets, build_targets


class StepModel(Protocol):
    def step(self, state, dt, surfaces, gravity=None, eval_mode=False,
             create_graph=None): ...
    def reset_lifecycle(self) -> None: ...
    def n_parameters(self) -> tuple[int, int]: ...


@dataclass
class TrainConfig:
    updates: int = 200
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 50
    log_every: int = 10
    seed: int = 0
    lambda_delta_p: float = 1.0
    lambda_delta_L: float = 0.0     # informativa en el MVP normal
    rollout_horizon: int = 0
    lambda_rollout: float = 0.0


@dataclass
class Dataset:
    """Trayectorias con sus targets y etiquetas, ya construidos una vez."""

    trajectories: list[Trajectory]
    targets: list[TransitionTargets]
    labels: list[RegimeLabels]
    index: TransitionIndex

    @staticmethod
    def build(trajectories: list[Trajectory], scene: Scene,
              high_compression: float = 0.05) -> "Dataset":
        from .data import DATASETS

        spec = DATASETS[scene.dataset_key]
        box_min = box_max = None
        if spec.geometry == "box":
            box_min = [scene.scales.length(x) for x in spec.box_min]
            box_max = [scene.scales.length(x) for x in spec.box_max]
        labels = [label_transitions(t, box_min, box_max) for t in trajectories]
        targets = [build_targets(t) for t in trajectories]
        return Dataset(trajectories, targets, labels,
                       build_index(trajectories, labels, high_compression))


def _batch_loss(model, scene: Scene, data: Dataset, picks, cfg: TrainConfig,
                training: bool):
    """Pérdida de un batch. Cada transición es un sistema independiente."""
    items = [(data.trajectories[t], k) for t, k in picks]
    state = scene.batch_at(items)
    target_dp = torch.cat([data.targets[t].delta_p[k] for t, k in picks])
    target_dL = torch.cat([data.targets[t].delta_L[k] for t, k in picks])

    model.reset_lifecycle()
    res = model.step(state, scene.dt, scene.surfaces, scene.gravity,
                     eval_mode=not training, create_graph=training)
    n = max(target_dp.shape[0], 1)
    l_dp = ((res.delta_p - target_dp) ** 2).sum() / n
    l_dL = ((res.delta_L - target_dL) ** 2).sum() / n
    loss = cfg.lambda_delta_p * l_dp + cfg.lambda_delta_L * l_dL
    return loss, {"delta_p": float(l_dp.detach()), "delta_L": float(l_dL.detach()),
                  "total": float(loss.detach())}, res


def train(
    model: StepModel,
    scene: Scene,
    train_data: Dataset,
    val_data: Dataset | None,
    sampler: StratifiedSampler,
    cfg: TrainConfig,
    run_dir=None,
    on_best=None,
) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    best = math.inf
    best_update = -1
    history: list[dict] = []
    t0 = time.perf_counter()
    examples = 0

    for update in range(1, cfg.updates + 1):
        model.train()
        picks = sampler.sample(cfg.batch_size)
        examples += len(picks)
        opt.zero_grad(set_to_none=True)
        loss, parts, _ = _batch_loss(model, scene, train_data, picks, cfg, True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        record = {"update": update, "phase": "train", **parts,
                  "elapsed_s": time.perf_counter() - t0}
        if update % cfg.log_every == 0 or update == 1:
            history.append(record)
            if run_dir is not None:
                run_dir.log_metrics(record)

        if val_data is not None and (update % cfg.eval_every == 0 or update == cfg.updates):
            v = validate(model, scene, val_data, cfg)
            vrec = {"update": update, "phase": "val", **v,
                    "elapsed_s": time.perf_counter() - t0}
            history.append(vrec)
            if run_dir is not None:
                run_dir.log_metrics(vrec)
            if v["total"] < best:
                best, best_update = v["total"], update
                if on_best is not None:
                    on_best(update, v)

    return {
        "best_val_loss": best if math.isfinite(best) else None,
        "best_update": best_update,
        "final_train_loss": history[-1].get("total") if history else None,
        "updates": cfg.updates,
        "examples_seen": examples,
        "train_seconds": time.perf_counter() - t0,
        "history": history,
    }


@torch.no_grad()
def _no_grad_guard():
    return None


def validate(model: StepModel, scene: Scene, data: Dataset, cfg: TrainConfig,
             max_batches: int = 8) -> dict[str, float]:
    """Validación determinista: un barrido fijo, no muestreado al azar."""
    model.eval()
    n = len(data.index)
    stride = max(1, n // (max_batches * cfg.batch_size))
    picks_all = data.index.items[::stride][: max_batches * cfg.batch_size]
    if not picks_all:
        return {"total": float("nan"), "delta_p": float("nan"), "delta_L": float("nan")}
    acc = {"total": 0.0, "delta_p": 0.0, "delta_L": 0.0}
    n_batches = 0
    for s in range(0, len(picks_all), cfg.batch_size):
        picks = picks_all[s: s + cfg.batch_size]
        loss, parts, _ = _batch_loss(model, scene, data, picks, cfg, False)
        for k in acc:
            acc[k] += parts[k]
        n_batches += 1
    return {k: v / n_batches for k, v in acc.items()}


def evaluate_one_step(
    model: StepModel, scene: Scene, data: Dataset, stride: int = 1,
    max_transitions: int | None = None,
) -> dict[str, float]:
    """Métricas de un paso, desglosadas por régimen del nodo."""
    model.eval()
    acc_dp, acc_dL = RegimeAccumulator(), RegimeAccumulator()
    contact = ContactMetrics()
    split_reach = split_orth = 0.0
    n_split = 0
    picks = data.index.items[::stride]
    if max_transitions:
        picks = picks[:max_transitions]
    t0 = time.perf_counter()
    for t_idx, k in picks:
        tr = data.trajectories[t_idx]
        state = scene.state_at(tr, k)
        model.reset_lifecycle()
        res = model.step(state, scene.dt, scene.surfaces, scene.gravity,
                         eval_mode=True, create_graph=False)
        reg = data.labels[t_idx].node_regime[k]
        acc_dp.add(res.delta_p.detach(), data.targets[t_idx].delta_p[k], reg)
        acc_dL.add(res.delta_L.detach(), data.targets[t_idx].delta_L[k], reg)
        # Separación normal/tangencial del error, medida igual para todos los
        # modelos: la componente ortogonal al span de las normales activas es
        # inalcanzable para cualquier canal puramente normal, así que reportarla
        # como "fallo del modelo" sería atribuir mal el residual (§16.4, §16.5).
        nrm = active_contact_normals(state, scene.surfaces)
        if nrm is not None:
            err = (res.delta_p.detach() - data.targets[t_idx].delta_p[k])
            d = reachable_decomposition(err, nrm[0], nrm[1], state.particles.n)
            split_reach += d["reachable_fraction"]
            split_orth += d["unreachable_fraction"]
            n_split += 1
        # Contactos predichos en k+1 contra contactos reales en k+1.
        t_next = state.time_scalar() + scene.dt
        pred_keys, pred_gaps = active_contact_keys(
            res.next_state.particles.detach(), scene.surfaces, t_next
        )
        true_particles = scene.state_at(tr, k + 1).particles
        true_keys, _ = active_contact_keys(true_particles, scene.surfaces, t_next)
        contact.add_keys(pred_keys, true_keys)
        contact.add_gaps(pred_gaps, tunnel_threshold=float(tr.radius.min()))
    if n_split:
        out_split = {
            "err_normal_fraction": split_reach / n_split,
            "err_tangential_fraction": split_orth / n_split,
            "n_split_frames": n_split,
        }
    else:
        out_split = {}
    out = {f"dp_{k}": v for k, v in acc_dp.result().items()}
    out.update({f"dL_{k}": v for k, v in acc_dL.result().items()})
    out.update(contact.result())
    out.update(out_split)
    out["eval_seconds"] = time.perf_counter() - t0
    out["eval_transitions"] = len(picks)
    out["seconds_per_step"] = out["eval_seconds"] / max(len(picks), 1)
    return out


def evaluate_rollout(
    model: StepModel, scene: Scene, data: Dataset, horizons: list[int],
    starts: list[tuple[int, int]] | None = None, n_starts: int = 8,
) -> dict[str, float]:
    """Rollout autoregresivo a varios horizontes. Conserva la curva completa."""
    model.eval()
    H = max(horizons)
    if starts is None:
        candidates = [
            (t, k) for t, k in data.index.items
            if k + H < data.trajectories[t].n_steps
        ]
        if not candidates:
            return {}
        stride = max(1, len(candidates) // n_starts)
        starts = candidates[::stride][:n_starts]

    curve = RolloutCurve(horizons=sorted(horizons))
    for t_idx, k0 in starts:
        tr = data.trajectories[t_idx]
        state = scene.state_at(tr, k0)
        model.reset_lifecycle()
        failed_at = None
        for h in range(1, H + 1):
            # Sin `no_grad`: la fuerza conservativa de v3 es `-grad_q V` y
            # necesita el grafo de autograd aunque se esté evaluando. El estado
            # se **desconecta** entre pasos, que es lo que acota la memoria en
            # un rollout largo; envolver todo en `no_grad` simplemente rompe V.
            res = model.step(state, scene.dt, scene.surfaces, scene.gravity,
                             eval_mode=True, create_graph=False)
            p = res.next_state.particles.detach()
            state = type(res.next_state)(p, time=res.next_state.time,
                                         memory=getattr(res.next_state, "memory", None))
            if not torch.isfinite(p.q).all():
                failed_at = h
                break
            if h in curve.horizons:
                with torch.no_grad():
                    curve.add(
                        h,
                        float(torch.sqrt(((p.q - tr.q[k0 + h]) ** 2).mean())),
                        float(torch.sqrt(((p.v - tr.v[k0 + h]) ** 2).mean())),
                        float(torch.sqrt(((p.omega - tr.omega[k0 + h]) ** 2).mean())),
                    )
        curve.nan_step.append(failed_at if failed_at else H + 1)
    out = curve.result()
    out["rollout_starts"] = len(starts)
    return out
