"""Métricas por régimen y agregación por semilla (§17).

Regla de §22.15: **no se reporta solo RMSE global.** Toda métrica de estado se
desglosa por régimen (`free`, `pp`, `pw`, `mixed`) y las curvas por horizonte
se conservan enteras: una media agrupada esconde en qué paso empieza a fallar
el rollout.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .contact_labels import CATEGORY_NAMES


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return float("nan")
    return float(torch.sqrt(((a - b) ** 2).mean()))


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return float("nan")
    return float((a - b).abs().mean())


@dataclass
class RegimeAccumulator:
    """Acumula errores por nodo separados por régimen de ese nodo."""

    sq: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    ab: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    n: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, pred: torch.Tensor, target: torch.Tensor,
            node_regime: torch.Tensor) -> None:
        err = (pred - target).detach()
        for code, name in enumerate(CATEGORY_NAMES):
            m = node_regime == code
            if not bool(m.any()):
                continue
            e = err[m]
            self.sq[name] += float((e**2).sum())
            self.ab[name] += float(e.abs().sum())
            self.n[name] += int(e.numel())

    def result(self) -> dict[str, float]:
        out: dict[str, float] = {}
        tot_sq = tot_ab = 0.0
        tot_n = 0
        for name in CATEGORY_NAMES:
            k = self.n.get(name, 0)
            if k:
                out[f"rmse_{name}"] = math.sqrt(self.sq[name] / k)
                out[f"mae_{name}"] = self.ab[name] / k
                out[f"n_{name}"] = k
                tot_sq += self.sq[name]
                tot_ab += self.ab[name]
                tot_n += k
        if tot_n:
            out["rmse_all"] = math.sqrt(tot_sq / tot_n)
            out["mae_all"] = tot_ab / tot_n
            out["n_all"] = tot_n
        return out


@dataclass
class ContactMetrics:
    """Precisión/recall de contacto, penetración y tunneling.

    La comparación es entre los contactos **predichos en `k+1`** —los que
    resultan del estado que el modelo produce— y los contactos **reales en
    `k+1`** según el DEM. Comparar el conjunto de contactos del modelo consigo
    mismo daría precisión 1 siempre y no mediría nada.

    Si nunca se acumula un frame, las métricas salen `NaN`, no cero: un modelo
    que no expone geometría de contacto no tiene precisión cero, tiene
    precisión indefinida.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    min_gap: float = float("inf")
    max_penetration: float = 0.0
    tunneling_events: int = 0
    n_frames: int = 0

    def add_keys(self, predicted: set, actual: set) -> None:
        self.tp += len(predicted & actual)
        self.fp += len(predicted - actual)
        self.fn += len(actual - predicted)
        self.n_frames += 1

    def add_gaps(self, gaps: torch.Tensor, tunnel_threshold: float) -> None:
        if gaps.numel() == 0:
            return
        g = float(gaps.min())
        self.min_gap = min(self.min_gap, g)
        self.max_penetration = max(self.max_penetration, max(-g, 0.0))
        self.tunneling_events += int((gaps < -tunnel_threshold).sum())

    def result(self) -> dict[str, float]:
        if self.n_frames == 0:
            nan = float("nan")
            return {"contact_precision": nan, "contact_recall": nan,
                    "contact_f1": nan, "min_gap": nan, "max_penetration": nan,
                    "tunneling_events": nan}
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec == prec and rec == rec and prec + rec > 0 else float("nan"))
        return {
            "contact_precision": prec,
            "contact_recall": rec,
            "contact_f1": f1,
            "min_gap": self.min_gap if math.isfinite(self.min_gap) else float("nan"),
            "max_penetration": self.max_penetration,
            "tunneling_events": float(self.tunneling_events),
        }


@dataclass
class RolloutCurve:
    """Error por horizonte. Se conserva la curva completa, no su media."""

    horizons: list[int]
    q_rmse: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    v_rmse: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    w_rmse: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    nan_step: list[int] = field(default_factory=list)

    def add(self, h: int, q: float, v: float, w: float) -> None:
        self.q_rmse[h].append(q)
        self.v_rmse[h].append(v)
        self.w_rmse[h].append(w)

    def result(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for h in self.horizons:
            if self.q_rmse.get(h):
                out[f"rollout_q_rmse_h{h}"] = statistics.fmean(self.q_rmse[h])
                out[f"rollout_v_rmse_h{h}"] = statistics.fmean(self.v_rmse[h])
                out[f"rollout_w_rmse_h{h}"] = statistics.fmean(self.w_rmse[h])
        if self.nan_step:
            # `nan_step` guarda `H+1` como centinela de "no falló". Se expone
            # también la fracción de rollouts que llegaron enteros, porque una
            # media de centinelas se lee mal: "26" no son 26 pasos hasta el
            # NaN, es que nunca lo hubo con H = 25.
            horizon = max(self.horizons) if self.horizons else 0
            failed = [x for x in self.nan_step if x <= horizon]
            out["rollout_finite_fraction"] = 1.0 - len(failed) / len(self.nan_step)
            out["steps_to_nan_mean"] = (
                statistics.fmean(failed) if failed else float("nan")
            )
            out["steps_to_nan_sentinel"] = statistics.fmean(self.nan_step)
        return out


def reachable_decomposition(
    target: torch.Tensor,
    normals: torch.Tensor,
    node_of_contact: torch.Tensor,
    n_nodes: int,
    tol: float = 1e-10,
) -> dict[str, float]:
    """Descompone un target por nodo en la parte **alcanzable** por un modelo
    de contacto puramente normal y la parte que no lo es.

    Un canal normal solo puede aplicar a la partícula `i` una fuerza en
    `span{n_alpha : alpha incidente a i}`. La componente de `Delta p` ortogonal
    a ese subespacio es inalcanzable **para cualquier** modelo normal, por bien
    entrenado que esté: es fricción tangencial que el MVP no implementa.

    Devuelve la energía relativa de cada parte. Es una cota independiente del
    modelo: separa "no se ha aprendido" de "no se puede representar".
    """
    reach_sq = 0.0
    orth_sq = 0.0
    total_sq = float((target**2).sum())
    for i in range(n_nodes):
        m = node_of_contact == i
        t = target[i]
        if not bool(m.any()):
            orth_sq += float((t**2).sum())
            continue
        # base ortonormal del span de las normales incidentes
        basis, _ = torch.linalg.qr(normals[m].T)
        keep = basis.abs().sum(dim=0) > tol
        basis = basis[:, keep]
        proj = basis @ (basis.T @ t)
        reach_sq += float((proj**2).sum())
        orth_sq += float(((t - proj) ** 2).sum())
    return {
        "reachable_fraction": reach_sq / max(total_sq, 1e-30),
        "unreachable_fraction": orth_sq / max(total_sq, 1e-30),
        "reachable_norm": math.sqrt(reach_sq),
        "unreachable_norm": math.sqrt(orth_sq),
        "target_norm": math.sqrt(total_sq),
    }


def aggregate_seeds(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Media, desviación estándar y valores por semilla, para cada métrica.

    §17: los agregados deben incluir las tres cosas. Una media sola no permite
    juzgar si una diferencia entre arquitecturas es real.
    """
    keys = sorted({k for d in per_seed for k in d})
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [d[k] for d in per_seed if k in d and not math.isnan(d[k])]
        if not vals:
            continue
        entry = {"mean": statistics.fmean(vals), "n_seeds": len(vals)}
        entry["std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        for i, v in enumerate(vals):
            entry[f"seed_{i}"] = v
        out[k] = entry
    return out
