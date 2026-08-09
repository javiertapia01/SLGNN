"""Pérdidas de SLGNN-v3 (§13 de las instrucciones, §13.4-13.8 de la formulación).

Las pérdidas primarias supervisan **incrementos de momento**, adimensionalizados
por `P0` y `L0 P0`:

    L_dp = (1/N) sum_i || (dp_i,theta - dp_i^DEM) / P0 ||^2
    L_dL = (1/N) sum_i || (dL_i,theta - dL_i^DEM) / (L0 P0) ||^2

Los tensores ya llegan adimensionales desde `slgnn_experiments`, así que `P0`
vale 1 dentro del modelo; los factores quedan explícitos igualmente para que
un cambio de escala no pase inadvertido.

Regla de §13.2: **una propiedad garantizada exactamente por construcción no se
duplica como penalización.** La conservación de momento sale de `J^T` y la
complementariedad del solver; ambas se monitorean como diagnóstico. Solo
entran como término de pérdida cuando el solver *no* las satisface de forma
exacta, que es el caso de la penetración con discretización finita.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .contact_kinematics import ContactSet
from .integrator import StepResult


@dataclass
class LossWeights:
    """Pesos de la pérdida total. `dL` es informativa en el MVP normal."""

    delta_p: float = 1.0
    delta_L: float = 1.0
    position: float = 0.0
    rollout: float = 0.0
    penetration: float = 0.0
    complementarity: float = 0.0
    l2: float = 0.0

    # Escalas de adimensionalización; 1.0 si los datos ya vienen adimensionales.
    P0: float = 1.0
    LP0: float = 1.0
    L0: float = 1.0


@dataclass
class LossTerms:
    total: torch.Tensor
    parts: dict[str, float] = field(default_factory=dict)


def momentum_losses(
    result: StepResult,
    target_dp: torch.Tensor,
    target_dL: torch.Tensor,
    w: LossWeights,
    node_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(L_dp, L_dL)`, medias por partícula."""
    dp = (result.delta_p - target_dp) / w.P0
    dL = (result.delta_L - target_dL) / w.LP0
    if node_mask is not None:
        dp, dL = dp[node_mask], dL[node_mask]
    n = max(dp.shape[0], 1)
    return (dp**2).sum() / n, (dL**2).sum() / n


def position_loss(result: StepResult, target_q: torch.Tensor, w: LossWeights):
    dq = (result.next_state.particles.q - target_q) / w.L0
    return (dq**2).sum() / max(dq.shape[0], 1)


def penetration_loss(next_gap: torch.Tensor, w: LossWeights) -> torch.Tensor:
    """`relu(-g_{k+1})^2`. No está garantizada por construcción con paso
    finito, CCD aproximado o SDF inexacta, así que sí es una pérdida legítima."""
    if next_gap.numel() == 0:
        return next_gap.sum()
    return (((-next_gap).clamp_min(0.0) / w.L0) ** 2).mean()


def complementarity_loss(result: StepResult, w: LossWeights) -> torch.Tensor:
    """Residuo de complementariedad del solver.

    Se usa como **diagnóstico** cuando el solver converge; solo tiene sentido
    como término de aprendizaje si se detecta que no converge, y en ese caso
    el arreglo es el solver, no la pérdida.
    """
    lam = result.impulses
    if lam.numel() == 0:
        return lam.sum()
    neg = (-lam).clamp_min(0.0)
    return ((neg / w.P0) ** 2).mean()


def total_loss(
    result: StepResult,
    target_dp: torch.Tensor,
    target_dL: torch.Tensor,
    w: LossWeights,
    target_q: torch.Tensor | None = None,
    next_gap: torch.Tensor | None = None,
    parameters=None,
) -> LossTerms:
    l_dp, l_dL = momentum_losses(result, target_dp, target_dL, w)
    total = w.delta_p * l_dp + w.delta_L * l_dL
    parts = {"delta_p": float(l_dp.detach()), "delta_L": float(l_dL.detach())}

    if w.position and target_q is not None:
        l_q = position_loss(result, target_q, w)
        total = total + w.position * l_q
        parts["position"] = float(l_q.detach())
    if w.penetration and next_gap is not None:
        l_pen = penetration_loss(next_gap, w)
        total = total + w.penetration * l_pen
        parts["penetration"] = float(l_pen.detach())
    if w.complementarity:
        l_c = complementarity_loss(result, w)
        total = total + w.complementarity * l_c
        parts["complementarity"] = float(l_c.detach())
    if w.l2 and parameters is not None:
        l2 = sum((p**2).sum() for p in parameters)
        total = total + w.l2 * l2
        parts["l2"] = float(l2.detach())
    parts["total"] = float(total.detach())
    return LossTerms(total=total, parts=parts)


def rollout_loss(
    results: list[StepResult],
    q_targets: torch.Tensor,
    v_targets: torch.Tensor,
    omega_targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    lambda_q: float = 1.0,
    lambda_v: float = 1.0,
    lambda_w: float = 1.0,
) -> torch.Tensor:
    """Pérdida de rollout a horizonte `H`, con pesos por horizonte.

    Se conservan las curvas por horizonte fuera de aquí: una media agrupada
    esconde en qué paso empieza el fallo.
    """
    H = len(results)
    if weights is None:
        weights = torch.ones(H, dtype=results[0].delta_p.dtype)
    total = results[0].delta_p.sum() * 0.0
    for h, res in enumerate(results):
        p = res.next_state.particles
        total = total + weights[h] * (
            lambda_q * ((p.q - q_targets[h]) ** 2).mean()
            + lambda_v * ((p.v - v_targets[h]) ** 2).mean()
            + lambda_w * ((p.omega - omega_targets[h]) ** 2).mean()
        )
    return total


def next_gaps(result: StepResult) -> torch.Tensor:
    """Gaps recalculados en `q_{k+1}` para la pérdida de penetración."""
    cs: ContactSet | None = result.contacts
    if cs is None or cs.n_contacts == 0:
        return result.delta_p.new_zeros(0)
    p = result.next_state.particles
    pp = ~cs.is_wall
    out = []
    if bool(pp.any()):
        i, j = cs.i[pp], cs.j[pp]
        d = (p.q[j] - p.q[i]).norm(dim=-1)
        out.append(d - (p.radius[i] + p.radius[j]))
    return torch.cat(out) if out else result.delta_p.new_zeros(0)
